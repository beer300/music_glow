import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim  # Import optimizer
import random  # Import random for seed setting

import time
from tqdm import tqdm 
import VQ_VAE_2_1D_data 
from VQ_VAE_2_1D_data import create_dataloader
from scipy.io.wavfile import write  # For saving audio
class ResidualBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0) # 1x1 equivalent


        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        #print(f"ResidualBlock1D input shape: {x.shape}")
        residual = self.skip(x)
        out = F.relu(x)
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = out + residual
        #print(f"ResidualBlock1D output shape: {out.shape}")
        return out



class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs):
        #print(f"VectorQuantizer input shape: {inputs.shape}")
        # Input shape: (B, C, L) for 1D or (B, C, H, W) for 2D
        # Ensure C == embedding_dim
        input_dims = len(inputs.shape)
        if input_dims == 3: # 1D case (B, C, L)
            # Permute to (B, L, C)
            inputs_permuted = inputs.permute(0, 2, 1).contiguous()
        elif input_dims == 4: # 2D case (B, C, H, W)
             # Permute to (B, H, W, C)
             inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()
        else:
            raise ValueError(f"Input tensor dimensions {input_dims} not supported (expected 3 or 4).")

        input_shape = inputs_permuted.shape
        # Flatten input to (B*L or B*H*W, C)
        #print(f"VectorQuantizer permuted input shape: {inputs_permuted.shape}")
        flat_input = inputs_permuted.view(-1, self.embedding_dim)
        #print(f"VectorQuantizer flat input shape: {flat_input.shape}")
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        #print(f"VectorQuantizer distances shape: {distances.shape}")
        # Find the closest embedding vector indices
        # shape: (B*L,) or (B*H*W,)
        encoding_indices = torch.argmin(distances, dim=1)
        #print(f"VectorQuantizer encoding indices shape: {encoding_indices.shape}")
        # Quantize
        quantized = self.embedding(encoding_indices) # Shape: (B*L, C) or (B*H*W, C)
        #print(f"VectorQuantizer quantized shape: {quantized.shape}")
        quantized = quantized.view(input_shape) # Reshape back to (B, L, C) or (B, H, W, C)
        #print(f"VectorQuantizer reshaped quantized shape: {quantized.shape}")
        # Loss Calculation
        e_latent_loss = F.mse_loss(quantized.detach().view(flat_input.shape), flat_input)
        q_latent_loss = F.mse_loss(quantized.view(flat_input.shape), flat_input.detach())
        vq_loss = e_latent_loss + self.commitment_cost * q_latent_loss
        #print(f"VectorQuantizer q_latent_loss shape: {q_latent_loss.shape}")
        vq_loss = e_latent_loss + self.commitment_cost * q_latent_loss
        #print(f"VectorQuantizer vq_loss shape: {vq_loss.shape}")
        # Straight-Through Estimator
        quantized_st = inputs_permuted + (quantized - inputs_permuted).detach()

        # Permute back to (B, C, L) or (B, C, H, W)
        if input_dims == 3:
             quantized_st = quantized_st.permute(0, 2, 1).contiguous()
             # indices shape: (B, L)
             indices_reshaped = encoding_indices.view(input_shape[:-1])
        else: # input_dims == 4
             quantized_st = quantized_st.permute(0, 3, 1, 2).contiguous()
             # indices shape: (B, H, W)
             indices_reshaped = encoding_indices.view(input_shape[:-1])

        #print(f"VectorQuantizer quantized shape: {quantized_st.shape}")
        return quantized_st, vq_loss, indices_reshaped

    def get_codebook_entry(self, indices):
        # indices shape: (B, L) or (B, H, W) or (N,)
        batch_size = indices.shape[0]
        indices_flatten = indices.reshape(-1) # (B*L,) or (B*H*W,) or (N,)
        quantized = self.embedding(indices_flatten) # (B*L, C) or (B*H*W, C) or (N, C)

        # Reshape if needed
        if len(indices.shape) >= 2: # Multi-dimensional indices
            C = quantized.shape[-1]
            spatial_shape = indices.shape[1:] # (L,) or (H, W)
            quantized = quantized.view(batch_size, *spatial_shape, C)
            if len(indices.shape) == 2: # 1D case (B, L) -> (B, L, C) -> (B, C, L)
                quantized = quantized.permute(0, 2, 1).contiguous()
            elif len(indices.shape) == 3: # 2D case (B, H, W) -> (B, H, W, C) -> (B, C, H, W)
                 quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized



class Encoder1D(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_res_blocks, res_channels, downsample_factor=2, num_downsample_layers_top=2):
        super().__init__()
        ks = downsample_factor * 2  # Kernel size often related to stride

        # --- Bottom Level Encoder ---
        bottom_layers = []
        bottom_layers.append(nn.Conv1d(in_channels, hidden_channels // 2, kernel_size=ks, stride=downsample_factor, padding=ks // 2 - downsample_factor // 2))  # Halve length
        bottom_layers.append(nn.ReLU(inplace=True))
        bottom_layers.append(nn.Conv1d(hidden_channels // 2, hidden_channels, kernel_size=ks, stride=downsample_factor, padding=ks // 2 - downsample_factor // 2))  # Halve length again
        bottom_layers.append(nn.ReLU(inplace=True))
        bottom_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))  # Adjust channels

        for _ in range(num_res_blocks):
            bottom_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels))
        bottom_layers.append(nn.ReLU(inplace=True))  # Activation before VQ projection
        self.encoder_b = nn.Sequential(*bottom_layers)

        # --- Medium Level Encoder ---
        medium_layers = []
        medium_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))  # Adjust channels
        for _ in range(num_res_blocks):
            medium_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels))
        medium_layers.append(nn.ReLU(inplace=True))  # Activation before passing to top encoder
        self.encoder_m = nn.Sequential(*medium_layers)

        # --- Top Level Encoder ---
        top_layers = []
        current_channels = hidden_channels
        for i in range(num_downsample_layers_top):
            out_ch = hidden_channels  # Keep hidden_channels for simplicity
            top_layers.append(nn.Conv1d(current_channels, out_ch, kernel_size=ks, stride=downsample_factor, padding=ks // 2 - downsample_factor // 2))
            top_layers.append(nn.ReLU(inplace=True))
            current_channels = out_ch

        top_layers.append(nn.Conv1d(current_channels, hidden_channels, kernel_size=3, stride=1, padding=1))  # Adjust channels before ResBlocks
        for _ in range(num_res_blocks):
            top_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels))
        top_layers.append(nn.ReLU(inplace=True))  # Activation before VQ projection
        self.encoder_t = nn.Sequential(*top_layers)

    def forward(self, x):
        # Bottom encoder
        encoded_b = self.encoder_b(x)

        # Medium encoder
        encoded_m = self.encoder_m(encoded_b)

        # Top encoder
        encoded_t = self.encoder_t(encoded_m)

        return encoded_b, encoded_m, encoded_t


class Decoder1D(nn.Module):

    def __init__(self, out_channels, hidden_channels, num_res_blocks, res_channels, embedding_dim, upsample_factor=2, num_upsample_layers_top=2):
        super().__init__()
        ks = upsample_factor * 2  # Kernel size for transpose conv

        # --- Top Decoder ---
        top_layers = []
        top_layers.append(nn.Conv1d(embedding_dim, hidden_channels, kernel_size=3, stride=1, padding=1))
        for _ in range(num_res_blocks):
            top_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels))

        current_channels = hidden_channels
        for i in range(num_upsample_layers_top):
            out_ch = hidden_channels  # Keep hidden_channels for simplicity
            top_layers.append(nn.ConvTranspose1d(current_channels, out_ch, kernel_size=ks, stride=upsample_factor, padding=ks // 2 - upsample_factor // 2))
            if i < num_upsample_layers_top - 1:  # Don't apply final ReLU before concatenation
                top_layers.append(nn.ReLU(inplace=True))
            current_channels = out_ch

        self.decoder_t = nn.Sequential(*top_layers)

        # --- Medium Decoder ---
        medium_layers = []
        medium_layers.append(nn.Conv1d(embedding_dim, hidden_channels, kernel_size=3, stride=1, padding=1))
        for _ in range(num_res_blocks):
            medium_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels))
        self.decoder_m = nn.Sequential(*medium_layers)

        # --- Bottom Decoder ---
        bottom_layers = []
        # Input channels = bottom embedding dim + hidden channels from processed top and medium latents
        bottom_layers.append(nn.Conv1d(embedding_dim + 2 * hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
        for _ in range(num_res_blocks):
            bottom_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels))

        # Upsample back to original resolution
        bottom_layers.append(nn.ConvTranspose1d(hidden_channels, hidden_channels // 2, kernel_size=ks, stride=upsample_factor, padding=ks // 2 - upsample_factor // 2))
        bottom_layers.append(nn.ReLU(inplace=True))
        bottom_layers.append(nn.ConvTranspose1d(hidden_channels // 2, out_channels, kernel_size=ks, stride=upsample_factor, padding=ks // 2 - upsample_factor // 2))
        # Potentially add a final activation like Tanh if data is normalized to [-1, 1]

        self.decoder_b = nn.Sequential(*bottom_layers)

    def forward(self, z_q_b, z_q_m, z_q_t):
        # Decode top latent
        decoded_t = self.decoder_t(z_q_t)

        # Decode medium latent
        decoded_m = self.decoder_m(z_q_m)

        # Ensure sequence lengths match before concatenation
        target_len = z_q_b.shape[2]
        if decoded_t.shape[2] != target_len:
            decoded_t = F.interpolate(decoded_t, size=target_len, mode='nearest')  # Use 'linear' or 'nearest'
        if decoded_m.shape[2] != target_len:
            decoded_m = F.interpolate(decoded_m, size=target_len, mode='nearest')

        # Concatenate bottom latents, upsampled top latents, and medium latents along the channel dimension
        combined = torch.cat([z_q_b, decoded_m, decoded_t], dim=1)  # Shape: (B, EmbDim + 2 * HiddenCh, L_b)

        # Decode combined latents
        reconstructed_x = self.decoder_b(combined)  # Shape: (B, OutCh, L_original)

        return reconstructed_x
class SpectralLoss(nn.Module):
    def __init__(self, n_fft=1024):
        super().__init__()
        self.n_fft = n_fft

    def forward(self, x, x_hat):
        # Ensure input tensors are 2D: [batch_size, sequence_length]
        x = x.squeeze(1)  # Remove the channel dimension if it exists
        x_hat = x_hat.squeeze(1)

        # Compute STFT
        X = torch.stft(x, self.n_fft, return_complex=True).abs()
        X_hat = torch.stft(x_hat, self.n_fft, return_complex=True).abs()

        # Compute L1 loss
        return F.l1_loss(X_hat, X)

class VQVAE2_1D(nn.Module):

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 hidden_channels=128,
                 res_channels=64,
                 num_res_blocks=2,
                 num_embeddings=512,
                 embedding_dim=64,
                 commitment_cost=0.25,
                 downsample_factor=2,
                 num_downsample_layers_top=2,
                 decay=0.99): # Decay is for potential EMA updates
        super().__init__()

        self.encoder = Encoder1D(in_channels, hidden_channels, num_res_blocks, res_channels,
                                 downsample_factor, num_downsample_layers_top)

        # Projection layers to ensure encoder outputs match embedding_dim
        self.pre_vq_conv_b = nn.Conv1d(hidden_channels, embedding_dim, kernel_size=1, stride=1)
        self.pre_vq_conv_m = nn.Conv1d(hidden_channels, embedding_dim, kernel_size=1, stride=1)
        self.pre_vq_conv_t = nn.Conv1d(hidden_channels, embedding_dim, kernel_size=1, stride=1)

        # Vector Quantizer layers (shared class works for 1D)
        self.vq_b = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.vq_m = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.vq_t = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self.decoder = Decoder1D(out_channels, hidden_channels, num_res_blocks, res_channels, embedding_dim,
                                 upsample_factor=downsample_factor, # Match encoder
                                 num_upsample_layers_top=num_downsample_layers_top) # Match encoder
        self.spectral_loss_fn = SpectralLoss(n_fft=1024)  # Initialize spectral loss
    def forward(self, x):
        # Encode
        z_b_pre_vq, z_m_pre_vq, z_t_pre_vq = self.encoder(x)

        z_b_pre_vq = self.pre_vq_conv_b(z_b_pre_vq)
        z_m_pre_vq = self.pre_vq_conv_m(z_m_pre_vq)
        z_t_pre_vq = self.pre_vq_conv_t(z_t_pre_vq)

        # Quantize
        z_q_b, vq_loss_b, _ = self.vq_b(z_b_pre_vq)
        z_q_m, vq_loss_m, _ = self.vq_m(z_m_pre_vq)
        z_q_t, vq_loss_t, _ = self.vq_t(z_t_pre_vq)

        # Decode
        x_recon = self.decoder(z_q_b, z_q_m, z_q_t)

        # Calculate reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        # Combine VQ losses (including commitment loss)
        total_vq_loss = vq_loss_b + vq_loss_t + vq_loss_m

        # Spectral loss
        spectral_loss = self.spectral_loss_fn(x_recon, x)
        spectral_loss= 0.5*spectral_loss  # Scale down the spectral loss
        # Total loss
        total_loss = recon_loss + total_vq_loss + spectral_loss

        # Ensure reconstruction has the same length as input
        if x_recon.shape[2] != x.shape[2]:
            x_recon = F.interpolate(x_recon, size=x.shape[2], mode='linear', align_corners=False)

        return x_recon, total_loss, recon_loss, total_vq_loss, spectral_loss

    def encode(self, x):
        """ Encodes input x to quantized latents and indices. """
        z_b_pre_vq, z_t_pre_vq = self.encoder(x)
        z_b_pre_vq = self.pre_vq_conv_b(z_b_pre_vq)
        z_t_pre_vq = self.pre_vq_conv_t(z_t_pre_vq)

        z_q_b, _, indices_b = self.vq_b(z_b_pre_vq)
        z_q_t, _, indices_t = self.vq_t(z_t_pre_vq)

        return z_q_b, z_q_t, indices_b, indices_t

    def decode(self, z_q_b, z_q_t):
        """ Decodes quantized latents z_q_b and z_q_t into a sequence. """
        x_recon = self.decoder(z_q_b, z_q_t)

        return x_recon


    def decode_from_indices(self, indices_b, indices_t):
        """ Decodes directly from codebook indices. """
        z_q_b = self.vq_b.get_codebook_entry(indices_b)
        z_q_t = self.vq_t.get_codebook_entry(indices_t)
        return self.decode(z_q_b, z_q_t)


def train_vqvae_1d(model, train_loader, optimizer, device, epoch, log_interval=100, gradient_clip_val=None):
    model.train()  
    total_loss_epoch = 0.0
    recon_loss_epoch = 0.0
    vq_loss_epoch = 0.0
    spectral_loss_epoch = 0.0
    num_batches = len(train_loader)
    sample_recon = None  

    if num_batches == 0:
        print("Warning: DataLoader is empty. Skipping training for this epoch.")
        return {'total_loss': 0.0, 'recon_loss': 0.0, 'vq_loss': 0.0}, None

    start_time = time.time()


    with tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch") as progress_bar:
        for batch_idx, data in enumerate(progress_bar):
            data = data.to(device) 


            optimizer.zero_grad()  
            x_recon, total_loss, recon_loss, vq_loss, spectral_loss = model(data)


            total_loss.backward()


            if gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

            optimizer.step()  


            total_loss_epoch += total_loss.item()
            recon_loss_epoch += recon_loss.item()
            vq_loss_epoch += vq_loss.item()
            spectral_loss_epoch += spectral_loss.item()

            if batch_idx == num_batches - 1:
                sample_recon = x_recon[0].detach().cpu().numpy() 


            progress_bar.set_postfix({
                "Total Loss": total_loss.item(),
                "Recon Loss": recon_loss.item(),
                "VQ Loss": vq_loss.item(),
                "Spectral Loss": spectral_loss.item()
            })


    avg_total_loss = total_loss_epoch / num_batches
    avg_recon_loss = recon_loss_epoch / num_batches
    avg_vq_loss = vq_loss_epoch / num_batches
    avg_spectral_loss = spectral_loss_epoch / num_batches

    epoch_time = time.time() - start_time
    print(f'====> Epoch: {epoch} Completed \t'
          f'Average Total Loss: {avg_total_loss:.4f} (Recon: {avg_recon_loss:.4f}, VQ: {avg_vq_loss:.4f}, Sperctral: {avg_spectral_loss:.4f})\t'
          f'Time: {epoch_time:.2f}s')

    return {
        'total_loss': avg_total_loss,
        'recon_loss': avg_recon_loss,
        'vq_loss': avg_vq_loss,
        'spectral_loss': avg_spectral_loss
    }, sample_recon

def reset_unused_codes(vq_layer, threshold=0.1):
    """
    Reset underused codebook entries in the vector quantizer layer.
    """
    with torch.no_grad():
        usage = vq_layer.embedding.weight.norm(dim=1)  # Compute norm as a proxy for usage
        unused = usage < threshold
        if unused.any():
            print(f"Resetting {unused.sum().item()} underused codes.")
            vq_layer.embedding.weight.data[unused] = torch.randn_like(vq_layer.embedding.weight[unused]) * 0.1

if __name__ == '__main__':

    # Set random seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"Random seed set to {SEED}")

    DUMMY_DATA_DIR = r"C:\Users\lukas\Music\youtube_playlist_chopped" 
    if not os.path.exists(DUMMY_DATA_DIR) or not os.listdir(DUMMY_DATA_DIR):
        print("Dummy data directory is missing or empty.")
        print("Please run the DataLoader script first to generate dummy data, or point to your own dataset.")
        exit() # Or call the data generation code here


    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 40 # Set a small number for demonstration
    BATCH_SIZE = 16
    TARGET_SEQ_LENGTH = 64000 # Must match DataLoader and be compatible with model
    LOG_INTERVAL = 10     # Print progress every 10 batches
    SAVE_INTERVAL = 2      # Save model every 2 epochs
    GRADIENT_CLIP = 1.0    
    CHECKPOINT_DIR = r"C:\Users\lukas\Music\VQ_VAE_2_FINAL_1D\spectral"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"Created checkpoint directory: {CHECKPOINT_DIR}")


    model = VQVAE2_1D(
        in_channels=1, out_channels=1, hidden_channels=128, res_channels=64,
        num_res_blocks=2, num_embeddings=512, embedding_dim=64,
        commitment_cost=0.25, downsample_factor=2, num_downsample_layers_top=2
    ).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")


    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    train_loader = create_dataloader(
        data_dir=DUMMY_DATA_DIR,
        sequence_length=TARGET_SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        channels=1,
        file_extension='.wav',
        target_sample_rate=16000, 
        num_workers=2,
        shuffle=True
    )

    print("\nStarting Training...")
    all_train_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_losses, sample_recon = train_vqvae_1d(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=LOG_INTERVAL,
            gradient_clip_val=GRADIENT_CLIP
        )
        all_train_losses.append(epoch_losses)

        # Reset underused codes in vector quantizers
        reset_unused_codes(model.vq_b)
        reset_unused_codes(model.vq_m)
        reset_unused_codes(model.vq_t)

        if epoch % SAVE_INTERVAL == 0 or epoch == NUM_EPOCHS:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'vqvae1d_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_losses,  # Save last epoch's average loss
            }, checkpoint_path)
            print(f"====> Saved checkpoint to {checkpoint_path}")

        if sample_recon is not None:
            sample_recon = sample_recon.squeeze()  
            sample_recon = (sample_recon - sample_recon.min()) / (sample_recon.max() - sample_recon.min())  
            sample_recon = 2 * sample_recon - 1 

            # Save reconstructed audio
            audio_path_recon = os.path.join(CHECKPOINT_DIR, f'reconstruction_epoch_{epoch}.wav')
            write(audio_path_recon, 16000, (sample_recon * 32767).astype('int16'))  
            print(f"====> Saved reconstructed audio to {audio_path_recon}")

            # Save input audio
            sample_input = train_loader.dataset[0].squeeze().numpy()  # Assuming the dataset has a __getitem__ method
            audio_path_input = os.path.join(CHECKPOINT_DIR, f'input_epoch_{epoch}.wav')
            write(audio_path_input, 16000, (sample_input * 32767).astype('int16'))
            print(f"====> Saved input audio to {audio_path_input}")

    print("\nTraining Finished.")


    try:
        import matplotlib.pyplot as plt

        epochs_range = range(1, NUM_EPOCHS + 1)
        total_losses = [l['total_loss'] for l in all_train_losses]
        recon_losses = [l['recon_loss'] for l in all_train_losses]
        spectral_losses = [l['spectral_loss'] for l in all_train_losses]
        vq_losses = [l['vq_loss'] for l in all_train_losses]

        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, total_losses, label='Total Loss')
        plt.plot(epochs_range, recon_losses, label='Reconstruction Loss', linestyle='--')
        plt.plot(epochs_range, vq_losses, label='VQ Loss', linestyle=':')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(CHECKPOINT_DIR, 'training_losses.png'))
        print(f"Saved training loss plot to {os.path.join(CHECKPOINT_DIR, 'training_losses.png')}")


    except ImportError:
        print("\nmatplotlib not found. Skipping loss plot generation.")
        print("Install it with: pip install matplotlib")
