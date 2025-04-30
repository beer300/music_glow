import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim  # Import optimizer
import random  # Import random for seed setting
from torch.amp import GradScaler, autocast
import time
from tqdm import tqdm 
import VQ_VAE_2_1D_data 
from VQ_VAE_2_1D_data import create_dataloader
from scipy.io.wavfile import write  # For saving audio
from torch.optim.lr_scheduler import LambdaLR 
import matplotlib.pyplot as plt

class ResidualBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_prob=0.2):  # Add dropout_prob
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)  # 1x1 equivalent
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout layer

        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(x)
        out = F.relu(self.conv1(out))
        out = self.dropout(out)  # Apply dropout
        out = self.conv2(out)
        out = out + residual
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
        input_dims = len(inputs.shape)
        if input_dims == 3:  # 1D case (B, C, L)
            inputs_permuted = inputs.permute(0, 2, 1).contiguous()
        elif input_dims == 4:  # 2D case (B, C, H, W)
            inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()
        else:
            raise ValueError(f"Input tensor dimensions {input_dims} not supported (expected 3 or 4).")

        input_shape = inputs_permuted.shape
        flat_input = inputs_permuted.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1)

        # Jukebox's codebook restart: Replace dead codes with encoder outputs
        if self.training:
            # Identify codes not used in the current batch
            unique_indices = torch.unique(encoding_indices)
            used_mask = torch.zeros(self.num_embeddings, device=encoding_indices.device, dtype=torch.bool)
            used_mask[unique_indices] = True
            dead_codes = ~used_mask

            num_dead = dead_codes.sum().item()
            if num_dead > 0:
                rand_indices = torch.randint(0, flat_input.size(0), (num_dead,), device=flat_input.device)
                replacement = flat_input[rand_indices].detach()
                
                # Convert replacement to FP32 to match embedding dtype
                with torch.no_grad():
                    self.embedding.weight.data[dead_codes] = replacement.float()  # <-- FIX HERE

        quantized = self.embedding(encoding_indices)
        quantized = quantized.view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach().view(flat_input.shape), flat_input)
        q_latent_loss = F.mse_loss(quantized.view(flat_input.shape), flat_input.detach())
        vq_loss = e_latent_loss + self.commitment_cost * q_latent_loss

        quantized_st = inputs_permuted + (quantized - inputs_permuted).detach()

        if input_dims == 3:
            quantized_st = quantized_st.permute(0, 2, 1).contiguous()
            indices_reshaped = encoding_indices.view(input_shape[:-1])
        else:
            quantized_st = quantized_st.permute(0, 3, 1, 2).contiguous()
            indices_reshaped = encoding_indices.view(input_shape[:-1])

        return quantized_st, vq_loss, indices_reshaped

    def get_codebook_entry(self, indices):
        # indices shape: (B, L) or (B, H, W) or (N,)
        #print(f"Indices shape: {indices.shape}")
        batch_size = indices.shape[0]
        #print(f"Batch size: {batch_size}")
        indices_flatten = indices.reshape(-1) # (B*L,) or (B*H*W,) or (N,)
        #print(f"Flattened indices shape: {indices_flatten.shape}")
        quantized = self.embedding(indices_flatten) # (B*L, C) or (B*H*W, C) or (N, C)
        #print(f"Quantized shape: {quantized.shape}")
        # Reshape if needed
        if len(indices.shape) >= 2: # Multi-dimensional indices
            C = quantized.shape[-1]
            spatial_shape = indices.shape[1:] # (L,) or (H, W)
            quantized = quantized.view(batch_size, *spatial_shape, C)
            if len(indices.shape) == 2: # 1D case (B, L) -> (B, L, C) -> (B, C, L)
                quantized = quantized.permute(0, 2, 1).contiguous()
            elif len(indices.shape) == 3: # 2D case (B, H, W) -> (B, H, W, C) -> (B, C, H, W)
                 quantized = quantized.permute(0, 3, 1, 2).contiguous()
        #print(f"Final quantized shape: {quantized.shape}")
        return quantized



class Encoder1D(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_res_blocks, res_channels, downsample_factor=2, num_downsample_layers_top=2, dropout_prob=0.2):  # Add dropout_prob
        super().__init__()
        ks = downsample_factor * 2  # Kernel size often related to stride

        # --- Bottom Level Encoder ---
        bottom_layers = []
        bottom_layers.append(nn.Conv1d(in_channels, hidden_channels // 2, kernel_size=ks, stride=downsample_factor, padding=ks // 2 - downsample_factor // 2))  # Halve length
        bottom_layers.append(nn.ReLU(inplace=True))
        bottom_layers.append(nn.Dropout(dropout_prob))  # Add dropout
        bottom_layers.append(nn.Conv1d(hidden_channels // 2, hidden_channels, kernel_size=ks, stride=downsample_factor, padding=ks // 2 - downsample_factor // 2))  # Halve length again
        bottom_layers.append(nn.ReLU(inplace=True))
        bottom_layers.append(nn.Dropout(dropout_prob))  # Add dropout
        bottom_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))  # Adjust channels

        for _ in range(num_res_blocks):
            bottom_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels, dropout_prob))  # Pass dropout_prob
        bottom_layers.append(nn.ReLU(inplace=True))  # Activation before VQ projection
        self.encoder_b = nn.Sequential(*bottom_layers)

        # --- Medium Level Encoder ---
        medium_layers = []
        medium_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))  # Adjust channels
        for _ in range(num_res_blocks):
            medium_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels, dropout_prob))  # Pass dropout_prob
        medium_layers.append(nn.ReLU(inplace=True))  # Activation before passing to top encoder
        self.encoder_m = nn.Sequential(*medium_layers)

        # --- Top Level Encoder ---
        top_layers = []
        current_channels = hidden_channels
        for i in range(num_downsample_layers_top):
            out_ch = hidden_channels  # Keep hidden_channels for simplicity
            top_layers.append(nn.Conv1d(current_channels, out_ch, kernel_size=ks, stride=downsample_factor, padding=ks // 2 - downsample_factor // 2))
            top_layers.append(nn.ReLU(inplace=True))
            top_layers.append(nn.Dropout(dropout_prob))  # Add dropout
            current_channels = out_ch

        top_layers.append(nn.Conv1d(current_channels, hidden_channels, kernel_size=3, stride=1, padding=1))  # Adjust channels before ResBlocks
        for _ in range(num_res_blocks):
            top_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels, dropout_prob))  # Pass dropout_prob
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

    def __init__(self, out_channels, hidden_channels, num_res_blocks, res_channels, embedding_dim, upsample_factor=2, num_upsample_layers_top=2, dropout_prob=0.2):  # Add dropout_prob
        super().__init__()
        ks = upsample_factor * 2  # Kernel size for transpose conv

        # --- Top Decoder ---
        top_layers = []
        top_layers.append(nn.Conv1d(embedding_dim, hidden_channels, kernel_size=3, stride=1, padding=1))
        for _ in range(num_res_blocks):
            top_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels, dropout_prob))  # Pass dropout_prob

        current_channels = hidden_channels
        for i in range(num_upsample_layers_top):
            out_ch = hidden_channels  # Keep hidden_channels for simplicity
            top_layers.append(nn.ConvTranspose1d(current_channels, out_ch, kernel_size=ks, stride=upsample_factor, padding=ks // 2 - upsample_factor // 2))
            if i < num_upsample_layers_top - 1:  # Don't apply final ReLU before concatenation
                top_layers.append(nn.ReLU(inplace=True))
                top_layers.append(nn.Dropout(dropout_prob))  # Add dropout
            current_channels = out_ch

        self.decoder_t = nn.Sequential(*top_layers)

        # --- Medium Decoder ---
        medium_layers = []
        medium_layers.append(nn.Conv1d(embedding_dim, hidden_channels, kernel_size=3, stride=1, padding=1))
        for _ in range(num_res_blocks):
            medium_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels, dropout_prob))  # Pass dropout_prob
        self.decoder_m = nn.Sequential(*medium_layers)

        # --- Bottom Decoder ---
        bottom_layers = []
        # Input channels = bottom embedding dim + hidden channels from processed top and medium latents
        bottom_layers.append(nn.Conv1d(embedding_dim + 2 * hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
        for _ in range(num_res_blocks):
            bottom_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels, dropout_prob))  # Pass dropout_prob

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
    def __init__(self, fft_sizes=[256, 512, 1024, 2048, 4096],  # More scales
                 mag_weight=1.0, logmag_weight=0.5):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.mag_weight = mag_weight
        self.logmag_weight = logmag_weight

    def forward(self, x, x_hat):
        # Ensure input tensors are 2D: [batch_size, sequence_length]
        x = x.squeeze(1)  # Remove the channel dimension if it exists
        x_hat = x_hat.squeeze(1)

        mag_loss = 0
        logmag_loss = 0

        for n_fft in self.fft_sizes:
            # Compute STFT
            X = torch.stft(x, n_fft, return_complex=True).abs()
            X_hat = torch.stft(x_hat, n_fft, return_complex=True).abs()

            # Compute magnitude loss
            mag_loss += F.l1_loss(X_hat, X)

            # Compute log-magnitude loss
            logmag_loss += F.l1_loss(torch.log(X_hat + 1e-7), torch.log(X + 1e-7))

        # Average the losses across all FFT sizes
        mag_loss /= len(self.fft_sizes)
        logmag_loss /= len(self.fft_sizes)

        # Combine magnitude and log-magnitude losses with weights
        return mag_loss * self.mag_weight + logmag_loss * self.logmag_weight
class VQVAE2_1D(nn.Module):

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 hidden_channels=128,
                 res_channels=64,
                 num_res_blocks=2,
                 num_embeddings=512,
                 embedding_dim=256,
                 commitment_cost=0.25,
                 downsample_factor=2,
                 num_downsample_layers_top=2,
                 decay=0.99,
                 dropout_prob=0.2):  # Add dropout_prob
        super().__init__()
        self.num_embeddings = num_embeddings
        self.encoder = Encoder1D(in_channels, hidden_channels, num_res_blocks, res_channels,
                                 downsample_factor, num_downsample_layers_top, dropout_prob)  # Pass dropout_prob

        # Projection layers to ensure encoder outputs match embedding_dim
        self.pre_vq_conv_b = nn.Conv1d(hidden_channels, embedding_dim, kernel_size=1, stride=1)
        self.pre_vq_conv_m = nn.Conv1d(hidden_channels, embedding_dim, kernel_size=1, stride=1)
        self.pre_vq_conv_t = nn.Conv1d(hidden_channels, embedding_dim, kernel_size=1, stride=1)

        # Vector Quantizer layers (shared class works for 1D)
        self.vq_b = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.vq_m = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.vq_t = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self.decoder = Decoder1D(out_channels, hidden_channels, num_res_blocks, res_channels, embedding_dim,
                                 upsample_factor=downsample_factor,  # Match encoder
                                 num_upsample_layers_top=num_downsample_layers_top, dropout_prob=dropout_prob)  # Pass dropout_prob
        self.spectral_loss_fn = SpectralLoss(fft_sizes=[512, 1024, 2048])

    def forward(self, x):
        # Encode
        z_b_pre_vq, z_m_pre_vq, z_t_pre_vq = self.encoder(x)

        z_b_pre_vq = self.pre_vq_conv_b(z_b_pre_vq)
        z_m_pre_vq = self.pre_vq_conv_m(z_m_pre_vq)
        z_t_pre_vq = self.pre_vq_conv_t(z_t_pre_vq)

        # Quantize
        z_q_b, vq_loss_b, indices_b = self.vq_b(z_b_pre_vq)
        z_q_m, vq_loss_m, indices_m = self.vq_m(z_m_pre_vq)
        z_q_t, vq_loss_t, indices_t = self.vq_t(z_t_pre_vq)

        # Decode
        x_recon = self.decoder(z_q_b, z_q_m, z_q_t)

        # Reconstruct audio from indices
        x_recon_from_indices = self.reconstruct_from_indices(indices_b, indices_m, indices_t)
        if x_recon.shape[2] != x.shape[2]:
            x_recon = F.pad(x_recon, (0, x.shape[2] - x_recon.shape[2]))


        # Calculate reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        # Combine VQ losses (including commitment loss)
        total_vq_loss = vq_loss_b + vq_loss_t + vq_loss_m

        # Spectral loss
        spectral_loss = self.spectral_loss_fn(x_recon, x)
        spectral_loss = 0.2 * spectral_loss  # Scale down the spectral loss

        # Total loss
        total_loss = recon_loss + total_vq_loss + spectral_loss

        # Ensure reconstruction has the same length as input
        if x_recon.shape[2] != x.shape[2]:
            x_recon = F.interpolate(x_recon, size=x.shape[2], mode='linear', align_corners=False)

        return x_recon, x_recon_from_indices, total_loss, recon_loss, total_vq_loss, spectral_loss
    def reconstruct_from_indices(self, indices_b, indices_m, indices_t):
        """
        Reconstruct audio from quantized indices.

        Args:
            indices_b (torch.Tensor): Indices for the bottom quantizer.
            indices_m (torch.Tensor): Indices for the medium quantizer.
            indices_t (torch.Tensor): Indices for the top quantizer.

        Returns:
            torch.Tensor: Reconstructed audio.
        """
        # Get quantized embeddings from indices
        #print(f"Indices shapes: {indices_b.shape}, {indices_m.shape}, {indices_t.shape}")
        z_q_b = self.vq_b.get_codebook_entry(indices_b)
        z_q_m = self.vq_m.get_codebook_entry(indices_m)
        z_q_t = self.vq_t.get_codebook_entry(indices_t)
        #print(f"z_q_b shape: {z_q_b.shape}, z_q_m shape: {z_q_m.shape}, z_q_t shape: {z_q_t.shape}")
        # Decode the quantized embeddings
        reconstructed_audio = self.decoder(z_q_b, z_q_m, z_q_t)

        return reconstructed_audio
    def encode(self, x):
        """ Encodes input x to quantized latents and indices. """
        z_b_pre_vq, z_m_pre_vq, z_t_pre_vq = self.encoder(x)
        z_b_pre_vq = self.pre_vq_conv_b(z_b_pre_vq)
        z_m_pre_vq = self.pre_vq_conv_m(z_m_pre_vq)
        z_t_pre_vq = self.pre_vq_conv_t(z_t_pre_vq)

        z_q_b, _, indices_b = self.vq_b(z_b_pre_vq)
        z_q_m, _, indices_m = self.vq_m(z_m_pre_vq)
        z_q_t, _, indices_t = self.vq_t(z_t_pre_vq)

        return z_q_b, z_q_m, z_q_t, indices_b, indices_m, indices_t

    def decode(self, z_q_b, z_q_t):
        """ Decodes quantized latents z_q_b and z_q_t into a sequence. """
        x_recon = self.decoder(z_q_b, z_q_t)

        return x_recon




def train_vqvae_1d(model, train_loader, optimizer, device, epoch, log_interval=100, gradient_clip_val=None):
    model.train()  
    total_loss_epoch = 0.0
    recon_loss_epoch = 0.0
    vq_loss_epoch = 0.0
    spectral_loss_epoch = 0.0
    num_batches = len(train_loader)
    sample_recon = None  
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    if num_batches == 0:
        print("Warning: DataLoader is empty. Skipping training for this epoch.")
        return {'total_loss': 0.0, 'recon_loss': 0.0, 'vq_loss': 0.0}, None

    start_time = time.time()

    scaler = GradScaler("cuda")
    with tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch") as progress_bar:
        for batch_idx, data in enumerate(progress_bar):
            data, _ = data
            data = data.to(device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)


            optimizer.zero_grad()  
            with autocast("cuda"):
                xrecon_normal, x_recon, total_loss, recon_loss, vq_loss, spectral_loss = model(data)

            
            total_loss.backward()


            if gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

            optimizer.step()  


            total_loss_epoch += total_loss.item()
            recon_loss_epoch += recon_loss.item()
            vq_loss_epoch += vq_loss.item()
            spectral_loss_epoch += spectral_loss.item()

            if batch_idx == num_batches - 1:
                sample_input = data[0].detach().cpu().numpy()  # Capture input from last batch
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
    }, sample_recon,sample_input

def reset_unused_codes(vq_layer, threshold=0.1):
    """
    Reset underused codebook entries in the vector quantizer layer.
    """
    with torch.no_grad():
        usage = vq_layer.embedding.weight.norm(dim=1)  # Compute norm as a proxy for usage
        unused = usage < threshold
        print(f"Unused codes: {unused.sum().item()} out of {vq_layer.num_embeddings}")
        if unused.any():
            print(f"Resetting {unused.sum().item()} underused codes.")
            vq_layer.embedding.weight.data[unused] = torch.randn_like(vq_layer.embedding.weight[unused]) * 0.1
def warmup_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs  # Linear warm-up
        return 1.0  # Keep learning rate constant after warm-up
    return LambdaLR(optimizer, lr_lambda)

if __name__ == '__main__':


    # Set random seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"Random seed set to {SEED}")

    DATA_DIR_1 = r"C:\Users\lukas\Documents\datasets\wavenet_dataset\input_audio" 
    DATA_DIR_2 = r"C:\Users\lukas\Documents\datasets\wavenet_dataset\target_audio" 
    
    if not os.path.exists(DATA_DIR_1) or not os.listdir(DATA_DIR_1):
        print(" data directory is missing or empty.")
        print("Please run the DataLoader script first to generate dummy data, or point to your own dataset.")
        exit() # Or call the data generation code here


    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 60 # Set a small number for demonstration
    BATCH_SIZE = 4
    TARGET_SEQ_LENGTH = 16000*9 # Must match DataLoader and be compatible with model
    LOG_INTERVAL = 10     # Print progress every 10 batches
    SAVE_INTERVAL = 2      # Save model every 2 epochs
    GRADIENT_CLIP = 1.0    
    CHECKPOINT_DIR = r"C:\Users\lukas\Music\VQ_VAE_2_FINAL_1D\indicies_33"
    WEIGHT_DECAY = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"Created checkpoint directory: {CHECKPOINT_DIR}")


    model = VQVAE2_1D(
        in_channels=1, out_channels=1, hidden_channels=128, res_channels=256,
        num_res_blocks=2, num_embeddings=512, embedding_dim=256,
        commitment_cost=1, downsample_factor=2, num_downsample_layers_top=3,
        decay=0.95  # Zmniejszono decay
    ).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.")


    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sample_rate = 16000  # Set the sample rate for your audio data

    train_loader = create_dataloader(
        data_dir=DATA_DIR_1,
        data_dir_2=DATA_DIR_2,
        sequence_length=TARGET_SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        channels=1,
        file_extension='.wav',
        target_sample_rate=sample_rate, 
        num_workers=2,
        shuffle=True
    )

    print("\nStarting Training...")
    all_train_losses = []
    WARMUP_EPOCHS = 5  # Number of warm-up epochs
    scheduler = warmup_lr_scheduler(optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=NUM_EPOCHS)
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_losses, sample_recon,sample_input = train_vqvae_1d(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=LOG_INTERVAL,
            gradient_clip_val=GRADIENT_CLIP
        )
        all_train_losses.append(epoch_losses)
        scheduler.step()  # Step the learning rate scheduler
        # Reset underused codes in vector quantizers
        #reset_unused_codes(model.vq_b)
        #reset_unused_codes(model.vq_m)
        #reset_unused_codes(model.vq_t)

        if epoch % SAVE_INTERVAL == 0 or epoch == NUM_EPOCHS:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'vqvae1d_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_losses,  # Save last epoch's average loss
            }, checkpoint_path)
            print(f"====> Saved checkpoint to {checkpoint_path}")
            try:
                # Adjust epochs_range to match the length of all_train_losses
                epochs_range = range(1, len(all_train_losses) + 1)
                total_losses = [l['total_loss'] for l in all_train_losses]
                recon_losses = [l['recon_loss'] for l in all_train_losses]
                spectral_losses = [l['spectral_loss'] for l in all_train_losses]
                vq_losses = [l['vq_loss'] for l in all_train_losses]

                plt.figure(figsize=(10, 5))
                plt.plot(epochs_range, total_losses, label='Total Loss')
                plt.plot(epochs_range, recon_losses, label='Reconstruction Loss', linestyle='--')
                plt.plot(epochs_range, spectral_losses, label='Spectral Loss', linestyle='-.')
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

        if sample_recon is not None and sample_input is not None:
            # Save input audio
            sample_input = sample_input.squeeze()  # Remove channel dimension
            print(f"Sample input shape: {sample_input.shape}")
            sample_input = np.clip(sample_input, -1.0, 1.0)  # Ensure within valid range
            audio_path_input = os.path.join(CHECKPOINT_DIR, f'input_epoch_{epoch}.wav')
            write(audio_path_input, sample_rate, (sample_input * 32767).astype('int16'))
            print(f"====> Saved input audio to {audio_path_input}")

            # Save reconstructed audio
            sample_recon = sample_recon.squeeze()  
            sample_recon = np.clip(sample_recon, -1.0, 1.0)  # Clip to valid range
            audio_path_recon = os.path.join(CHECKPOINT_DIR, f'reconstruction_epoch_{epoch}.wav')
            write(audio_path_recon, sample_rate, (sample_recon * 32767).astype('int16'))  
            print(f"====> Saved reconstructed audio to {audio_path_recon}")

    print("\nTraining Finished.")


