from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import os
from data_load_extended import WavDataset  # Import WavDataset from data_load.py
from data_load_extended import spectrum2wav  # Import spectrum2wav from data_load.py
from tqdm import tqdm  # Import tqdm for the progress bar
from torch.cuda.amp import autocast, GradScaler
N_FFT = 512
N_CHANNELS = round(1 + N_FFT/2)
OUT_CHANNELS = 32
scaler = torch.amp.GradScaler('cuda')


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(x)
        x = self.conv3x3(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        return x + residual
    
class Encoder(nn.Module):
    def __init__(self, input_channels, embedding_dim, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout2d(dropout_rate)
        self.conv1 = nn.Conv2d(input_channels, embedding_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)
        self.resblocks = nn.Sequential(
            ResidualBlock(embedding_dim),
            ResidualBlock(embedding_dim)
        )
        self.pad = nn.ZeroPad2d((0, 0, 0, 1))

    def forward(self, x):
        if x.size(2) % 2 != 0:
            x = self.pad(x)
        x = F.relu(self.dropout(self.conv1(x)))
        x = F.relu(self.dropout(self.conv2(x)))
        return self.resblocks(x)
    
class Decoder(nn.Module):
    def __init__(self, embedding_dim, output_channels, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout2d(dropout_rate)
        self.resblocks = nn.Sequential(
            ResidualBlock(embedding_dim),
            self.dropout,
            ResidualBlock(embedding_dim),
            self.dropout
        )
        self.deconv1 = nn.ConvTranspose2d(embedding_dim, embedding_dim, 
                                        kernel_size=4, stride=2, 
                                        padding=1, output_padding=(0, 1))
        self.deconv2 = nn.ConvTranspose2d(embedding_dim, output_channels, 
                                        kernel_size=4, stride=2, 
                                        padding=1, output_padding=(1, 1))

    def forward(self, x):
        x = self.resblocks(x)
        x = F.relu(self.dropout(self.deconv1(x)))
        return self.deconv2(x)



class TimeExpansionModel(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(TimeExpansionModel, self).__init__()
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # First expansion: 172 -> ~344
        self.expand1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 4),
            stride=(1, 2),
            padding=(0, 1)
        )
        
        # Second expansion: ~344 -> ~646
        self.expand2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 4),
            stride=(1, 2),
            padding=(0, 1)
        )
        
        # Residual connection with dropout
        self.residual = nn.Sequential(
            nn.Upsample(size=(256, 646), mode='bilinear', align_corners=True),
            self.dropout,
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            self.dropout,
            nn.Conv2d(128, 64, kernel_size=3, padding=1)
        )

    def forward(self, x):
        identity = x
        
        x = F.relu(self.dropout(self.expand1(x)))
        x = F.relu(self.dropout(self.expand2(x)))
        
        target_size = (x.size(0), x.size(1), x.size(2), 646)
        x = F.interpolate(x, size=(target_size[2], target_size[3]), 
                         mode='bilinear', align_corners=True)
        
        x = x + self.residual(identity)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e):
        # Flatten z_e to (BHW, D)
        z_e_flattened = z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        codebook = self.embedding.weight  # (K, D)

        # Compute squared norms for codebook vectors and z_e
        codebook_norms = torch.sum(codebook ** 2, dim=1)  # (K,)
        z_norms = torch.sum(z_e_flattened ** 2, dim=1)  # (N,)

        # Chunked computation of dot products
        chunk_size = 4096  # Adjust based on available GPU memory
        num_chunks = (z_e_flattened.size(0) + chunk_size - 1) // chunk_size
        dot_products = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, z_e_flattened.size(0))
            chunk = z_e_flattened[start_idx:end_idx]  # (chunk_size, D)
            dot = torch.matmul(chunk, codebook.t())  # (chunk_size, K)
            dot_products.append(dot)

        dot_products = torch.cat(dot_products, dim=0)  # (N, K)

        # Compute distances using expanded L2 formula
        distances = z_norms.unsqueeze(1) - 2 * dot_products + codebook_norms.unsqueeze(0)

        # Find nearest embeddings
        encoding_indices = torch.argmin(distances, dim=1)
        z_q_flattened = self.embedding(encoding_indices)  # (N, D)

        # Reshape back to original dimensions
        B, D, H, W = z_e.size()
        z_q = z_q_flattened.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # Compute VQ loss
        vq_loss = F.mse_loss(z_q.detach(), z_e) + F.mse_loss(z_q, z_e.detach())

        # Straight-through estimator
        z_q = z_e + (z_q - z_q.detach())

        return z_q, vq_loss
class VQVAE(nn.Module):
    def __init__(self, input_channels, num_embeddings, embedding_dim, output_channels, beta=0.25):
        super().__init__()
        self.encoder = Encoder(input_channels, embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, output_channels)
        self.beta = beta
        self.time_expansion = TimeExpansionModel()
    def forward(self, x):
        # Save original dimensions
        original_h = x.size(2)
        original_w = x.size(3)

        # Encode and decode

        z_e = self.encoder(x)

        z_e = self.time_expansion(z_e)

        z_q, vq_loss = self.vq(z_e)

        x_reconstructed = self.decoder(z_q)

        x_reconstructed = x_reconstructed[:, :, :, :2584]
        return x_reconstructed, vq_loss
    def compute_loss(self, x, y, x_reconstructed, z_e, z_q, vq_loss):

        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(x_reconstructed, y)

        # Commitment loss
        commitment_loss = self.beta * F.mse_loss(z_e, z_q.detach())

        # Total loss
        total_loss = reconstruction_loss + vq_loss + commitment_loss

        return total_loss, reconstruction_loss, vq_loss, commitment_loss
    
def train_vqvae(model, dataloader, num_epochs, device, save_path=r"C:\Users\lukas\Music\VQ_extended_project\reconstructed_audio"):
    """
    Train the VQ-VAE model and save reconstructed audio every 10% of an epoch.

    Args:
        model (nn.Module): VQ-VAE model.
        dataloader (DataLoader): DataLoader for the training data.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to train on (CPU or GPU).
        save_path (str): Directory to save reconstructed audio.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Create directory to save reconstructed audio
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, 
                          desc=f"Epoch {epoch + 1}/{num_epochs}",
                          unit="batch")

        for batch_idx, batch in enumerate(progress_bar):
            x, y, filenames = batch
            x = x.to(device)
            y = y.to(device)

            # Correct autocast implementation
            with torch.amp.autocast('cuda'):
                # Forward pass
                x_reconstructed, vq_loss = model(x)

                # Compute loss
                z_e = model.encoder(x)
                z_e = model.time_expansion(z_e)
                z_q, _ = model.vq(z_e)
                
                loss, reconstruction_loss, vq_loss, commitment_loss = model.compute_loss(
                    x, y, x_reconstructed, z_e, z_q, vq_loss
                )

            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({
                'total_loss': f'{loss.item():.4f}',
                'reconstruction_loss': f'{reconstruction_loss.item():.4f}',
                'vq_loss': f'{vq_loss.item():.4f}',
                'commitment_loss': f'{commitment_loss.item():.4f}'
            })
            # Save reconstructed audio every 10% of an epoch
            if batch_idx % (len(dataloader) // 2) == 0:
                print(f"Saving audio at epoch {epoch + 1}, batch {batch_idx}")
                model.eval()
                with torch.no_grad():
                    # Save reconstructed audio
                    for i in range(min(2, x.size(0))):
                        spectrogram = x_reconstructed[i].cpu().squeeze(0).numpy()
                        spectrogram = spectrogram.astype(np.float32)
                        
                        save_file = os.path.join(
                            save_path, 
                            f'epoch_{epoch+1}_batch_{batch_idx}_sample_{i}.wav'
                        )
                        
                        try:
                            spectrum2wav(spectrogram, 44100, save_file)
                            print(f"Saved reconstructed audio sample {i}")
                        except Exception as e:
                            print(f"Error saving reconstructed audio: {str(e)}")
                
                model.train()

            # Update the progress bar with the current loss
            progress_bar.set_postfix(loss=loss.item())
    model_save_path = os.path.join(os.path.dirname(save_path), 'model')
    os.makedirs(model_save_path, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'final_loss': loss.item(),
    }, os.path.join(model_save_path, 'vqvae_final.pth'))
    print(f"Model saved to {os.path.join(model_save_path, 'vqvae_final.pth')}")
    return loss, reconstruction_loss, vq_loss, commitment_loss
        # Print epoch loss
def process_audio(train_mode=0, audio_path=None, model_path=None, device=None):
    """
    Process audio files through the VQ-VAE model. If train_mode is 1, train a new model.
    If train_mode is 0, load an existing model and process provided audio files.
    
    Args:
        train_mode (int): 1 for training, 0 for inference
        audio_path (str): Path to audio file or directory
        model_path (str): Path to saved model checkpoint
        device (torch.device): Device to run on (if None, will auto-select)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = VQVAE(input_channels=1,
                  num_embeddings=128,
                  embedding_dim=64,
                  output_channels=1)
    
    if train_mode == 1:
        # Training mode
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        dataset = WavDataset(audio_path, target_length=2584)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        loss, reconstruction_loss, vq_loss, commitment_loss = train_vqvae(
            model, dataloader, num_epochs=10, device=device
        )
        
        # Create losses directory if it doesn't exist
        losses_dir = os.path.join(r"C:\Users\lukas\Music\VQ_extended_project\losses")
        os.makedirs(losses_dir, exist_ok=True)

        # Move tensors to CPU and detach before converting to numpy
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().numpy()
        if isinstance(reconstruction_loss, torch.Tensor):
            reconstruction_loss = reconstruction_loss.detach().cpu().numpy()
        if isinstance(vq_loss, torch.Tensor):
            vq_loss = vq_loss.detach().cpu().numpy()
        if isinstance(commitment_loss, torch.Tensor):
            commitment_loss = commitment_loss.detach().cpu().numpy()

        # Plot and save loss curves
        plt.figure()
        plt.plot(loss)
        plt.title('Total Loss')
        plt.savefig(os.path.join(losses_dir, "loss_curve.png"))
        plt.close()
        
        plt.figure()
        plt.plot(reconstruction_loss)
        plt.title('Reconstruction Loss')
        plt.savefig(os.path.join(losses_dir, "reconstruction_loss_curve.png"))
        plt.close()

        plt.figure()
        plt.plot(vq_loss)
        plt.title('VQ Loss')
        plt.savefig(os.path.join(losses_dir, "vq_loss_curve.png"))
        plt.close()

        plt.figure()
        plt.plot(commitment_loss)
        plt.title('Commitment Loss')
        plt.savefig(os.path.join(losses_dir, "commitment_loss_curve.png"))
        plt.close()
        return loss, reconstruction_loss, vq_loss, commitment_loss
    
    else:
        # Inference mode
        if model_path is None or not os.path.exists(model_path):
            raise ValueError("Model path must be provided for inference mode")
        
        # Load the model
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Process audio files
        output_dir = os.path.join(os.path.dirname(audio_path), 'processed_audio')
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle single file or directory
        if os.path.isfile(audio_path):
            files = [audio_path]
        else:
            files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) 
                    if f.endswith('.wav')]
        
        # Process each file
        with torch.no_grad():
            for file in tqdm(files, desc="Processing audio files"):
                try:
                    # Create dataset for single file
                    single_dataset = WavDataset(os.path.dirname(file))
                    single_loader = DataLoader(single_dataset, batch_size=1, shuffle=False)
                    
                    # Get the spectrogram
                    for batch in single_loader:
                        if os.path.basename(file) == batch[1][0]:  # Match filename
                            spectrogram, _ = batch
                            spectrogram = spectrogram.to(device)
                            
                            # Process through model
                            reconstructed, _ = model(spectrogram)
                            
                            # Save reconstructed audio
                            output_path = os.path.join(
                                output_dir, 
                                f'processed_{os.path.basename(file)}'
                            )
                            
                            reconstructed_spec = reconstructed[0].cpu().squeeze(0).numpy()
                            spectrum2wav(reconstructed_spec, 44100, output_path)
                            print(f"Processed {file} -> {output_path}")
                            break
                            
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

# Update the main block to use the new function
if __name__ == "__main__":
    # Example usage
    TRAIN_MODE = 1 # Set to 1 for training, 0 for inference
    AUDIO_PATH = r"C:\Users\lukas\Music\chopped_segments"  # Your audio path
    MODEL_PATH = r"C:\Users\lukas\Music\VQ_project\model\vqvae_final.pth"
    
    process_audio(
        train_mode=TRAIN_MODE,
        audio_path=AUDIO_PATH,
        model_path=MODEL_PATH
    )

