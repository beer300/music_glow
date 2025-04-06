from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import os
from data_load import WavDataset  # Import WavDataset from data_load.py
from data_load import spectrum2wav  # Import spectrum2wav from data_load.py
from tqdm import tqdm  # Import tqdm for the progress bar

N_FFT = 512
N_CHANNELS = round(1 + N_FFT/2)
OUT_CHANNELS = 32



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
    def __init__(self, input_channels, embedding_dim):
        super(Encoder, self).__init__()
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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.resblocks(x)
class Decoder(nn.Module):
    def __init__(self, embedding_dim, output_channels):
        super(Decoder, self).__init__()
        self.resblocks = nn.Sequential(
            ResidualBlock(embedding_dim),
            ResidualBlock(embedding_dim))
        self.deconv1 = nn.ConvTranspose2d(embedding_dim, embedding_dim, 
                                        kernel_size=4, stride=2, 
                                        padding=1, output_padding=(0, 1))
        self.deconv2 = nn.ConvTranspose2d(embedding_dim, output_channels, 
                                        kernel_size=4, stride=2, 
                                        padding=1, output_padding=(1, 1) )

    def forward(self, x):
        x = self.resblocks(x)
        x = self.deconv1(x)
        return self.deconv2(x)

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

    def forward(self, x):
        # Save original dimensions
        original_h = x.size(2)
        original_w = x.size(3)

        # Encode and decode
        z_e = self.encoder(x)

        z_q, vq_loss = self.vq(z_e)

        x_reconstructed = self.decoder(z_q)

        # Crop to original input size
        x_reconstructed = x_reconstructed[:, :, :original_h, :original_w]

        return x_reconstructed, vq_loss
    def compute_loss(self, x, x_reconstructed, z_e, z_q, vq_loss):

        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(x_reconstructed, x)

        # Commitment loss
        commitment_loss = self.beta * F.mse_loss(z_e, z_q.detach())

        # Total loss
        total_loss = reconstruction_loss + vq_loss + commitment_loss

        return total_loss, reconstruction_loss, vq_loss, commitment_loss
    
def train_vqvae(model, dataloader, num_epochs, device, save_path=r"C:\Users\lukas\Music\VQ_project\reconstructed_audio"):
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

        # Initialize the progress bar
        progress_bar = tqdm(dataloader, 
                          desc=f"Epoch {epoch + 1}/{num_epochs}",
                          unit="batch")

        for batch_idx, batch in enumerate(progress_bar):
            # Extract spectrograms and move them to the device
            x, filenames = batch  # x is the spectrogram, filenames are the file names
            x = x.to(device)

            # Forward pass
            x_reconstructed, vq_loss = model(x)

            # Compute loss
            z_e = model.encoder(x)
            z_q, _ = model.vq(z_e)
            loss, reconstruction_loss, vq_loss, commitment_loss = model.compute_loss(x, x_reconstructed, z_e, z_q, vq_loss)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({
                'total_loss': f'{loss.item():.4f}',
                'reconstruction_loss': f'{reconstruction_loss.item():.4f}',
                'vq_loss': f'{vq_loss.item():.4f}',
                'commitment_loss': f'{commitment_loss.item():.4f}'
            })
            # Save reconstructed audio every 10% of an epoch
            if batch_idx % (len(dataloader) // 4) == 0:
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
                    
                    # Generate and save audio from random vectors
                    random_z = torch.randn_like(z_q).to(device)
                    random_output = model.decoder(random_z)
                    
                    for i in range(2):  # Save 2 random samples
                        random_spectrogram = random_output[i].cpu().squeeze(0).numpy()
                        random_spectrogram = random_spectrogram.astype(np.float32)
                        
                        random_save_file = os.path.join(
                            save_path, 
                            f'epoch_{epoch+1}_random_output_{i}.wav'
                        )
                        
                        try:
                            spectrum2wav(random_spectrogram, 44100, random_save_file)
                            print(f"Saved random audio sample {i}")
                        except Exception as e:
                            print(f"Error saving random audio: {str(e)}")
                
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

if __name__ == "__main__":
    data_path = r"C:\Users\lukas\Music\youtube_playlist_chopped"
    dataset = WavDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Reduced embedding dimensions
    model = VQVAE(input_channels=1,
                num_embeddings=128,   # Reduced from 512
                embedding_dim=64,     # Reduced from 256
                output_channels=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Optional: Load existing model
    model_path = r"C:\Users\lukas\Music\VQ_project\reconstructed_audio\model\vqvae_final.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")

    loss, reconstruction_loss, vq_loss, commitment_loss = train_vqvae(model, dataloader, num_epochs=10, device=device)

    plt.figure()
    plt.plot(loss)
    plt.savefig(os.path.join(r"C:\Users\lukas\Music\VQ_project\losses", "loss_curve.png"))