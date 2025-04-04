import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(x)
        x = self.conv3x3(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        x += residual
        return x
    
class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        
        # Define residual blocks
        self.residual1 = ResidualBlock(256, 256)
        self.residual2 = ResidualBlock(256, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.residual1(x)  # Use the defined residual block
        x = self.residual2(x)  # Use the defined residual block
        return x
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels):
        super(Decoder, self).__init__()
        
        # Define residual blocks
        self.residual1 = ResidualBlock(256, 256)
        self.residual2 = ResidualBlock(256, 256)
        
        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.residual1(x)  # Use the defined residual block
        x = self.residual2(x)  # Use the defined residual block
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x
    
class VQVAE(nn.Module):
    def __init__(self, input_channels, latent_dim, output_channels, num_embeddings, embedding_dim, beta=0.25):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(input_channels, latent_dim)
        self.embd = nn.Embedding(10 ,64)
        self.decoder = Decoder(latent_dim, output_channels)
        self.beta = beta

    def forward(self, z_e):
        """
        Forward pass for vector quantization.
        
        Args:
            z_e (torch.Tensor): Encoder output of shape (B, D, H, W).
        
        Returns:
            z_q (torch.Tensor): Quantized latent representation of shape (B, D, H, W).
            vq_loss (torch.Tensor): Vector quantization loss.
        """
        # Flatten z_e to (BHW, D)
        z_e_flattened = z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # Compute distances between z_e and embedding vectors
        distances = torch.cdist(z_e_flattened, self.embedding.weight)

        # Find the nearest embedding index for each z_e
        encoding_indices = torch.argmin(distances, dim=1)

        # Quantize z_e by replacing with nearest embedding vector
        z_q_flattened = self.embedding(encoding_indices)

        # Reshape z_q back to (B, D, H, W)
        z_q = z_q_flattened.view(z_e.size(0), z_e.size(2), z_e.size(3), self.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # Compute VQ loss (commitment loss)
        vq_loss = F.mse_loss(z_q.detach(), z_e) + F.mse_loss(z_q, z_e.detach())

        # Straight-through estimator: pass gradients to encoder
        z_q = z_e + (z_q - z_q.detach())
        x_reconstructed = self.decoder(z_q)


        return x_reconstructed, z_e, z_q, vq_loss
    
    def compute_loss(self, x, x_reconstructed, z_e, z_q, vq_loss):
        """
        Compute the total loss for VQ-VAE.

        Args:
            x (torch.Tensor): Original input.
            x_reconstructed (torch.Tensor): Reconstructed input.
            z_e (torch.Tensor): Encoder output.
            z_q (torch.Tensor): Quantized latent representation.
            vq_loss (torch.Tensor): Vector quantization loss.

        Returns:
            torch.Tensor: Total loss.
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(x_reconstructed, x)

        # Commitment loss
        commitment_loss = self.beta * F.mse_loss(z_e, z_q.detach())

        # Total loss
        total_loss = reconstruction_loss + vq_loss + commitment_loss

        return total_loss
    
def train_vqvae(model, dataloader, num_epochs, device):
    """
    Train the VQ-VAE model.

    Args:
        model (nn.Module): VQ-VAE model.
        dataloader (DataLoader): DataLoader for the training data.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to train on (CPU or GPU).
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            # Move data to device
            x = batch.to(device)

            # Forward pass
            x_reconstructed, z_e, z_q, vq_loss = model(x)

            # Compute loss
            loss = model.compute_loss(x, x_reconstructed, z_e, z_q, vq_loss)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")