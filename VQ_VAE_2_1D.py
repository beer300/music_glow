import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim  # Import optimizer

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

        input_dims = len(inputs.shape)
        if input_dims == 3: 

            inputs_permuted = inputs.permute(0, 2, 1).contiguous()
        elif input_dims == 4:

             inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()
        else:
            raise ValueError(f"Input tensor dimensions {input_dims} not supported (expected 3 or 4).")

        input_shape = inputs_permuted.shape

        flat_input = inputs_permuted.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1)

        quantized = self.embedding(encoding_indices) 

        quantized = quantized.view(input_shape) 

        e_latent_loss = F.mse_loss(quantized.detach().view(flat_input.shape), flat_input)
        q_latent_loss = F.mse_loss(quantized.view(flat_input.shape), flat_input.detach())
        vq_loss = e_latent_loss + self.commitment_cost * q_latent_loss

        vq_loss = e_latent_loss + self.commitment_cost * q_latent_loss
 

        quantized_st = inputs_permuted + (quantized - inputs_permuted).detach()


        if input_dims == 3:
             quantized_st = quantized_st.permute(0, 2, 1).contiguous()

             indices_reshaped = encoding_indices.view(input_shape[:-1])
        else: 
             quantized_st = quantized_st.permute(0, 3, 1, 2).contiguous()

             indices_reshaped = encoding_indices.view(input_shape[:-1])

        #print(f"VectorQuantizer quantized shape: {quantized_st.shape}")
        return quantized_st, vq_loss, indices_reshaped

    def get_codebook_entry(self, indices):

        batch_size = indices.shape[0]
        indices_flatten = indices.reshape(-1) 
        quantized = self.embedding(indices_flatten) 


        if len(indices.shape) >= 2: 
            C = quantized.shape[-1]
            spatial_shape = indices.shape[1:] 
            quantized = quantized.view(batch_size, *spatial_shape, C)
            if len(indices.shape) == 2: 
                quantized = quantized.permute(0, 2, 1).contiguous()
            elif len(indices.shape) == 3:
                 quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized




class Encoder1D(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_res_blocks, res_channels, downsample_factor=2, num_downsample_layers_top=2):
        super().__init__()
        ks = downsample_factor * 2 # Kernel size often related to stride

        # --- Bottom Level Encoder ---
        bottom_layers = []
        # Initial downsampling (e.g., by 4 if downsample_factor=2, done twice)
        bottom_layers.append(nn.Conv1d(in_channels, hidden_channels // 2, kernel_size=ks, stride=downsample_factor, padding=ks//2 - downsample_factor//2)) # Halve length
        bottom_layers.append(nn.ReLU(inplace=True))
        bottom_layers.append(nn.Conv1d(hidden_channels // 2, hidden_channels, kernel_size=ks, stride=downsample_factor, padding=ks//2 - downsample_factor//2)) # Halve length again
        bottom_layers.append(nn.ReLU(inplace=True))
        bottom_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)) # Adjust channels

        # Residual Blocks
        for _ in range(num_res_blocks):
            bottom_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels))
        bottom_layers.append(nn.ReLU(inplace=True)) # Activation before VQ projection
        self.encoder_b = nn.Sequential(*bottom_layers)

        # --- Top Level Encoder ---
        top_layers = []
        # Takes output of bottom encoder as input
        # Downsample further based on `num_downsample_layers_top`
        current_channels = hidden_channels
        for i in range(num_downsample_layers_top):
            out_ch = hidden_channels # Keep hidden_channels for simplicity, could change
            top_layers.append(nn.Conv1d(current_channels, out_ch, kernel_size=ks, stride=downsample_factor, padding=ks//2 - downsample_factor//2))
            top_layers.append(nn.ReLU(inplace=True))
            print(f"Encoder1D top layer {i} output channels: {out_ch}")
            current_channels = out_ch

        top_layers.append(nn.Conv1d(current_channels, hidden_channels, kernel_size=3, stride=1, padding=1)) # Adjust channels before ResBlocks

        # Residual Blocks
        for _ in range(num_res_blocks):
            top_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels))
        top_layers.append(nn.ReLU(inplace=True)) # Activation before VQ projection
        self.encoder_t = nn.Sequential(*top_layers)


    def forward(self, x):

        encoded_b = self.encoder_b(x) 

        encoded_t = self.encoder_t(encoded_b) 

        return encoded_b, encoded_t 



class Decoder1D(nn.Module):

    def __init__(self, out_channels, hidden_channels, num_res_blocks, res_channels, embedding_dim, upsample_factor=2, num_upsample_layers_top=2):
        super().__init__()
        ks = upsample_factor * 2 # Kernel size for transpose conv


        top_layers = []

        top_layers.append(nn.Conv1d(embedding_dim, hidden_channels, kernel_size=3, stride=1, padding=1))

        for _ in range(num_res_blocks):
            top_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels))


        current_channels = hidden_channels
        for i in range(num_upsample_layers_top):
             out_ch = hidden_channels # Keep hidden_channels for simplicity
             top_layers.append(nn.ConvTranspose1d(current_channels, out_ch, kernel_size=ks, stride=upsample_factor, padding=ks//2 - upsample_factor//2))
             if i < num_upsample_layers_top - 1: # Don't apply final ReLU before concatenation
                 top_layers.append(nn.ReLU(inplace=True))
             current_channels = out_ch

        self.decoder_t = nn.Sequential(*top_layers)

        bottom_layers = []
        # Input channels = bottom embedding dim + hidden channels from processed top latent
        bottom_layers.append(nn.Conv1d(embedding_dim + hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
        # Residual Blocks
        for _ in range(num_res_blocks):
            bottom_layers.append(ResidualBlock1D(hidden_channels, hidden_channels, res_channels))

        # Upsample back to original image resolution (two initial upsamples corresponding to encoder)
        bottom_layers.append(nn.ConvTranspose1d(hidden_channels, hidden_channels // 2, kernel_size=ks, stride=upsample_factor, padding=ks//2 - upsample_factor//2))
        bottom_layers.append(nn.ReLU(inplace=True))
        bottom_layers.append(nn.ConvTranspose1d(hidden_channels // 2, out_channels, kernel_size=ks, stride=upsample_factor, padding=ks//2 - upsample_factor//2))
        # Potentially add a final activation like Tanh if data is normalized to [-1, 1]

        self.decoder_b = nn.Sequential(*bottom_layers)

    def forward(self, z_q_b, z_q_t):
        #print(f"Decoder1D bottom latent shape: {z_q_b.shape}")
        #print(f"Decoder1D top latent shape: {z_q_t.shape}")
        decoded_t = self.decoder_t(z_q_t) # Shape: (B, HiddenCh, L_b)
        #print(f"Decoder1D top decoded shape: {decoded_t.shape}")

        # Ensure sequence lengths match before concatenation
        if decoded_t.shape[2] != z_q_b.shape[2]:
             target_len = z_q_b.shape[2]
             decoded_t = F.interpolate(decoded_t, size=target_len, mode='nearest') # Use 'linear' or 'nearest'

        # Concatenate bottom latents and upsampled top latents along channel dimension
        combined = torch.cat([z_q_b, decoded_t], dim=1) # Shape: (B, EmbDim + HiddenCh, L_b)
        #print(f"Decoder1D combined shape: {combined.shape}")

        # Decode combined latents
        reconstructed_x = self.decoder_b(combined) # Shape: (B, OutCh, L_original)

        return reconstructed_x



