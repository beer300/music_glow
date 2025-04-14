import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from wavenet_1_dataloader import WaveNetDataset  # Corrected import
import torch.optim as optim  # Import optimizer
import numpy as np  # Import for saving audio
from scipy.io.wavfile import write  # Import for saving audio as WAV

class WaveNet(nn.Module):
    def __init__(self, input_channels, residual_channels, dilation_channels, skip_channels, kernel_size, num_classes, num_layers):
        super(WaveNet, self).__init__()
        self.input_channels = input_channels
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Initial causal convolution
        self.causal_conv = nn.Conv1d(input_channels, residual_channels, kernel_size, padding=kernel_size - 1)

        # Dilated convolution layers with gated activation units
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i
            self.dilated_convs.append(
                nn.Conv1d(residual_channels, 2 * dilation_channels, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation)
            )
            self.residual_convs.append(nn.Conv1d(dilation_channels, residual_channels, 1))
            self.skip_convs.append(nn.Conv1d(dilation_channels, skip_channels, 1))

        # Output layers
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.output_conv2 = nn.Conv1d(skip_channels, num_classes, 1)

    def forward(self, x):
        # Initial causal convolution
        x = self.causal_conv(x)
        x = x[:, :, :-self.kernel_size + 1]  # Remove padding to ensure causality
        skip_connections = []

        # Dilated convolutions with gated activation units
        for dilated_conv, residual_conv, skip_conv in zip(self.dilated_convs, self.residual_convs, self.skip_convs):
            residual = x
            x = dilated_conv(x)
            x = x[:, :, :residual.size(2)]  # Adjust size to match residual
            gate, filter = torch.chunk(x, 2, dim=1)  # Split into gate and filter
            x = torch.tanh(filter) * torch.sigmoid(gate)  # Gated activation unit
            skip = skip_conv(x)
            skip_connections.append(skip)
            x = residual_conv(x) + residual

        # Combine skip connections
        x = sum(skip_connections)
        x = F.relu(x)

        # Output layers
        x = self.output_conv1(x)
        x = F.relu(x)
        x = self.output_conv2(x)
        x = F.softmax(x, dim=1)

        return x
# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_channels = 1
    residual_channels = 32
    dilation_channels = 32
    skip_channels = 32
    kernel_size = 2
    num_classes = 256  # For categorical distribution
    num_layers = 10
    learning_rate = 0.001
    num_epochs = 5

    # Create model
    model = WaveNet(input_channels, residual_channels, dilation_channels, skip_channels, kernel_size, num_classes, num_layers)

    # Path to audio file or directory
    audio_path = r"C:\Users\lukas\Music\youtube_playlist_chopped"

    # Initialize dataset and dataloader
    sequence_length = 16000  # 1 second of audio at 16kHz
    batch_size = 1
    dataset = WaveNetDataset(audio_path, sequence_length, num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
# Training loop
    model.train()  # Set model to training mode
    step = 0  # Initialize step counter
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.float()  # Convert inputs to float

            # Forward pass
            outputs = model(inputs)

            # Reshape outputs and targets for CrossEntropyLoss
            outputs = outputs.permute(0, 2, 1)  # Shape: (batch_size, sequence_length, num_classes)
            targets = targets.view(-1)  # Flatten targets

            # Compute loss
            loss = criterion(outputs.view(-1, num_classes), targets)

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Accumulate loss
            epoch_loss += loss.item()

            # Save audio every 500 steps
            if step % 500 == 0:
                # Convert model output to audio signal
                predicted_audio = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
                audio_path = f"generated_audio_step_{step}.wav"
                write(audio_path, 16000, predicted_audio.astype(np.int16))  # Save as 16-bit PCM WAV
                print(f"Saved audio at step {step}: {audio_path}")

            step += 1  # Increment step counter

        # Print epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")