import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from wavenet_1_dataloader import WaveNetDataset  # Corrected import
import torch.optim as optim  # Import optimizer
import numpy as np  # Import for saving audio
from scipy.io.wavfile import write  # Import for saving audio as WAV
from tqdm import tqdm
import matplotlib.pyplot as plt 
import os

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
        self.dropout = nn.Dropout(0.2)
        for i in range(num_layers):
            dilation = 2 ** (i % 10)  # Example: Cycle dilations for very deep networks
            self.dilated_convs.append(
                nn.Conv1d(residual_channels, 2 * dilation_channels, kernel_size, 
                        dilation=dilation, padding=(kernel_size - 1) * dilation)
            )
            self.residual_convs.append(nn.Conv1d(dilation_channels, residual_channels, 1))
            self.skip_convs.append(nn.Conv1d(dilation_channels, skip_channels, 1))

        # Output layers
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.output_conv2 = nn.Conv1d(skip_channels, num_classes, 1)
        for conv in self.dilated_convs:
            nn.init.normal_(conv.weight, mean=0.0, std=0.01)
            nn.init.constant_(conv.bias, 0.0)
            with torch.no_grad():

                conv.bias[conv.out_channels // 2:].fill_(1.0)
    def forward(self, x):
        # Initial causal convolution
        x = self.causal_conv(x)
        x = x[:, :, :-self.kernel_size + 1]  # Remove padding to ensure causality
        skip_connections = []

        # Dilated convolutions with gated activation units
        
        for dilated_conv, residual_conv, skip_conv in zip(self.dilated_convs, self.residual_convs, self.skip_convs):
            residual = x
            x = dilated_conv(x)
            x = x[:, :, :residual.size(2)]    # Adjust size to match residual
            gate, filter = torch.chunk(x, 2, dim=1)  # Split into gate and filter
            x = torch.tanh(filter) * torch.sigmoid(gate)  # Gated activation unit
            skip = skip_conv(x)
            skip_connections.append(skip)
            x = residual_conv(x) + residual
          
        # Combine skip connections
        x = sum(skip_connections)
        x = F.relu(x)
        x = self.dropout(x) 
        # Output layers
        x = self.output_conv1(x)
        x = F.relu(x)
        x = self.dropout(x) 
        x = self.output_conv2(x)
        #x = F.softmax(x, dim=1)

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
    audio_path = r"C:\Users\lukas\Music\chopped_30-20250108T164615Z-001\chopped_30"
    # Define base directories for saving files
    base_dir = r"C:\Users\lukas\Music\wavenet"
    audio_dir = os.path.join(base_dir, "audio")
    model_dir = os.path.join(base_dir, "model")
    loss_dir = os.path.join(base_dir, "loss")

    # Create directories if they don't exist
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
        # Initialize dataset and dataloader
    sequence_length = 16000*4*3  # 1 second of audio at 16kHz
    batch_size = 8
    dataset = WaveNetDataset(audio_path, sequence_length, num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)



    scaler = torch.amp.GradScaler("cuda")
    step_losses = []
    # Training loop
    model.train()  # Set model to training mode
    step = 0  # Initialize step counter
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        
        # Wrap the dataloader with tqdm for a progress bar
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Move model to the device
        model = model.to(device)

        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")
        for batch in progress_bar:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()  # Convert inputs to float

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda"):
                #print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
                outputs = model(inputs)
                # Reshape outputs and targets for CrossEntropyLoss
                #print(f"Model output shape: {outputs.shape}")
                outputs = outputs.permute(0, 2, 1)  # Shape: (batch_size, sequence_length, num_classes)
                outputs = outputs.reshape(-1, outputs.size(-1))  # Flatten outputs to (batch_size * sequence_length, num_classes)
                targets = targets.view(-1)  # Flatten targets to (batch_size * sequence_length)

                # Ensure outputs and targets have the same size
                min_size = min(outputs.size(0), targets.size(0))  # Find the smaller size
                outputs = outputs[:min_size]  # Trim outputs to match the smaller size
                targets = targets[:min_size]  # Trim targets to match the smaller size

                # Compute loss
                #print(f"Outputs shape: {outputs.shape}, Targets shape: {targets.shape}")

                loss = criterion(outputs, targets)

            # Backward pass and optimization with mixed precision
            optimizer.zero_grad()  # Clear gradients
            scaler.scale(loss).backward()  # Scale the loss for mixed precision
            scaler.step(optimizer)  # Update weights
            scaler.update()  # Update the scaler for next iteration

            # Accumulate loss
            epoch_loss += loss.item()

            # Update progress bar with current loss
            progress_bar.set_postfix(loss=loss.item())
            if step % 1 == 0:
                step_losses.append(loss.item())

            # Save audio every 200 steps
            if step % 200 == 0:
                # Reshape outputs back to [batch_size, sequence_length, num_classes]
                outputs = outputs.view(batch_size, -1, num_classes)  # Restore original shape

                # Convert model output to audio signal
                predicted_audio_encoded = torch.argmax(outputs[0], dim=1).cpu().numpy()  # Take the first batch

                # Decode the µ-law encoded audio
                predicted_audio = WaveNetDataset.mu_law_decode(predicted_audio_encoded, num_classes)
                print(f"Decoded audio shape: {predicted_audio.shape}")

                # Save the decoded audio as a WAV file
                audio_path = os.path.join(audio_dir, f"generated_audio_step_{step}.wav")
                write(audio_path, 16000, predicted_audio.astype(np.float32))  # Save as 32-bit float WAV
                print(f"Saved decoded audio at step {step}: {audio_path}")

                # Plot the loss graph
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(step_losses) + 1), step_losses, marker='o', label='Training Loss')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Training Loss Over Steps')
                plt.legend()
                plt.grid()

                # Save the loss graph
                loss_path = os.path.join(loss_dir, f"training_loss_step_{step}.png")
                plt.savefig(loss_path)  # Save as PNG file
                print(f"Saved loss graph at step {step}: {loss_path}")
                # plt.show()  # Show the plot after saving

            # Save the model every 500 steps
            if step % 500 == 0:
                model_save_path = os.path.join(model_dir, f"wavenet_model_step_{step}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at step {step}: {model_save_path}")

            step += 1  # Increment step counter

            # Print epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")