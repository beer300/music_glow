import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import soundfile as sf  # Import soundfile for saving audio

class WaveNetDataset(Dataset):
    def __init__(self, audio_path, sequence_length, num_classes=256, sample_rate=16000):
        """
        Dataset for WaveNet with µ-law encoding using a single audio file.

        Args:
            audio_path (str): Path to the audio file or directory containing audio files.
            sequence_length (int): Length of input sequences.
            num_classes (int): Number of quantization levels (default: 256 for µ-law).
            sample_rate (int): Sampling rate for the audio file.
        """
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.sample_rate = sample_rate

        # Check if the path is a directory
        if os.path.isdir(audio_path):
            # Get the first audio file in the directory
            audio_files = [f for f in os.listdir(audio_path) if f.endswith(('.wav', '.mp3'))]
            if not audio_files:
                raise ValueError(f"No audio files found in directory: {audio_path}")
            audio_path = os.path.join(audio_path, audio_files[0])  # Use the first audio file

        # Load and preprocess the audio
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        audio = np.clip(audio, -1.0, 1.0)  # Normalize to [-1, 1]
        self.original_audio = audio  # Store original audio for comparison
        self.encoded_audio = self.mu_law_encode(audio, self.num_classes)

    def __len__(self):
        # Number of sequences that can be extracted from the audio
        return len(self.encoded_audio) - self.sequence_length

    def __getitem__(self, idx):
        # Extract input and target sequences
        input_sequence = self.encoded_audio[idx:idx + self.sequence_length]
        target_sequence = self.encoded_audio[idx + 1:idx + self.sequence_length + 1]

        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0)  # Shape: (1, sequence_length)
        target_tensor = torch.tensor(target_sequence, dtype=torch.long)  # Shape: (sequence_length)

        return input_tensor, target_tensor

    @staticmethod
    def mu_law_encode(audio, num_classes):
        """
        Apply µ-law encoding to audio.

        Args:
            audio (np.ndarray): Input audio array in range [-1, 1].
            num_classes (int): Number of quantization levels.

        Returns:
            np.ndarray: µ-law encoded audio.
        """
        mu = num_classes - 1
        encoded = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
        encoded = ((encoded + 1) / 2 * mu + 0.5).astype(np.int32)
        return encoded

    @staticmethod
    def mu_law_decode(encoded_audio, num_classes):
        """
        Decode µ-law encoded audio.

        Args:
            encoded_audio (np.ndarray): µ-law encoded audio.
            num_classes (int): Number of quantization levels.

        Returns:
            np.ndarray: Decoded audio in range [-1, 1].
        """
        mu = num_classes - 1
        audio = (encoded_audio / mu) * 2 - 1
        decoded = np.sign(audio) * (1 / mu) * ((1 + mu) ** np.abs(audio) - 1)
        return decoded

    def reconstruct_audio(self):
        """
        Reconstruct the audio from µ-law encoded data.

        Returns:
            np.ndarray: Reconstructed audio in range [-1, 1].
        """
        return self.mu_law_decode(self.encoded_audio, self.num_classes)

    def compare_audio(self):
        """
        Compare the original audio with the reconstructed audio.

        Returns:
            float: Mean Squared Error (MSE) between the original and reconstructed audio.
        """
        reconstructed_audio = self.reconstruct_audio()
        mse = np.mean((self.original_audio - reconstructed_audio) ** 2)
        return mse


# Example usage
if __name__ == "__main__":
    # Path to a directory or single audio file
    audio_path = r"C:\Users\lukas\Music\youtube_playlist_chopped"

    # Hyperparameters
    sequence_length = 16000  # 1 second of audio at 16kHz
    num_classes = 256
    batch_size = 4

    # Create dataset and dataloader
    dataset = WaveNetDataset(audio_path, sequence_length, num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Compare original and reconstructed audio
    mse = dataset.compare_audio()
    print(f"Mean Squared Error between original and reconstructed audio: {mse}")

    # Save reconstructed audio to desktop
    reconstructed_audio = dataset.reconstruct_audio()
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "reconstructed_audio.wav")
    sf.write(desktop_path, reconstructed_audio, samplerate=dataset.sample_rate)
    print(f"Reconstructed audio saved to: {desktop_path}")

    # Iterate through the dataloader
    for inputs, targets in dataloader:
        print(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
        break