import torch
import torchaudio # For loading audio files
import torchaudio.transforms as T # For potential resampling
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import glob # For finding files
import random
import numpy as np # If loading .npy files

# --- Custom Dataset for 1D Sequences ---

class SequenceDataset1D(Dataset):
    """
    Custom PyTorch Dataset for loading 1D sequences (e.g., audio).

    Handles loading files, ensuring consistent length via random cropping or padding,
    and converting to the desired channel format.

    Args:
        data_dir (str): Path to the directory containing sequence files.
        sequence_length (int): The fixed length required for each sequence fed to the model.
        channels (int): The desired number of channels (e.g., 1 for mono, 2 for stereo).
        file_extension (str): Extension of the data files (e.g., '.wav', '.npy').
        target_sample_rate (int, optional): If loading audio, resample to this rate. Defaults to None.
        transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
    """
    def __init__(self, data_dir, sequence_length, channels=1, file_extension='.wav', target_sample_rate=None, transform=None, data_dir_2=r"C:\Users\lukas\Documents\datasets\wavenet_dataset\target_audio"):
        super().__init__()
        self.data_dir = data_dir
        self.data_dir_2 = data_dir_2
        self.sequence_length = sequence_length
        self.channels = channels
        self.file_extension = file_extension.lower()
        self.target_sample_rate = target_sample_rate
        self.transform = transform

        # Find all files with the specified extension
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, f'*{self.file_extension}')))
        self.file_paths_2 = sorted(glob.glob(os.path.join(data_dir_2, f'*{self.file_extension}')))
        if not self.file_paths:
            raise FileNotFoundError(f"No files with extension '{self.file_extension}' found in directory '{data_dir}'.")

        self.resampler = None
        if self.target_sample_rate and self.file_extension == '.wav':
            # Initialized later inside __getitem__ after knowing the original sample rate
            pass

    def __len__(self):
        """Returns the total number of sequence files."""
        return len(self.file_paths)

    def __getitem__(self, index):
        """Loads and processes a single sequence file."""
        file_path = self.file_paths[index]
        file_path_2 =self.file_paths_2[index]
        waveform = None
        original_sr = None
        waveform_2 = None
        # --- Load Data ---
        try:
            if self.file_extension == '.wav':
                waveform, original_sr = torchaudio.load(file_path)
                waveform_2, _= torchaudio.load(file_path_2)
            elif self.file_extension == '.npy':
                # Assuming .npy stores a (channels, length) or (length,) array
                data = np.load(file_path)
                data_2 = np.load(file_path_2)
                if data.ndim == 1: # Add channel dimension if missing
                    data = data[np.newaxis, :]
                    data_2 = data_2[np.newaxis, :]
                waveform = torch.from_numpy(data).float()
                waveform_2 = torch.from_numpy(data_2).float()
                original_sr = -1 # Indicate sample rate doesn't apply or is unknown
            else:
                raise NotImplementedError(f"Loading for extension {self.file_extension} not implemented.")

        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            print(f"Error loading file {file_path_2}: {e}")
            # Return a dummy tensor of the correct shape or handle appropriately
            # Here, we'll return zeros. In a real scenario, you might skip the file.
            return torch.zeros((self.channels, self.sequence_length))

        # --- Handle Sample Rate (for Audio) ---
        if self.target_sample_rate and original_sr is not None and original_sr != self.target_sample_rate and original_sr != -1:
            if self.resampler is None or self.resampler.orig_freq != original_sr:
                 # Initialize resampler dynamically based on file's original sample rate
                 self.resampler = T.Resample(orig_freq=original_sr, new_freq=self.target_sample_rate)
            waveform = self.resampler(waveform)
            waveform_2 = self.resampler(waveform_2)


        # --- Handle Channels ---
        num_input_channels = waveform.shape[0]
        if num_input_channels != self.channels:
            if num_input_channels == 1 and self.channels == 2: # Mono to Stereo
                waveform = waveform.repeat(2, 1)
            elif num_input_channels > 1 and self.channels == 1: # Stereo/Multi-channel to Mono
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            else:
                # Other cases might need specific handling
                 raise ValueError(f"Cannot convert from {num_input_channels} channels to {self.channels} channels for file {file_path}.")
        num_input_channels = waveform_2.shape[0]
        if num_input_channels != self.channels:
            if num_input_channels == 1 and self.channels == 2: # Mono to Stereo
                waveform_2 = waveform_2.repeat(2, 1)
            elif num_input_channels > 1 and self.channels == 1: # Stereo/Multi-channel to Mono
                waveform_2 = torch.mean(waveform_2, dim=0, keepdim=True)
            else:
                # Other cases might need specific handling
                 raise ValueError(f"Cannot convert from {num_input_channels} channels to {self.channels} channels for file {file_path}.")


        # --- Handle Sequence Length ---
        current_length = waveform.shape[1]
        current_length_2 = waveform_2.shape[1]

        if current_length == self.sequence_length:
            processed_waveform = waveform
        elif current_length > self.sequence_length:
            # Random crop
            start_idx = random.randint(0, current_length - self.sequence_length)
            processed_waveform = waveform[:, start_idx : start_idx + self.sequence_length]
        else: # current_length < self.sequence_length
            # Pad with zeros
            padding_needed = self.sequence_length - current_length
            # Pad evenly on both sides (or just one side if preferred)
            pad_left = padding_needed // 2
            pad_right = padding_needed - pad_left
            # F.pad format: (pad_left, pad_right) for the last dimension
            processed_waveform = F.pad(waveform, (pad_left, pad_right))

        if current_length_2 == self.sequence_length:
            processed_waveform_2 = waveform_2
        elif current_length_2 > self.sequence_length:
            # Random crop
            start_idx = random.randint(0, current_length_2 - self.sequence_length)
            processed_waveform_2 = waveform_2[:, start_idx : start_idx + self.sequence_length]
        else: # current_length < self.sequence_length
            # Pad with zeros
            padding_needed = self.sequence_length - current_length_2
            # Pad evenly on both sides (or just one side if preferred)
            pad_left = padding_needed // 2
            pad_right = padding_needed - pad_left
            # F.pad format: (pad_left, pad_right) for the last dimension
            processed_waveform_2 = F.pad(waveform_2, (pad_left, pad_right))
        # --- Apply Transforms ---
        if self.transform:
            processed_waveform = self.transform(processed_waveform)
            processed_waveform_2 = self.transform(processed_waveform_2)

        # --- Basic Normalization (Example) ---
        # You might want more sophisticated normalization
        # Normalize to [-1, 1] based on max absolute value
        max_val = torch.max(torch.abs(processed_waveform))
        max_val_2 = torch.max(torch.abs(processed_waveform_2))
        if max_val > 1e-6: # Avoid division by zero
            processed_waveform = processed_waveform / max_val
        if max_val > 1e-6:
            processed_waveform_2 = processed_waveform_2 / max_val_2

        # Ensure final shape is correct
        assert processed_waveform.shape == (self.channels, self.sequence_length), \
            f"Final shape mismatch for {file_path}. Got {processed_waveform.shape}, expected {(self.channels, self.sequence_length)}"
        assert processed_waveform_2.shape == (self.channels, self.sequence_length), \
            f"Final shape mismatch for {file_path}. Got {processed_waveform_2.shape}, expected {(self.channels, self.sequence_length)}"

        return processed_waveform, processed_waveform_2

# --- Function to Create DataLoader ---

def create_dataloader(data_dir, data_dir_2, sequence_length, batch_size,
                      channels=1, file_extension='.wav', target_sample_rate=None,
                      transform=None, num_workers=0, shuffle=True, pin_memory=True, drop_last=True):

    dataset = SequenceDataset1D(
        data_dir=data_dir,
        sequence_length=sequence_length,
        channels=channels,
        file_extension=file_extension,
        target_sample_rate=target_sample_rate,
        transform=transform,
        data_dir_2=data_dir_2
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last # Often important for models expecting fixed batch sizes
    )

    print(f"Created DataLoader with {len(dataset)} samples.")
    return dataloader

# --- Example Usage ---

if __name__ == '__main__':
    # --- Create Dummy Data (if you don't have any) ---
    DUMMY_DATA_DIR = './dummy_audio_data'
    NUM_DUMMY_FILES = 50
    DUMMY_SEQ_LEN_MIN = 8000
    DUMMY_SEQ_LEN_MAX = 24000
    DUMMY_SAMPLE_RATE = 16000 # Sample rate for dummy files

    if not os.path.exists(DUMMY_DATA_DIR):
        print(f"Creating dummy data directory: {DUMMY_DATA_DIR}")
        os.makedirs(DUMMY_DATA_DIR)
        for i in range(NUM_DUMMY_FILES):
            file_path = os.path.join(DUMMY_DATA_DIR, f'dummy_{i:03d}.wav')
            length = random.randint(DUMMY_SEQ_LEN_MIN, DUMMY_SEQ_LEN_MAX)
            # Create random audio-like data (sine wave + noise)
            time = torch.linspace(0, length / DUMMY_SAMPLE_RATE, length)
            waveform = (torch.sin(2 * torch.pi * 440 * time) * 0.5 + # 440 Hz tone
                        torch.randn(length) * 0.1)      # Add some noise
            waveform = waveform.unsqueeze(0) # Add channel dimension (mono)
            torchaudio.save(file_path, waveform, DUMMY_SAMPLE_RATE)
        print(f"Created {NUM_DUMMY_FILES} dummy .wav files.")
    else:
        print(f"Dummy data directory '{DUMMY_DATA_DIR}' already exists.")

    # --- DataLoader Parameters ---
    TARGET_SEQ_LENGTH = 4096   # Must be compatible with VQ-VAE-2 model's downsampling
    BATCH_SIZE = 16
    NUM_CHANNELS = 1           # Mono audio
    TARGET_SR = 16000          # Resample dummy data to this rate (if different)

    # --- Create the DataLoader ---
    print("\nCreating DataLoader...")
    train_loader = create_dataloader(
        data_dir=DUMMY_DATA_DIR,
        sequence_length=TARGET_SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        channels=NUM_CHANNELS,
        file_extension='.wav',
        target_sample_rate=TARGET_SR, # Matches dummy data SR, so no resampling needed here
        num_workers=2,             # Use 2 worker processes for loading
        shuffle=True
    )

    # --- Test the DataLoader ---
    print("\nFetching a sample batch...")
    try:
        sample_batch = next(iter(train_loader))
        print(f"Batch shape: {sample_batch.shape}") # Expected: (BATCH_SIZE, NUM_CHANNELS, TARGET_SEQ_LENGTH)
        print(f"Batch data type: {sample_batch.dtype}")
        print(f"Batch min value: {sample_batch.min():.4f}")
        print(f"Batch max value: {sample_batch.max():.4f}")

        # Verify shape
        expected_shape = (BATCH_SIZE, NUM_CHANNELS, TARGET_SEQ_LENGTH)
        assert sample_batch.shape == expected_shape, f"Expected shape {expected_shape}, but got {sample_batch.shape}"
        print("Batch shape is correct.")

    except StopIteration:
        print("DataLoader is empty (maybe no files found or issue during loading).")
    except Exception as e:
        print(f"An error occurred while fetching a batch: {e}")