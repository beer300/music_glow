import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from packaging import version
import soundfile
from pydub import AudioSegment
import tempfile

N_FFT = 2048  # Define the FFT size
TARGET_DURATION = 8  # Target duration in seconds

def wav2spectrum(filename, target_duration=TARGET_DURATION):
    """
    Convert a WAV file to its spectrogram representation, ensuring it is exactly `target_duration` seconds long.
    """
    x, sr = librosa.load(filename, sr=44100)
    target_length = target_duration * sr
    if len(x) > target_length:
        x = x[:target_length]
    elif len(x) < target_length:
        x = np.pad(x, (0, target_length - len(x)), mode='constant')
    S = librosa.stft(x, n_fft=N_FFT)
    S = np.log1p(np.abs(S))
    return S, sr

def spectrum2wav(spectrum, sr, outfile):
    a = np.exp(spectrum) - 1
    p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, n_fft=N_FFT))
    librosa_write(outfile, x, sr)

def librosa_write(outfile, x, sr):
    if version.parse(librosa.__version__) < version.parse('0.8.0'):
        librosa.output.write_wav(outfile, x, sr)
    else:
        soundfile.write(outfile, x, sr)

class WavPairDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.input_path = os.path.join(data_path, "input_audio")
        self.target_path = os.path.join(data_path, "target_audio")
        self.input_files = sorted([f for f in os.listdir(self.input_path) if f.endswith('.wav')])
        self.target_files = sorted([f for f in os.listdir(self.target_path) if f.endswith('.wav')])
        assert len(self.input_files) == len(self.target_files), "Input and target folders must have the same number of files."
        self.transform = transform
        self.temp_dir = tempfile.mkdtemp()

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_filename = self.input_files[idx]
        target_filename = self.target_files[idx]
        input_filepath = os.path.join(self.input_path, input_filename)
        target_filepath = os.path.join(self.target_path, target_filename)

        input_spectrum, _ = wav2spectrum(input_filepath)
        target_spectrum, _ = wav2spectrum(target_filepath)

        if self.transform:
            input_spectrum = self.transform(input_spectrum)
            target_spectrum = self.transform(target_spectrum)

        input_spectrum = torch.from_numpy(input_spectrum)[None, :, :].float()
        target_spectrum = torch.from_numpy(target_spectrum)[None, :, :].float()
        return input_spectrum, target_spectrum, (input_filename, target_filename)

# Example usage
if __name__ == "__main__":
    # Path to the directory containing input_audio and target_audio subfolders
    data_path = r"C:\Users\lukas\Documents\datasets\wavenet_dataset"

    dataset = WavPairDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for batch in dataloader:
        input_spectrograms, target_spectrograms, filenames = batch
        print(f"Input Spectrograms shape: {input_spectrograms.shape}")
        print(f"Target Spectrograms shape: {target_spectrograms.shape}")
        print(f"Filenames: {filenames}")