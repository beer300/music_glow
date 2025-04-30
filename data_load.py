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
#([8, 64, 256, 646])
#([8, 64, 256, 172])
def wav2spectrum(filename, target_duration=TARGET_DURATION):
    """
    Convert a WAV file to its spectrogram representation, ensuring it is exactly `target_duration` seconds long.
    
    Args:
        filename (str): Path to the WAV file.
        target_duration (int): Target duration in seconds.
    
    Returns:
        spectrum (np.ndarray): Log-scaled spectrogram.
        sr (int): Sampling rate of the audio.
    """
    x, sr = librosa.load(filename, sr=44100)  # Load audio with its original sampling rate

    target_length = target_duration * sr  # Calculate the target length in samples

    # Trim or pad the audio to the target length
    if len(x) > target_length:
        x = x[:target_length]  # Trim to target length
    elif len(x) < target_length:
        x = np.pad(x, (0, target_length - len(x)), mode='constant')  # Pad with zeros
    #print(f"sr: {sr}, x.shape: {x.shape}, target_length: {target_length}")
    # Compute the spectrogram
    S = librosa.stft(x, n_fft=N_FFT)
    S = np.log1p(np.abs(S))  # Log-scaled magnitude
    return S, sr
def spectrum2wav(spectrum, sr, outfile):
    # Return the all-zero vector with the same shape of `a_content`
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
class WavDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        Dataset for loading WAV files and converting them to spectrograms.
        
        Args:
            data_path (str): Path to the directory containing WAV files.
            transform (callable, optional): Optional transform to apply to the spectrogram.
        """
        self.data_path = data_path
        self.filenames = [f for f in os.listdir(data_path) if f.endswith('.wav')]
        self.transform = transform
        self.temp_dir = tempfile.mkdtemp()  # Create temporary directory for WAV conversions
    def __len__(self):
        return len(self.filenames)
    
    def convert_mp3_to_wav(self, mp3_path):
        """Convert MP3 to WAV format"""
        wav_path = os.path.join(self.temp_dir, 
                               os.path.splitext(os.path.basename(mp3_path))[0] + '.wav')
        if not os.path.exists(wav_path):
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format='wav')
        return wav_path
    def __getitem__(self, idx):
        """
        Get a spectrogram and its corresponding filename.
        
        Args:
            idx (int): Index of the file.
        
        Returns:
            torch.Tensor: Spectrogram as a tensor.
            str: Filename of the WAV file.
        """
        filepath = os.path.join(self.data_path, self.filenames[idx])
        spectrum, _ = wav2spectrum(filepath)
        
        if self.transform:
            spectrum = self.transform(spectrum)
        
        # Convert to PyTorch tensor
        spectrum = torch.from_numpy(spectrum)[None, :, :].float()  # Add channel dimension
        return spectrum, self.filenames[idx]

# Example usage
if __name__ == "__main__":
    # Path to the directory containing WAV files
    data_path = r"C:\Users\lukas\Music\youtube_playlist_chopped"

    # Create the dataset and dataloader
    dataset = WavDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Iterate through the dataloader
    for batch in dataloader:
        spectrograms, filenames = batch
        print(f"Spectrograms shape: {spectrograms.shape}")
        print(f"Filenames: {filenames}")