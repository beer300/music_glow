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

def wav2spectrum(filename, target_duration=TARGET_DURATION, return_full=False):
    """
    Convert a WAV file to its spectrogram representation.
    
    Args:
        filename (str): Path to the WAV file.
        target_duration (int): Target duration in seconds for the x sample.
        return_full (bool): Whether to return the full audio spectrogram.
    
    Returns:
        tuple: (x_spectrum, y_spectrum, sr) where:
            - x_spectrum is the first target_duration seconds
            - y_spectrum is the full audio (if return_full=True)
            - sr is the sampling rate
    """
    x, sr = librosa.load(filename, sr=44100)
    target_length = target_duration * sr

    # Create x sample (first target_duration seconds)
    if len(x) > target_length:
        x_short = x[:target_length]
    else:
        x_short = np.pad(x, (0, target_length - len(x)), mode='constant')

    # Compute spectrograms
    S_x = librosa.stft(x_short, n_fft=N_FFT)
    S_x = np.log1p(np.abs(S_x))

    if return_full:
        # Compute full audio spectrogram
        S_y = librosa.stft(x, n_fft=N_FFT)
        S_y = np.log1p(np.abs(S_y))
        return S_x, S_y, sr
    
    return S_x, None, sr
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
    def __init__(self, data_path, transform=None, target_length=2584):
        """
        Dataset for loading WAV files and converting them to spectrograms.
        
        Args:
            data_path (str): Path to the directory containing WAV files.
            transform (callable, optional): Optional transform to apply to the spectrogram.
        """
        self.data_path = data_path
        self.filenames = [f for f in os.listdir(data_path) if f.endswith('.wav')]
        self.transform = transform
        self.target_length = target_length
    def __len__(self):
        return len(self.filenames)
    def pad_or_trim_spectrogram(self, spec, target_length):
        """Pad or trim spectrogram to target length in time dimension"""
        curr_length = spec.shape[-1]
        if curr_length > target_length:
            # Trim
            return spec[..., :target_length]
        elif curr_length < target_length:
            # Pad
            pad_length = target_length - curr_length
            return np.pad(spec, ((0, 0), (0, pad_length)), mode='constant')
        return spec
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
        Get a pair of spectrograms (x: first 8 seconds, y: full audio) and filename.
        
        Args:
            idx (int): Index of the file.
        
        Returns:
            tuple: (x_spectrum, y_spectrum, filename) where:
                - x_spectrum is tensor of first 8 seconds
                - y_spectrum is tensor of full audio
                - filename is the audio filename
        """
        filename = self.filenames[idx]
        filepath = os.path.join(self.data_path, filename)
        
        # Convert MP3 to WAV if necessary
        if filename.lower().endswith('.mp3'):
            filepath = self.convert_mp3_to_wav(filepath)
        
        # Get both short and full spectrograms
        x_spectrum, y_spectrum, _ = wav2spectrum(filepath, return_full=True)
        y_spectrum = self.pad_or_trim_spectrogram(y_spectrum, self.target_length)
        if self.transform:
            x_spectrum = self.transform(x_spectrum)
            if y_spectrum is not None:
                y_spectrum = self.transform(y_spectrum)
        
        # Convert to PyTorch tensors
        x_spectrum = torch.from_numpy(x_spectrum)[None, :, :].float()
        if y_spectrum is not None:
            y_spectrum = torch.from_numpy(y_spectrum)[None, :, :].float()
        
        return x_spectrum, y_spectrum, self.filenames[idx]

# Example usage
# Example usage
if __name__ == "__main__":
    data_path = r"C:\Users\lukas\Music\youtube_playlist_chopped"
    dataset = WavDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for batch in dataloader:
        x_specs, y_specs, filenames = batch
        print(f"X spectrograms shape: {x_specs.shape}")  # First 8 seconds
        print(f"Y spectrograms shape: {y_specs.shape}")  # Full audio
        print(f"Filenames: {filenames}")
        break