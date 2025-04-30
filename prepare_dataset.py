import os
import numpy as np
import soundfile as sf
import librosa

def split_audio_into_input_target(wav_path, input_duration=9, target_offset=3, sample_rate=44100, output_dir="."):
    # Load audio file
    audio, sr = sf.read(wav_path)
    if sr != sample_rate:
        print(f"Resampling {wav_path} from {sr}Hz to {sample_rate}Hz...")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    input_len = int(input_duration * sr)
    offset_len = int(target_offset * sr)
    total_len = input_len + offset_len

    # Create output directories
    input_audio_dir = os.path.join(output_dir, "input_audio")
    target_audio_dir = os.path.join(output_dir, "target_audio")
    os.makedirs(input_audio_dir, exist_ok=True)
    os.makedirs(target_audio_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(wav_path))[0]

    count = 0
    for start in range(0, len(audio) - total_len + 1, input_len):
        input_segment = audio[start : start + input_len]
        target_segment = audio[start + offset_len : start + offset_len + input_len]

        input_path = os.path.join(input_audio_dir, f"{base_name}_input_{count:04d}.wav")
        target_path = os.path.join(target_audio_dir, f"{base_name}_target_{count:04d}.wav")

        sf.write(input_path, input_segment, sr)
        sf.write(target_path, target_segment, sr)

        count += 1

    print(f"Saved {count} input/target pairs for {wav_path} to {output_dir}.")

def process_folder(folder_path, input_duration=9, target_offset=3, sample_rate=44100, output_dir="."):
    # Iterate through all .wav files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            wav_path = os.path.join(folder_path, file_name)
            split_audio_into_input_target(
                wav_path, input_duration, target_offset, sample_rate, output_dir
            )

# Example usage
process_folder(
    r"C:\Users\lukas\Music\youtube_wav_files",
    output_dir=r"C:\Users\lukas\Documents\datasets\wavenet_dataset",
)