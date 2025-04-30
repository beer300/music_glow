import os
from yt_dlp import YoutubeDL

def download_youtube_video(url, output_folder):
    """
    Downloads a YouTube video and extracts its audio as a WAV file using yt-dlp.
    
    Args:
        url (str): The URL of the YouTube video.
        output_folder (str): The folder where the WAV file will be saved.
    """
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Define yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
        }

        # Download and convert to WAV
        print(f"Downloading and converting video from {url}...")
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Downloaded and converted: {url}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # List of YouTube URLs
    youtube_urls = [
        "https://www.youtube.com/watch?v=qutfxAB36Zk",
        "https://www.youtube.com/watch?v=TMy_mYkwl4M",
        "https://www.youtube.com/watch?v=-tT32VTll5M",
        "https://www.youtube.com/watch?v=wfGuSP7PvW4",
        "https://www.youtube.com/watch?v=sbeCQes9IW0",
        "https://www.youtube.com/watch?v=7djdvzRcXmY",
        "https://www.youtube.com/watch?v=QgcBG82pfIg",
        "https://www.youtube.com/watch?v=Fb3A8m4iC9s",
        "https://www.youtube.com/watch?v=Ewuv_Wk_mOI",
        "https://www.youtube.com/watch?v=OpnB3qcDESI",
        "https://www.youtube.com/watch?v=sNY_2TEmzho"
    ]
    output_folder = r"C:\Users\lukas\Music\youtube_wav_files"  # Replace with your desired folder

    for url in youtube_urls:
        download_youtube_video(url, output_folder)