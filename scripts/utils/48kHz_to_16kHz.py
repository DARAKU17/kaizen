import os
import librosa
import soundfile as sf

INPUT_DIR = "data/test_samples"
OUTPUT_DIR = "data/raw/positive"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_audio(input_path, output_path):
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    sf.write(output_path, y, 16000)

def main():
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".wav") or file.endswith(".mp3"):
            input_path = os.path.join(INPUT_DIR, file)
            output_path = os.path.join(OUTPUT_DIR, file.replace(".mp3", ".wav"))

            print(f"Converting: {file}")
            convert_audio(input_path, output_path)

    print("\nDone converting all files.")

if __name__ == "__main__":
    main()