from pydub import AudioSegment
import os

INPUT_DIR = "../../audio_data_raw/raw/negative"
OUTPUT_DIR = "../../audio_data_raw/raw/new_neg_temp"

input_folder = INPUT_DIR
output_folder = OUTPUT_DIR


os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".flac"):
        flac_path = os.path.join(input_folder, file)

        wav_name = file.replace(".flac", ".wav")
        wav_path = os.path.join(output_folder, wav_name)

        audio = AudioSegment.from_file(flac_path, format="flac")
        audio.export(wav_path, format="wav")

        print(f"Converted: {file} -> {wav_name}")

print("All done 🎧")