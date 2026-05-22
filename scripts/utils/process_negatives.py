import os
import numpy as np
import librosa
from tqdm import tqdm
from util_features import extract_features

# =========================
# CONFIG
# =========================
INPUT_DIR = "../../audio_data_raw/raw/negative"
OUTPUT_DIR = "../../audio_data_processed/negative"
SR = 16000
CHUNK_SEC = 2
CHUNK = SR * CHUNK_SEC

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def process():

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".wav", ".flac", ".mp3"))
    ]

    print(f"💀 Negatives: {len(files)} files")

    count = 0

    for f in tqdm(files):

        try:
            path = os.path.join(INPUT_DIR, f)

            y = load_audio(path)

            # chunk into 2-sec windows
            n_chunks = len(y) // CHUNK

            for i in range(n_chunks):

                chunk = y[i*CHUNK:(i+1)*CHUNK]

                feat = extract_features(chunk)

                assert feat.shape == (1, 40, 200), f"Bad shape {feat.shape}"

                out_path = os.path.join(
                OUTPUT_DIR,
                f.rsplit(".", 1)[0] + ".npy"
                )

                np.save(out_path, feat)

                count += 1

        except Exception as e:
            print("skip:", f, e)

    print("✔ negatives saved:", count)


if __name__ == "__main__":
    process()