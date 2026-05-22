import os
import numpy as np
import librosa
from tqdm import tqdm
from util_features import extract_features

# =========================
# CONFIG
# =========================
INPUT_DIR = "../../audio_data_raw/raw/positive"
OUTPUT_DIR = "../../audio_data_processed/positive"

SR = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def preprocess():

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".wav", ".flac", ".webm"))
    ]

    print(f"🔥 Positives: {len(files)} files")

    count = 0

    for f in tqdm(files):

        try:
            path = os.path.join(INPUT_DIR, f)

            y = load_audio(path)

            feat = extract_features(y)

            assert feat.shape == (1, 40, 200), f"Bad shape {feat.shape}"

            out_path = os.path.join(
                OUTPUT_DIR,
                f.rsplit(".", 1)[0] + ".npy"
            )

            np.save(out_path, feat)

            count += 1

        except Exception as e:
            print("skip:", f, e)

    print("✔ positives saved:", count)


if __name__ == "__main__":
    preprocess()