import os
import numpy as np
import librosa
from tqdm import tqdm
from features import extract_features

INPUT_DIR = "random"
OUTPUT_DIR = "random_processed"
SR = 16000
CHUNK_SEC = 2        # change to 2 if wanted
CHUNK = SR * CHUNK_SEC

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0

for file in tqdm(os.listdir(INPUT_DIR)):
    if not file.endswith((".wav", ".mp3", ".flac", ".m4a")):
        continue

    path = os.path.join(INPUT_DIR, file)

    try:
        y, _ = librosa.load(path, sr=SR, mono=True)

        n_chunks = len(y) // CHUNK

        for i in range(n_chunks):
            chunk = y[i*CHUNK:(i+1)*CHUNK]

            feat = extract_features(chunk)

            np.save(
                os.path.join(OUTPUT_DIR, f"neg2_{count}.npy"),
                feat
            )
            count += 1

    except Exception as e:
        print("skip:", file, e)

print(f"Done ✔ created {count} negatives")