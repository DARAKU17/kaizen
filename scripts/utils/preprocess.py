import os
import numpy as np
from tqdm import tqdm
from features import extract_features  # or same file

def preprocess(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

    print(f"Processing {len(files)} files in {input_dir}")

    for file in tqdm(files):
        in_path = os.path.join(input_dir, file)

        try:
            features = extract_features(in_path)

            out_name = file.replace(".wav", ".npy")
            out_path = os.path.join(output_dir, out_name)

            np.save(out_path, features)

        except Exception as e:
            print(f"Failed on {file}: {e}")

if __name__ == "__main__":
    preprocess("data/raw/positive", "data/processed/positive")
    preprocess("data/raw/negative", "data/processed/negative")

    print("Preprocessing complete ✔")