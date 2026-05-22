import os
import numpy as np

def fix_folder(path):
    for f in os.listdir(path):
        if not f.endswith(".npy"):
            continue

        fp = os.path.join(path, f)
        x = np.load(fp)

        # force 2D
        x = np.asarray(x)

        if x.ndim != 2:
            print("Skipping bad shape:", f, x.shape)
            continue

        # fix orientation
        if x.shape[0] == 120:
            x = x.T  # (120, T) → (T, 120)

        if x.shape[1] != 120:
            print("Bad feature dim:", f, x.shape)
            continue

        np.save(fp, x.astype(np.float32))

fix_folder("../../audio_data_processed/positive")
fix_folder("../../audio_data_processed/negative")