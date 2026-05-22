import os
import numpy as np
from tqdm import tqdm

TARGET_MELS = 40   # change to 64 if you want 192-dim features
FORCE_SHAPE = (3, TARGET_MELS, None)  # (C, M, T)


def convert_file(path, out_path):
    x = np.load(path)

    # -------------------------
    # CASE 1: (3, mel, time)
    # -------------------------
    if x.ndim == 3:
        c, m, t = x.shape

        # optional sanity check
        if m != TARGET_MELS:
            # resize by cropping or padding mel axis
            if m > TARGET_MELS:
                x = x[:, :TARGET_MELS, :]
            else:
                pad = TARGET_MELS - m
                x = np.pad(x, ((0,0),(0,pad),(0,0)))

        # (C, M, T) → (T, C*M)
        x = x.transpose(2, 0, 1)        # (T, C, M)
        x = x.reshape(x.shape[0], -1)   # (T, C*M)

    # -------------------------
    # CASE 2: already flat weird
    # -------------------------
    elif x.ndim == 2:
        pass  # assume already (time, feature)

    else:
        raise ValueError(f"Bad shape: {x.shape}")

    # normalize
    x = x.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-9)

    np.save(out_path, x)


def convert_folder(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(in_dir) if f.endswith(".npy")]

    print(f"Converting {len(files)} files...")

    for f in tqdm(files):
        try:
            convert_file(
                os.path.join(in_dir, f),
                os.path.join(out_dir, f)
            )
        except Exception as e:
            print(f"Failed {f}: {e}")


if __name__ == "__main__":
    convert_folder(
        "../../audio_data_processed/positive",
        "../../audio_data_processed/positive_fixed"
    )