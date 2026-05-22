import os
import numpy as np

DIRS = [
    "../../audio_data_processed/positive",
    "../../audio_data_processed/negative"
]

bad = []

for folder in DIRS:
    print(f"\nScanning {folder}")

    for f in os.listdir(folder):
        if not f.endswith(".npy"):
            continue

        path = os.path.join(folder, f)

        try:
            x = np.load(path)

            if x.size == 0:
                raise ValueError("empty")

        except Exception as e:
            print("❌ removing:", path, e)
            bad.append(path)
            os.remove(path)

print(f"\nDone. Removed {len(bad)} bad files.")