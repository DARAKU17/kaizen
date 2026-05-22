import os
import numpy as np
import librosa
import soundfile as sf
import random


# =========================
# CONFIG
# =========================
INPUT_DIR = "../../audio_data_processed/positive"
OUTPUT_DIR = "./reconstructed"

NUM_SAMPLES = 10
SR = 16000
N_MELS = 40

RANDOM_PICK = False   # True = random files


# =========================
# RECONSTRUCT
# =========================
def reconstruct_feature(path, out_path):
    x = np.load(path)

    x = np.squeeze(x)

    # expected (T,120)
    if x.ndim != 2:
        raise ValueError(f"Bad shape: {x.shape}")

    # if stored as (120,T), fix it
    if x.shape[0] == 120:
        x = x.T

    if x.shape[1] != 120:
        raise ValueError(f"Expected (_,120), got {x.shape}")

    # first 40 dims are log-mel
    log_mel = x[:, :40].T   # -> (40,T)

    # dB -> power
    mel = librosa.db_to_power(log_mel)

    # reconstruct waveform
    audio = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=SR,
        n_iter=64
    )

    sf.write(out_path, audio, SR)


# =========================
# MAIN
# =========================
def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(".npy")
    ]

    if RANDOM_PICK:
        chosen = random.sample(
            files,
            min(NUM_SAMPLES, len(files))
        )
    else:
        chosen = files[:NUM_SAMPLES]

    print(f"\n🔥 reconstructing {len(chosen)} samples...\n")

    for i, f in enumerate(chosen):

        in_path = os.path.join(INPUT_DIR, f)

        out_name = f.replace(".npy", ".wav")
        out_path = os.path.join(OUTPUT_DIR, out_name)

        try:
            reconstruct_feature(in_path, out_path)
            print(f"[{i+1}] ✔ {f} -> {out_name}")

        except Exception as e:
            print(f"[{i+1}] ✖ {f}: {e}")

    print("\nDone.")
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()