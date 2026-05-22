import os
import numpy as np
import librosa
import soundfile as sf

INPUT_DIR = "../../audio_data_raw/raw/positive"
OUTPUT_DIR = "../../audio_data_raw/positive_aug"
SR = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_noise(y):
    noise = np.random.randn(len(y)) * 0.003
    return y + noise


def shift(y):
    return np.roll(y, int(0.1 * SR))


def speed_up(y):
    return librosa.effects.time_stretch(y, rate=1.1)


def slow_down(y):
    return librosa.effects.time_stretch(y, rate=0.9)


def pitch_up(y):
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=2)


AUGS = {
    "noise": add_noise,
    "shift": shift,
    "fast": speed_up,
    "slow": slow_down,
    "pitch": pitch_up,
}


for file in os.listdir(INPUT_DIR):
    if not file.endswith(".wav"):
        continue

    path = os.path.join(INPUT_DIR, file)
    y, _ = librosa.load(path, sr=SR, mono=True)

    base = file.replace(".wav", "")

    # save original
    sf.write(os.path.join(OUTPUT_DIR, f"{base}_orig.wav"), y, SR)

    for name, fn in AUGS.items():
        y_aug = fn(y)
        sf.write(
            os.path.join(OUTPUT_DIR, f"{base}_{name}.wav"),
            y_aug,
            SR
        )

print("Done")