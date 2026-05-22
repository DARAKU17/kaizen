import os
import numpy as np
import soundfile as sf
from scipy import signal

# =========================
# CONFIG
# =========================
OUT_DIR = "../../audio_data_raw/hard_negative_noise"

SR = 16000
DURATION = 2
N_PER_TYPE = 40

os.makedirs(OUT_DIR, exist_ok=True)

N = SR * DURATION


# =========================
# HELPERS
# =========================
def normalize(y):
    return y / (np.max(np.abs(y)) + 1e-9)


def save(name, y):
    sf.write(
        os.path.join(OUT_DIR, name),
        y.astype(np.float32),
        SR
    )


t = np.linspace(0, DURATION, N, endpoint=False)


# =========================
# GENERATORS
# =========================
def silence():
    return np.zeros(N)


def white():
    return normalize(np.random.randn(N))


def pink():
    white_noise = np.random.randn(N)
    pink_noise = np.cumsum(white_noise)
    return normalize(pink_noise)


def brown():
    x = np.cumsum(np.random.randn(N))
    return normalize(x)


def static():
    y = np.random.randn(N) * 0.2

    # add crackles
    for _ in range(np.random.randint(30, 100)):
        idx = np.random.randint(0, N)
        y[idx:idx+10] += np.random.uniform(0.5, 1.0)

    return normalize(y)


def sine():
    f = np.random.uniform(80, 5000)
    return normalize(np.sin(2*np.pi*f*t))


def multi_sine():
    y = np.zeros(N)

    for _ in range(np.random.randint(2, 6)):
        f = np.random.uniform(100, 5000)
        y += np.sin(2*np.pi*f*t)

    return normalize(y)


def square():
    f = np.random.uniform(80, 3000)
    return normalize(signal.square(2*np.pi*f*t))


def saw():
    f = np.random.uniform(80, 3000)
    return normalize(signal.sawtooth(2*np.pi*f*t))


def hum():
    f = np.random.choice([50, 60])
    return normalize(np.sin(2*np.pi*f*t))


GENERATORS = {
    "silence": silence,
    "white": white,
    "pink": pink,
    "brown": brown,
    "static": static,
    "sine": sine,
    "multi_sine": multi_sine,
    "square": square,
    "saw": saw,
    "hum": hum
}


# =========================
# MAIN
# =========================
print("Generating hard negatives...")

count = 0

for name, fn in GENERATORS.items():
    for i in range(N_PER_TYPE):

        try:
            y = fn()

            save(
                f"{name}_{i}.wav",
                y
            )

            count += 1

        except Exception as e:
            print("skip:", name, e)

print()
print("✔ Done")
print("Generated:", count)