import torch
import numpy as np
import librosa

from model import WakeCRNN
from features import extract_features

# =========================
# CONFIG
# =========================
SR = 16000
MODEL_PATH = "../../models/kaizen_crnn.pth"
THRESHOLD = 0.85

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# LOAD MODEL
# =========================
model = WakeCRNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("\n🔥 KAIZEN WAKEWORD TEST")
print("Device:", DEVICE)


# =========================
# INFERENCE FUNCTION
# =========================
def run_test(name, audio):

    audio = np.asarray(audio, dtype=np.float32)

    # feature extraction → (120, 200)
    x = extract_features(audio)

    assert x.shape == (120, 200), f"Bad feature shape: {x.shape}"

    # match training format → (1, 1, 120, 200)
    x = torch.tensor(x, dtype=torch.float32)
    x = x.unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        score = torch.sigmoid(out).item()

    print(f"{name:<20} | score: {score:.4f}")

    if score > THRESHOLD:
        print("🚨 WAKEWORD DETECTED\n")
    else:
        print("ok\n")


# =========================
# MAIN TESTS
# =========================
def main():

    # silence
    silence = np.zeros(SR * 2)

    # noise
    noise = np.random.randn(SR * 2) * 0.05

    # sine wave
    t = np.linspace(0, 2, SR * 2)
    sine = np.sin(2 * np.pi * 440 * t)

    print("\n--- SANITY TESTS ---")
    run_test("Silence", silence)
    run_test("Noise", noise)
    run_test("SineWave", sine)

    # =========================
    # WAKEWORD SAMPLE TEST
    # =========================
    wake_path = "n_38.wav"  # change this

    try:
        wake_audio, _ = librosa.load(wake_path, sr=SR, mono=True)
        run_test("Wakeword", wake_audio)
    except Exception as e:
        print("\n⚠️ Wakeword file not found or failed:", e)


if __name__ == "__main__":
    main()