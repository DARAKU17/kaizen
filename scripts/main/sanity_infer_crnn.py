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
THRESHOLD = 0.5

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

    # feature extraction -> (120, T)
    x = extract_features(audio)

    # ensure correct format (1, 40, 200) or (1,120,200)
    if x.ndim == 2:
        x = x[np.newaxis, :, :]  # add channel -> (1, F, T)

    # fix shape for CNN: (B,1,40,200)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        score = torch.sigmoid(out).item()

    print(f"{name:<15} | score: {score:.4f}")

    if score > THRESHOLD:
        print("🚨 TRIGGERED\n")
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

    # optional: real wakeword sample
    false_wakeword, _ = librosa.load("test_audio/hisense.wav", sr=SR)
    wakeword, _ = librosa.load("test_audio/Hopefully.wav", sr=SR)

    run_test("Silence", silence)
    run_test("Noise", noise)
    run_test("SineWave", sine)
    run_test("Hard Negative", false_wakeword)
    run_test("Wakeword", wakeword)


if __name__ == "__main__":
    main()