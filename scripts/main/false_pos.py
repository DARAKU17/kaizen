import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import WakeCRNN
from features import extract_features


# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../../models/kaizen_crnn.pth"

DATASETS = {
    "silence": "test_audio/silence",
    "noise": "test_audio/noise",
    "sine": "test_audio/sine",
    "wakeword": "test_audio/wakeword",
    "false":  "test_audio/hard_negatives"
}

THRESHOLD = 0.9


# =========================
# LOAD MODEL
# =========================
model = WakeCRNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# =========================
# SCORE FUNCTION
# =========================
def score_audio(path):
    import librosa
    y, _ = librosa.load(path, sr=16000)
    feat = extract_features(y)

    if feat.ndim == 2:
        feat = feat[np.newaxis, :, :]

    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        score = torch.sigmoid(out).item()

    return score


# =========================
# COLLECT SCORES
# =========================
results = {}

for label, folder in DATASETS.items():
    scores = []

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            scores.append(score_audio(path))

    results[label] = scores


# =========================
# FALSE POSITIVE RATE
# =========================
false_positive_rates = {
    k: np.mean(np.array(v) > THRESHOLD)
    for k, v in results.items()
}


# =========================
# HEATMAP PLOT
# =========================
labels = list(false_positive_rates.keys())
values = list(false_positive_rates.values())

plt.figure(figsize=(8, 4))
plt.bar(labels, values)

plt.title("False Positive Heatmap (Kaizen CRNN)")
plt.ylabel("Trigger Rate")
plt.ylim(0, 1)

for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

plt.show()