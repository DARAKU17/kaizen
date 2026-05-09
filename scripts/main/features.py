import numpy as np
import librosa

SR = 16000
N_MELS = 40   # your new choice (good for lightweight model)
HOP_LENGTH = 160
WIN_LENGTH = 400
FMAX = 8000


def extract_features(y, sr=SR):
    # =========================
    # 1. Normalize audio
    # =========================
    y = y.astype(np.float32)
    y = librosa.util.normalize(y)

    # =========================
    # 2. Fix length (1 sec window)
    # =========================
    target_len = sr
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # =========================
    # 3. Log-Mel Spectrogram
    # =========================
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmax=FMAX
    )

    log_mel = librosa.power_to_db(mel)

    # =========================
    # 4. Deltas (temporal motion)
    # =========================
    delta = librosa.feature.delta(log_mel)
    delta2 = librosa.feature.delta(log_mel, order=2)

    # =========================
    # 5. Stack → (3, n_mels, time)
    # =========================
    features = np.stack([log_mel, delta, delta2], axis=0)

    # =========================
    # 6. Normalize per channel
    # =========================
    for i in range(features.shape[0]):
        features[i] = (features[i] - features[i].mean()) / (features[i].std() + 1e-9)

    # =========================
    # 7. Collapse channel → LSTM-friendly
    # =========================
    # (3, mel, time) → (time, feature)
    features = features.mean(axis=0).T

    return features.astype(np.float32)