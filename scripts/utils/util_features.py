import numpy as np
import librosa

# =========================
# CONFIG
# =========================
SR = 16000

N_MELS = 40
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
FMAX = 8000

TARGET_LEN = SR * 2      # 2 seconds
TARGET_T = 200           # fixed time steps


# =========================
# MAIN FEATURE EXTRACTOR
# =========================
def extract_features(y, sr=SR):

    # -------------------------
    # 1. clean audio
    # -------------------------
    y = np.asarray(y, dtype=np.float32)
    y = librosa.util.normalize(y)

    # -------------------------
    # 2. force 2 seconds
    # -------------------------
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    else:
        y = y[:TARGET_LEN]

    # -------------------------
    # 3. log-mel spectrogram
    # -------------------------
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmax=FMAX,
        center=False
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    # -------------------------
    # 4. normalize per sample
    # -------------------------
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)

    # -------------------------
    # 5. fix time axis → TARGET_T
    # -------------------------
    T = log_mel.shape[1]

    if T < TARGET_T:
        pad = TARGET_T - T
        log_mel = np.pad(
            log_mel,
            ((0, 0), (0, pad)),
            mode="constant"
        )

    elif T > TARGET_T:
        log_mel = log_mel[:, :TARGET_T]

    # -------------------------
    # 6. add channel dim
    # -------------------------
    feat = log_mel[np.newaxis, :, :]   # (1,40,200)

    return feat.astype(np.float32)