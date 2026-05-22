import os
import time
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import webrtcvad

from model import WakeCRNN
from features import extract_features


# =========================
# CONFIG
# =========================
SR = 16000

WINDOW_SEC = 2
WINDOW_SIZE = SR * WINDOW_SEC

FRAME_MS = 30
FRAME_SIZE = int(SR * FRAME_MS / 1000)   # 480 samples

THRESHOLD = 0.8
VOTE_K = 2
VOTE_N = 3
COOLDOWN = 2.0

MODEL_PATH = "../../models/kaizen_crnn.pth"
SAVE_DIR = "../../audio_data_raw/raw/negative"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# LOAD MODEL
# =========================
model = WakeCRNN().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()


# =========================
# VAD
# =========================
vad = webrtcvad.Vad(3)   # 0-3, higher = stricter


def is_speech(frame):
    """
    frame must be int16 PCM
    """
    pcm = (frame * 32767).astype(np.int16)
    return vad.is_speech(
        pcm.tobytes(),
        SR
    )


# =========================
# AUDIO QUEUE
# =========================
audio_q = queue.Queue()


def callback(indata, frames, time_info, status):
    if status:
        print(status)

    audio_q.put(
        indata[:, 0].copy()
    )


# =========================
# INFERENCE
# =========================
def infer(audio):

    feat = extract_features(audio)

    # (120, 200) -> (1,1,120,200)
    x = torch.tensor(
    feat,
    dtype=torch.float32
    )
    
    # if feature is (40,200) -> make (1,1,40,200)
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    
    # if somehow already (1,40,200) -> make (1,1,40,200)
    elif x.ndim == 3:
        x = x.unsqueeze(1)
    
    x = x.to(DEVICE)

    with torch.no_grad():
        out = model(x)
        score = torch.sigmoid(out).item()

    return score


# =========================
# SAVE TRIGGER
# =========================
def save_trigger(audio, score):

    fname = os.path.join(
        SAVE_DIR,
        f"trigger_{int(time.time())}_{score:.3f}.wav"
    )

    sf.write(
        fname,
        audio,
        SR
    )

    print(f"💾 saved -> {fname}")


# =========================
# MAIN LOOP
# =========================
def main():

    print("\n🔥 REALTIME KAIZEN WAKEWORD ENGINE")
    print("Listening... 🎧 (Ctrl+C to stop)\n")

    ring = np.zeros(
        WINDOW_SIZE,
        dtype=np.float32
    )

    scores = []
    last_trigger = 0

    with sd.InputStream(
        samplerate=SR,
        channels=1,
        blocksize=FRAME_SIZE,
        callback=callback
    ):

        while True:

            frame = audio_q.get()

            # VAD gate
            try:
                if not is_speech(frame):
                    continue
            except:
                continue

            # rolling window
            ring = np.roll(
                ring,
                -FRAME_SIZE
            )
            ring[-FRAME_SIZE:] = frame

            score = infer(ring)

            print(f"score: {score:.4f}")

            scores.append(
                score > THRESHOLD
            )

            if len(scores) > VOTE_N:
                scores.pop(0)

            votes = sum(scores)

            now = time.time()

            # trigger logic
            if (
                votes >= VOTE_K and
                (now - last_trigger) > COOLDOWN
            ):

                print("\n🚨 KAIZEN DETECTED\n")

                save_trigger(
                    ring.copy(),
                    score
                )

                last_trigger = now


# =========================
# RUN
# =========================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye 👋")