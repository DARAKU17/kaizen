import torch
import numpy as np
import sounddevice as sd
import librosa
import queue
import threading

from model import KaizenWakeWord

# 🎯 CONFIG
SR = 16000
THRESHOLD = 0.85

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 Load model
model = KaizenWakeWord(
    feature_size=120,
    hidden_size=128
).to(device)

model.load_state_dict(torch.load("models/kaizen_lstm.pth", map_location=device))
model.eval()

print("🔥 Kaizen LSTM Wake Engine Running...")
print("Device:", device)

audio_queue = queue.Queue()
buffer = np.zeros((0, 1))


# 🎙️ mic callback
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


# 🔊 FEATURE ENGINE (FIXED → produces 120-dim feature vector)
def extract_features_from_array(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=40,
        fmax=8000
    )

    log_mel = librosa.power_to_db(mel)

    delta = librosa.feature.delta(log_mel)
    delta2 = librosa.feature.delta(log_mel, order=2)

    # stack → (3, mel, time)
    features = np.stack([log_mel, delta, delta2], axis=0)

    # normalize per channel
    for i in range(3):
        features[i] = (features[i] - np.mean(features[i])) / (np.std(features[i]) + 1e-9)

    # 🧠 CRITICAL FIX:
    # convert (3, mel, time) → (120, time)
    # 40 mel * 3 channels = 120 feature dims
    features = features.reshape(120, -1)

    return features.astype(np.float32)


# 🧠 preprocess audio window
def preprocess(audio):
    audio = audio.flatten()

    if len(audio) < SR:
        audio = np.pad(audio, (0, SR - len(audio)))
    else:
        audio = audio[:SR]

    return extract_features_from_array(audio)


# 🧠 inference loop
def infer_loop():
    global buffer

    while True:
        chunk = audio_queue.get()
        buffer = np.concatenate([buffer, chunk])

        if len(buffer) > SR:
            buffer = buffer[-SR:]

            features = preprocess(buffer)

            # (seq_len, batch, feature)
            x = torch.tensor(features, dtype=torch.float32)
            x = x.permute(1, 0).unsqueeze(1).to(device)

            with torch.no_grad():
                out = model(x)
                score = torch.sigmoid(out).item()

            print(f"\r🎧 Wake Score: {score:.4f}", end="")

            if score > THRESHOLD:
                print("\n🚨 WAKE WORD DETECTED 🚨")


# 🎙️ start microphone stream
stream = sd.InputStream(
    samplerate=SR,
    channels=1,
    callback=audio_callback
)

with stream:
    thread = threading.Thread(target=infer_loop)
    thread.start()
    thread.join()