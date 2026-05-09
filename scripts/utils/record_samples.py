import sounddevice as sd
import scipy.io.wavfile as wav
import os

SAMPLE_RATE = 48000
DURATION = 2  # seconds

def record_sample(filename):
    print(f"Recording: {filename}")
    audio = sd.rec(
        int(SAMPLE_RATE * DURATION),
        samplerate=SAMPLE_RATE,
        channels=2,
        device=0   # 👈 FORCE your mic
                    )
    sd.wait()
    wav.write(filename, SAMPLE_RATE, audio)
    print("Saved.\n")

def main():
    os.makedirs("data/positive", exist_ok=True)
    os.makedirs("data/negative", exist_ok=True)
    os.makedirs("data/test_samples", exist_ok=True)

    count = int(input("How many samples? "))
    label = input("Type 'p' for positive (Kaizen) or 'n' for negative or 't' for test: ")

    folder = "data/positive" if label == 'p' else "data/negative" if label == 'n' else "data/test_samples"


    for i in range(count):
        input(f"Press Enter to record sample {i+1}...")
        filename = os.path.join(folder, f"{label}_{i}.wav")
        record_sample(filename)

if __name__ == "__main__":
    main()