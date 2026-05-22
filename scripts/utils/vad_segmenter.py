import os
import wave
import contextlib
from pydub import AudioSegment
import webrtcvad


# ----------------------------
# CONFIG
# ----------------------------
INPUT_FILE = input("Enter audio file: ")
OUTPUT_DIR = "speech_clips"

VAD_MODE = 3          # 0 = least aggressive, 3 = most aggressive
FRAME_MS = 30         # must be 10, 20, or 30
MIN_SEGMENT_MS = 300  # ignore clips shorter than this
MERGE_GAP_MS = 200    # merge segments if gap smaller than this


# ----------------------------
# Convert audio to mono 16k PCM
# ----------------------------
def preprocess_audio(path):
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)

    temp_path = "temp_16k.wav"
    audio.export(temp_path, format="wav")
    return temp_path


# ----------------------------
# Read wav bytes
# ----------------------------
def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000

        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, wf.getframerate()


# ----------------------------
# Frame generator
# ----------------------------
def frame_generator(frame_ms, audio, sr):
    n = int(sr * (frame_ms / 1000.0) * 2)

    offset = 0
    timestamp = 0.0
    duration = frame_ms / 1000.0

    while offset + n <= len(audio):
        yield audio[offset:offset+n], timestamp
        timestamp += duration
        offset += n


# ----------------------------
# Detect speech timestamps
# ----------------------------
def detect_speech(audio_bytes, sr):
    vad = webrtcvad.Vad(VAD_MODE)

    segments = []
    current_start = None
    last_speech = None

    for frame, ts in frame_generator(FRAME_MS, audio_bytes, sr):
        is_speech = vad.is_speech(frame, sr)

        if is_speech:
            if current_start is None:
                current_start = ts
            last_speech = ts + FRAME_MS / 1000.0

        else:
            if current_start is not None:
                segments.append((current_start, last_speech))
                current_start = None

    if current_start is not None:
        segments.append((current_start, last_speech))

    return merge_segments(segments)


# ----------------------------
# Merge close segments
# ----------------------------
def merge_segments(segments):
    if not segments:
        return []

    merged = [segments[0]]

    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]

        gap = (start - prev_end) * 1000

        if gap <= MERGE_GAP_MS:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # remove tiny clips
    filtered = [
        (s, e)
        for s, e in merged
        if (e - s) * 1000 >= MIN_SEGMENT_MS
    ]

    return filtered


# ----------------------------
# Export clips
# ----------------------------
def export_segments(original_file, segments):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    audio = AudioSegment.from_file(original_file)

    for i, (start, end) in enumerate(segments):
        clip = audio[start * 1000:end * 1000]

        filename = os.path.join(
            OUTPUT_DIR,
            f"clip_{i+1}_{start:.2f}_{end:.2f}.wav"
        )

        clip.export(filename, format="wav")
        print(f"Saved: {filename}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("Preprocessing...")
    wav_path = preprocess_audio(INPUT_FILE)

    print("Running VAD...")
    audio_bytes, sr = read_wave(wav_path)

    segments = detect_speech(audio_bytes, sr)

    print("\nDetected speech:")
    for s, e in segments:
        print(f"{s:.2f}s -> {e:.2f}s")

    print("\nExporting clips...")
    export_segments(INPUT_FILE, segments)

    os.remove(wav_path)
    print("\nDone.")