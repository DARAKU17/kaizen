import asyncio
import edge_tts
import os
import random
from pydub import AudioSegment

OUTPUT_DIR = "data/positive_synthetic"
TEMP_DIR = "temp_audio"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Voices to use (mix male/female accents)
VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural"
]

PHRASES = [
    "Kyzen",
    "Hey Kyzen"
]

RATES = ["-10%", "+15%", "+10%"]
PITCHES = ["-5Hz", "+1Hz", "+5Hz"]

async def generate_tts(text, voice, rate, pitch, filename):
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch
    )
    await communicate.save(filename)

def augment_audio(input_path, output_path):
    audio = AudioSegment.from_file(input_path)

    # Random slight speed change
    speed = random.uniform(0.9, 1.1)
    audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed)
    }).set_frame_rate(audio.frame_rate)

    # Add slight gain variation
    gain = random.uniform(-3, 3)
    audio = audio + gain

    # Export as WAV (16kHz mono)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")

async def main():
    count = 0

    for voice in VOICES:
        for phrase in PHRASES:
            for rate in RATES:
                for pitch in PITCHES:

                    temp_file = f"{TEMP_DIR}/temp_{count}.mp3"
                    final_file = f"{OUTPUT_DIR}/sample_{count}.wav"

                    print(f"Generating: {final_file}")

                    await generate_tts(phrase, voice, rate, pitch, temp_file)
                    augment_audio(temp_file, final_file)

                    count += 1

    print(f"\nDone. Generated {count} samples.")

if __name__ == "__main__":
    asyncio.run(main())