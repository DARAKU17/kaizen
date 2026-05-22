import os
import asyncio
import random
import edge_tts

# =========================
# CONFIG
# =========================
OUT_DIR = "../../audio_data_raw/hard_negative_words"
N_PER_WORD = 15

os.makedirs(OUT_DIR, exist_ok=True)

# rotate voices for more variety
VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
]

# words that may confuse "Kaizen"
WORDS = [
    "kaisen",
    "kayzen",
    "kyzen",
    "kaizon",
    "kaizan",
    "kazen",
    "kai sen",
    "kayson",
    "cason",
    "jason",
    "tyson",
    "raisin",
    "raison",
    "case in",
    "kay sin",
    "okay zen",
    "hey zen"
]

PREFIXES = [
    "",
    "hello ",
    "activate ",
    "run ",
    "please "
]

SUFFIXES = [
    "",
    " now",
    " please",
    " quickly"
]


# =========================
# GENERATOR
# =========================
async def generate():
    count = 0

    for word in WORDS:
        for i in range(N_PER_WORD):

            text = (
                random.choice(PREFIXES)
                + word
                + random.choice(SUFFIXES)
            )

            voice = random.choice(VOICES)

            filename = f"{word.replace(' ','_')}_{i}.wav"
            path = os.path.join(OUT_DIR, filename)

            try:
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=voice
                )

                await communicate.save(path)

                print(f"✔ {text} [{voice}]")
                count += 1

            except Exception as e:
                print("skip:", text, e)

    print()
    print("Done ✔")
    print("Generated:", count)


if __name__ == "__main__":
    asyncio.run(generate())