# Integrate Ollama (Phi-3.5) and Speech-to-Text

Integrate the Ollama LLM service running `phi3.5` into the wakeword engine. When the script is run, it will support both a manual text prompt and a microphone-based wakeword mode. Upon detecting the "Kaizen" wakeword, it will record the user's speech, transcribe it using a local Speech-to-Text (STT) model, and feed the transcribed text into Ollama.

## Proposed Changes

### Interactive Interface and LLM Integration

#### [NEW] [interactive_kaizen.py](file:///home/astro/Kaizen/scripts/main/interactive_kaizen.py)
* Create a new interactive main script `scripts/main/interactive_kaizen.py`.
* Provide a command-line interface that allows selecting between:
  1. **Manual Mode**: User types a prompt, and the script queries Ollama directly.
  2. **Voice (Wakeword) Mode**:
     * Continuously listens to the microphone.
     * Uses the pre-trained [WakeCRNN](file:///home/astro/Kaizen/scripts/main/model.py#L39) model to check for the "Kaizen" wakeword.
     * Once detected, it triggers audio recording.
     * Records audio using `sounddevice` and monitors voice activity via `webrtcvad`. After a duration of silence (e.g., 1.5 seconds) following active speech, it stops recording.
     * Transcribes the recorded audio to text using `openai-whisper`.
     * Sends the transcribed text to Ollama's local API.

### Environment & Dependencies

* Install `openai-whisper` and any missing system packages to support local offline transcription using CPU or GPU (CUDA).
* Pull the `phi3.5` model in Ollama.

---

## Verification Plan

### Automated/Manual Tests
* Verify Ollama is running and has the `phi3.5` model:
  `ollama run phi3.5 "Hello"`
* Run the interactive script:
  `/home/astro/KURA/kura_env/bin/python scripts/main/interactive_kaizen.py`
* Test **Manual Mode** by typing text.
* Test **Voice Mode** by speaking "Kaizen", waiting for the "Listening... 🎤" prompt, stating a query (e.g., "What is the capital of France?"), and verifying the transcription and Ollama response.
