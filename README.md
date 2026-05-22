# Kaizen Wake Word Engine

A local, lightweight voice activation and conversational AI system. Kaizen uses a hybrid CNN-GRU-Attention neural network trained on normalized log-mel spectrogram features to detect the wakeword **"Kaizen"** on a 16kHz audio stream.

---

## 🧠 Architecture Overview

Kaizen's detection pipeline processes audio in a sliding 2-second window:
1. **Feature Extraction**: Converts raw audio to a normalized log-mel spectrogram of shape `(1, 40, 200)` using `librosa`.
2. **Convolutional Network (CNN)**: Extracts spatial-temporal acoustic features from the spectrogram.
3. **Recurrent Network (Bidirectional GRU)**: Models temporal context and sequence information over time.
4. **Attention Mechanism**: Computes focus scores across the time frame to produce a context vector representing the entire 2-second clip.
5. **Classifier Head**: A linear classification layer producing raw activation logits.

---

## 🎯 Current Milestone: LLM & Speech-to-Text Integration

The current milestone focuses on integrating the detection engine with local LLM capabilities and offline transcription:
* **LLM Integration**: Connecting to a local **Ollama** instance running the **Phi-3.5** model.
* **Speech-to-Text (STT)**: Incorporating **Whisper** for local, offline transcription of the user's voice command immediately following a wakeword trigger.
* **Interactive Control Loop**: 
  * **Voice Mode**: Continuously listens for the "Kaizen" wakeword. When triggered, records the user's speech, transcribes it via Whisper, and sends the prompt to Phi-3.5.
  * **Manual Mode**: Direct text interface to query the model without voice activation.

---

## 📁 Repository Structure

* **`scripts/main/`**: Core training and inference engine.
  * [model.py](file:///home/astro/Kaizen/scripts/main/model.py): Defines the `WakeCRNN` and `Attention` classes.
  * [features.py](file:///home/astro/Kaizen/scripts/main/features.py): Extracts log-mel features.
  * [train.py](file:///home/astro/Kaizen/scripts/main/train.py): Training pipeline with early stopping and real-time metric plotting.
  * [realtime.py](file:///home/astro/Kaizen/scripts/main/realtime.py): Microphone listener with WebRTC Voice Activity Detection (VAD).
* **`scripts/utils/`**: Utilities for data preparation and synthesis.
  * [gen_hard_neg.py](file:///home/astro/Kaizen/scripts/utils/gen_hard_neg.py): Synthesizes phonetically similar negative words via TTS.
  * [hard_noise_neg.py](file:///home/astro/Kaizen/scripts/utils/hard_noise_neg.py): Generates static, pink, and brown noise signals.
  * [augment_positives.py](file:///home/astro/Kaizen/scripts/utils/augment_positives.py): Augments audio samples with pitch, speed, shifting, and noise.
  * [vad_segmenter.py](file:///home/astro/Kaizen/scripts/utils/vad_segmenter.py): Cuts raw continuous speech into clean voice snippets.
