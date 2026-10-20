# Local Audio Triage — Phase 1 refined

A local Streamlit prototype for inspecting waveform, short-time energy, approximate pitch, spectral flatness, and candidate low-energy intervals.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Spectral-flatness changes

- Computes spectral flatness from the **magnitude** STFT using `librosa.feature.spectral_flatness(..., power=2.0)`. The earlier implementation passed a power spectrum with `power=1`, which can compress the range and make the curve look unusually close to zero.
- Shows full-scale and active-frame zoom views, robust percentile summaries, and a histogram.
- Excludes low-energy frames when calculating the relative high-flatness review cue. This is still a relative cue, not a calibrated noise detector.

## Optional local transcription

Install the optional dependency in the same virtual environment:

```bash
pip install -r requirements-transcription.txt
```

Restart Streamlit. In the app, select a Whisper model and click **Transcribe locally**. The app uses `faster-whisper` on CPU with INT8 quantization. The first run downloads the selected model; after it is cached, transcription can run offline. `base` is a reasonable first demo on a Mac; `small` may improve accuracy at the cost of speed and memory. For noisy 8 kHz radio audio, inspect the output manually—Whisper may miss words, and it does not perform speaker separation.

## Limitations

This app does not infer stress, emotion, credibility, intent, or physiological state. Acoustic features and transcript outputs can be wrong, especially with noise, overlap, clipping, or narrow-band radio audio. Treat all flags as review aids only.


## Noise reduction before transcription

The transcription panel optionally applies spectral-gating noise reduction with `noisereduce` before Whisper. It keeps the original recording unchanged, previews and allows downloading the enhanced WAV, and provides a strength slider. Start around 0.5–0.7; stronger suppression can remove quiet consonants or introduce artifacts. The pipeline does not use pitch shifting or time stretching, but no enhancement method can guarantee perfect pitch/timbre preservation in difficult noise. Compare the enhanced result with the original. The transcription path disables VAD filtering to reduce the chance that quiet speech is discarded, and offers optional English forcing.

Install both optional packages with `pip install -r requirements-transcription.txt` and restart Streamlit.


## Optional: compare multiple open-source transcription models

The app can compare Whisper with Qwen3-ASR 0.6B and NVIDIA Parakeet CTC 1.1B. Install the extra dependencies in the same virtual environment:

```bash
pip install -r requirements-comparison.txt
```

Restart Streamlit, upload the same audio, and use **Open-source transcription model comparison**. The models download their weights on first use, so internet access and disk space are needed initially. CPU inference can be slow, particularly for larger models. The app reports each model's output and runtime; it does not claim which is most accurate without a verified reference transcript.

For a defensible ablation/comparison study, hold the audio, preprocessing, and language setting constant across models. Use recordings with reference transcripts and report word error rate (WER), whether a model returned no text, and runtime. Treat model family, model size, decoding settings, and preprocessing as separate experimental factors where possible.
