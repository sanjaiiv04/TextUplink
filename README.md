# Local Audio Triage — Phase 1

A local-first Streamlit prototype for inspecting basic acoustic properties of an audio recording.

## Features
- Upload audio and play it back
- Recording duration, sample rate, and channel count
- Waveform and short-time RMS energy timeline
- Approximate pitch contour using librosa YIN
- Spectral flatness and zero-crossing rate
- Candidate intervals for relatively quiet and high spectral-flatness regions
- Download frame-level features as CSV

## Explicit limitations
This prototype does not transcribe speech, separate or identify speakers, or reliably detect stress, emotion, deception, urgency, or intent. Flags are exploratory review cues, not calibrated alerts.

## Run locally
Recommended: Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501). For a clean demo, use a short recording (10–60 seconds) and compare a clean voice clip with one containing pauses or background noise. Avoid sensitive recordings on hosted instances; run locally when privacy matters.

## Next phase
Add voice activity detection, timestamped transcription, diarization, and an operator review timeline that makes uncertainty visible.
