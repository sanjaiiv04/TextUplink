# TextUplink · Social cue prototype

Local speech capture and transcription from Ray-Ban glasses, with staged testing
of vocal-emotion estimates. The app has three tabs:

- **Microphone test:** record, inspect audio levels, play and download a WAV.
- **Live transcription:** local Qwen3-ASR with revisable drafts and latency metrics.
- **Emotion lab · Stage 1:** compare nine-label emotion2vec+ with the four-label
  baseline, inspect alternatives/abstentions and export controlled test results.

```sh
source .venv/bin/activate
streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open http://localhost:8501 in Brave. Select the glasses in both **Mac Sound → Input**
and **Brave microphone settings**, then reload. See the full [setup and debugging
notes](docs/GLASSES_MIC_SETUP.md), which also retain the history of experiments.

For live speech, choose Apple GPU/MPS when offered, prepare/warm Qwen, select the
glasses microphone and press START. Pause or STOP to finalize drafts.

For emotion testing, stop live capture and record one clear speaker for 1–15 seconds.
Test identical words with different intended deliveries and save the log. Labels
are estimates, not facts about feelings; anger requires model corroboration.
Automatic spoken cues and speaker tracking are later stages.

## Files

- `app.py`: only app entry point.
- `textuplink/`: active audio, ASR, live capture and emotion code.
- `models/`: local Qwen and emotion weights (ignored by Git).
- `scripts/provision_models.py`: download active model weights on a new machine.
- `requirements/app.txt`: dependencies for the single `.venv`.
- `docs/`: current technical guide and cumulative reproduction notes.
- `tests/`: local verification, ignored by Git.

For model setup, offline loading and limitations, read [the technical guide](docs/TECHNICAL.md).
