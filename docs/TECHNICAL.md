# Current social-cue prototype

## Setup

Use the project's single `.venv`. Dependencies are in `requirements/app.txt`.
On a new machine, provision only the active models:

```sh
python scripts/provision_models.py qwen
python scripts/provision_models.py emotion
python scripts/provision_models.py emotion2vec
```

Qwen may use an existing local Hugging Face cache when `models/qwen3-asr` is absent.
All inference uses local files. The emotion2vec checkpoint revision is pinned in
the provisioning script. Model files and the environment are ignored by Git.

## Active code

`audio.py` reads/downmixes audio and resamples to 24 kHz for the emotion lab.
`transcript.py` loads Qwen3-ASR 0.6B without an aligner and the SUPERB four-label
comparison model. `live_transcription.py` uses audio-only WebRTC to receive 16 kHz
mono frames, runs local Qwen on phrase snapshots, and renders drafts and timing.
`emotion.py` handles nine-class emotion2vec+ inference, comparison, abstention
rules and test logs. Camera capture, face emotion inference and wearer cues are
not implemented.

## Live transcription

Audio is received through streamlit-webrtc 0.79.0 in SENDONLY mode with empty ICE
server lists. Run through localhost. No camera, audio echo or external STUN/TURN
relay is configured. A browser connected only to the phone does not automatically
receive the glasses microphone; the computer's selected input must be the glasses.

A background worker re-decodes a growing phrase about once per second. A 350 ms
quiet pause or ten-second cap finalizes the phrase. Provisional snapshots coalesce;
final audio is kept. A six-final-phrase backlog stops capture explicitly. Silence
handling is a level heuristic. The UI refreshes every 250 ms, exports finalized
text and segment times, and filters the exact standalone output `嗯。`.

MPS is available when supported by the host; CPU can be selected between sessions.
Preparation warms the selected ASR model before capture. Timings distinguish queue,
shared-model wait, inference and last-server-sample-to-result delay. They exclude
Bluetooth/browser capture delay and physical mouth-to-screen latency. These are
revisable short-batch drafts, not Qwen's native vLLM streaming API.

## Stage 1 emotion testing

FunASR 1.4.22 loads the local emotion2vec+ large checkpoint with remote code disabled
and updates disabled. The model receives whole 16 kHz utterances, not transcripts.
Its nine scores include surprise/fear, which the old four-label model lacks.
The old model remains as an actual comparison input to the decision gates.

Default gates: top score >=0.70, top-two margin >=0.15, one-speaker confirmation,
adequate level and no substantial clipping. Angry additionally requires comparison
model agreement at score >=0.60. Confident shared-category disagreement, Other and
Unknown abstain. These are experimental heuristics, not calibrated confidence or
validated emotion accuracy. Intended test delivery is an unverified human label.
Logs stay in session memory until downloaded. No spoken cue is generated.

Exact checkpoints, hashes, hardware observations, controlled-test instructions and
all historical changes are in `GLASSES_MIC_SETUP.md`. Keep updating that file.

## Archived work

The pre-cleanup prototype was saved and pushed as commit `47f5ad9` on
`codex/social-cues-stage1`. Earlier captions, vocoder/codec work, voice isolation,
forced alignment and sample recordings are recoverable from Git history. Their
code/models are not required by the current app and have been removed from this
working tree. Ignored model weights were not uploaded; historical provisioning
scripts identify the downloadable sources.
