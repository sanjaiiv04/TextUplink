# Expressive caption prototype

## Setup

Use the existing `.venv` on this Apple Silicon Mac. RE-USE is already installed.
For a fresh environment, install `requirements/app.txt` and provision the models:

```sh
python scripts/provision_models.py reuse
python scripts/provision_models.py qwen
python scripts/provision_models.py aligner
python scripts/provision_models.py emotion
# Optional alternative enhancers:
python scripts/provision_models.py deepfilter
python scripts/provision_models.py mossformer
```

Qwen3-ASR can also use its existing local Hugging Face cache. Inference uses offline
loading. The single `.venv` supports all Python components; RE-USE uses its native
helper in `models/reuse/runtime` and needs no extra environment.

## Flow

Capture a short recording → downmix/resample to 24 kHz mono → optional local isolation
→ Qwen3-ASR 0.6B + Qwen3-ForcedAligner 0.6B → aligned words → measured prosody → caption player.

The captured waveform is retained separately. Users choose captured or isolated audio
for expression measurements and synchronized playback. Enhancement may remove expressive
detail, while raw noise may bias pitch and level measurements. No gain processing or automatic peak limiting is applied. Float WAV exports retain the waveform.
Changing source or isolation settings clears old captions. Changing the prosody source
recomputes measurements. Emotion estimates record their input and score threshold and
must be rerun to change them. Transcript corrections preserve original timing.

## Visual encoding

`captions.py` uses librosa pYIN (65–500 Hz, 1024-sample frames, 10 ms hops) and RMS.
Only voiced pitch frames with model voicing probability at least 0.5 are used. These are
heuristic thresholds, not guarantees of trustworthy pitch. High/low voices outside this
range and heavily degraded audio may be missed. Word statistics use Qwen's aligned spans.

Median pitch over aligned voiced frames sets the recording reference. Word pitch relative
to that reference controls vertical offset; changes of at least two semitones between the
first and last thirds of voiced word frames produce arrows. Relative median RMS controls
font size and boldness. These are recording-relative measurements, not speaker-normalized
measurements or inferred emotion. Missing pitch stays on the baseline; it does not prove
whispering. Word duration is included in exports but is not mapped to width because natural
word length would confound an unnormalized duration display.

The caption player embeds float WAV audio and highlights aligned words during playback.
All transcript text and annotation tooltips are HTML escaped. HTML export includes the
recording; JSON export includes timings, measurements and provenance but no audio.

## Emotion annotations

The optional local SUPERB wav2vec2 classifier is an English utterance-emotion baseline:
neutral, happy, angry and sad. It runs on approximately three-second contexts and shares
predictions among words whose midpoints lie in those contexts. Scores below the selected
threshold produce uncertain annotations. These are not independent word emotion labels,
calibrated probabilities, or validated assessments of a person's feelings. Colors are
optional and separate from playback highlighting. Human label overrides retain model
scores and original predictions distinctly. The model may be unreliable on whispers,
other languages, laughter, emergency speech and recordings unlike its training data.

## Isolation models

Default: `nvidia/NVIDIA-NemotronLabs-RE-USE`, revision
`31f82f2efad1095034492f236e8dac9239ba713b` (October 22, 2026, OpenMDW-1.1).
SpeechLens CLI v1.0 runs the converted weights in strict mode on MLX/Metal. Provisioning
checks the fixed 30-block, state-16 architecture and converts tensor names/layout without
training or quantization. Compatibility and successful inference do not prove numerical
parity with NVIDIA's CUDA reference or unchanged emotional delivery.

Conversion in `scripts/provision_models.py` is adapted from SpeechLens's WeightPort.swift,
Copyright 2026 Çağatay Çallı, Apache-2.0, revision
`f6c640fce1d265e8b4b7564ec81dff1f08d67200`. Python conversion and configuration checks
are local modifications. Full license: `docs/licenses/SpeechLens-Apache-2.0.txt`.
Runtime licenses are retained beside the helper. The CLI archive is pinned by SHA-256:
`39cbf274ba93b805eabfa0bb1dd651e8e1a835a101b400fa0ab46304b4b76a04`.
Conversion-only source tensors are deleted after conversion. The local manifest records
hashes for converted weights and runtime assets; the worker
verifies them before invocation. Native inference uses temporary float WAVs and macOS
`sandbox-exec` with network denied. Output must retain exact sample rate/frame count and
contain finite samples. Setup requires internet; inference does not.

Alternative DeepFilterNet3 and MossFormer2 use the selected MIT-licensed MLX-Audio 0.6.2
inference components under `textuplink/_vendor/mlx_audio`. Unused streaming, older model
architectures, file conversion, remote loading and generic generation interfaces have
been removed, retaining local array inference. Namespaced imports avoid the
full SDK's incompatible Transformers dependencies. They resample 24 → 48 → 24 kHz;
upsampling does not recover lost bandwidth. MossFormer2 uses its FP32 checkpoint with a
relative `model.safetensors` symlink. The worker seeds MLX and uses evaluation mode.
Spectral gating is a baseline. Neural failures surface without silent fallback.

## Scope

Uploaded audio files only, up to 60 seconds. Stereo is downmixed; this
prototype cannot separate named speakers or preserve spatial direction. No native streaming ASR,
speaker diarization, speaker baseline tracking, wearable integration or validated emotion
benchmark is implemented. Complete noise removal and unchanged delivery are not guaranteed.
Codec packets, compression, decoding and receiver workflows have been removed.

Sources:
- https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE
- https://github.com/faraday/SpeechLens
- https://github.com/QwenLM/Qwen3-ASR
- https://huggingface.co/superb/wav2vec2-base-superb-er
- https://github.com/Rikorose/DeepFilterNet
- https://github.com/modelscope/ClearerVoice-Studio

## Live transcription

`textuplink/live_transcription.py` uses streamlit-webrtc 0.79.0 with an audio-only
SENDONLY connection and a CallbackSinkTrack. ICE server lists are explicitly empty
on both peers, and audio is not echoed. Run through localhost with browser microphone
permission. This is a Mac/browser hardware prototype, not a phone-to-Mac bridge.

The capture callback resamples frames with PyAV to 16 kHz float mono and appends to a
thread-safe phrase buffer. Every second a provisional full-phrase snapshot becomes
available for local Qwen inference. Newer pending drafts coalesce; the final phrase
retains all its audio. Final phrases have priority, and their queue never discards
accepted audio. If six final phrases are waiting, capture closes with an explicit
error and processes the accepted tail. A quiet pause of 350 ms or a 10-second cap
finalizes a phrase. Boundary words can still be split at the cap.

The UI distinguishes revisable drafts from final text and exports only finalized
phrases. A shared model lock prevents simultaneous live/recorded Qwen inference.
The fragment refresh interval is 250 ms. Each update records queue wait, model-lock
wait, inference duration, input duration and last-server-sample-to-result delay.
This is not a physical mouth-to-screen measurement; Bluetooth/browser capture and
up to one display interval are outside it.

The separate live loader omits alignment and selects MPS when available, with CPU
explicitly selectable between sessions. A silent two-second warm-up runs before
capture and is discarded. Recorded caption alignment remains on the original CPU
path. Model loading failures are visible, without silently changing devices.
The artifact `嗯。` is suppressed only when it is the complete recognized output.

Warm-up/startup and CPU/MPS measurements, exact user setup and new debugging steps
are recorded in `docs/GLASSES_MIC_SETUP.md`. Update that guide after future changes.

## Stage 1 emotion evaluation

`textuplink/emotion.py` implements local emotion2vec+ large through FunASR 1.4.22,
whole-phrase score validation, quality/ambiguity gates, comparison with the existing
four-class model and a session-only evaluation log. It runs on 16 kHz mono CPU input
with no word alignment or pitch-to-emotion rule. The candidate is not yet integrated
into live capture or audio cue playback. Exact source revision, checkpoint hash,
thresholds, offline test results and reproduction steps are maintained in
`docs/GLASSES_MIC_SETUP.md` under the November 6 Stage 1 section.
