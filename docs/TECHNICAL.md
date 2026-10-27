# TextUplink — local speech codec lab

Goal: compress speech captured in noisy environments into a small payload and reconstruct
voice while preserving intelligibility, speaker identity and expressive delivery as closely
as possible. Text alone is a content baseline, not a sufficient representation of delivery.
The main experiment now uses pretrained neural audio codecs, not ASR followed by TTS.

## Run

Use Python 3.12. The neural enhancers in this integration require an Apple Silicon Mac
with Metal access. Every component uses the same `.venv`. Enhancement runs in a worker process using
the active Python interpreter; it does not use another environment.

```sh
# Create the environment once (uv is available on this machine):
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements/app.txt
source .venv/bin/activate
# One-time connected preparation; no user audio is sent:
python scripts/provision_models.py encodec
python scripts/provision_models.py mimi
python scripts/provision_models.py focalcodec
python scripts/provision_models.py deepfilter
python scripts/provision_models.py mossformer
# Optional transcript and color models:
python scripts/provision_models.py qwen
python scripts/provision_models.py aligner
python scripts/provision_models.py emotion
# Runtime is local-only:
streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

The environment and weights above have already been prepared on this machine. For normal
use, activate `.venv` and run `streamlit run app.py`; do not recreate it each time.

`app.py` is the main app. Select an included recording or upload WAV/FLAC/OGG (up to
60 seconds). EnCodec is the default. Try 1.5, 3 and 6 kb/s on the same recording.
Use **Compare bitrate settings** to run the same input through every available setting,
view a rate/spectral-distance plot and download comparison CSV. Select individual settings
to listen to their reconstructions; signal metrics do not automatically choose a winner.
Mimi allows 4, 8, 16 or 32 codebooks (nominal 0.55, 1.1, 2.2 or 4.4 kb/s).
Actual file bitrate is higher because it includes end padding and packet metadata.
All codecs run CPU batch inference, not real-time streaming. The neural enhancers use Metal. Model weights are not in Git.

For air-gapped devices, provision on a connected preparation machine and copy the
`models/encodec_24khz` and/or `models/mimi` directory to the same locations on sender
and receiver. Model initialization uses local paths, `local_files_only=True`, and offline
Hugging Face flags. No hosted inference APIs are used. Check model/checkpoint licenses
for your intended use. Downloads are explicit setup actions, not part of inference.

## What changed

The sender converts the capture to 24 kHz mono, optionally adds controlled noise and/or
applies noise reduction, then encodes it to discrete acoustic tokens. No text transcript,
source waveform, speaker reference audio or encoder activations are placed in the packet.
The receiver reconstructs from the bit-packed tokens and matching pretrained weights.
Its packet upload works without the original recording. Download the sender packet,
load it in Receiver (or in the same app on another machine), and save the reconstructed WAV.

EnCodec is Meta's pretrained encoder, residual vector quantizer and waveform decoder.
Mimi is Kyutai's pretrained speech codec. We use their Transformers implementations to
avoid coupling the app to AudioCraft's broader training/generation dependencies. A vocoder
alone cannot solve this task: a bandwidth-constrained encoder/conditioning representation
is also necessary. AudioCraft's compression training and MultiBand Diffusion decoding are
future experiments, not implemented backends in this version. “EnCodec2” is not assumed
to name a particular checkpoint; the implemented baseline is `facebook/encodec_24khz`.

## Follow the five stages

1. **Captured voice:** upload your noisy file (leave added noise off). You can select
   the included noisy sample and explicitly confirm the included clean file as the reference.
2. **Reduce noise and raise voice level:** choose DeepFilterNet3 (default), MossFormer2 SE,
   or spectral gating. Neural models use a wet/dry mix (1 = enhanced output, 0 = original);
   spectral gating retains its suppression-strength/noise-type controls. The separate voice
   boost defaults to +9 dB with peak protection. Disable it for listening comparisons at
   matched levels. Use **Compare enhancement models (boost off)** to hear each output
   from identical captured audio; supply an aligned clean reference for signal metrics.
3. **Optional transcript and emotion colors:** run local Qwen3-ASR with forced alignment for words/times. Then optionally
   estimate phrase emotion or manually annotate words. This branch never drives the codec.
4. **Encode:** choose EnCodec, Mimi or FocalCodec, then run reconstruction.
5. **Decode:** listen to the reconstructed WAV. Compare it with processed speech to isolate
   compression loss, and with an aligned clean reference to evaluate the entire pipeline.

The audio path is captured waveform → optional enhancement → acoustic tokens → decoded waveform.
The transcript is a parallel view of processed speech. Word colors do not make the decoder
preserve emotion; expressive acoustic information is carried in the tokens themselves.

### Transcript colors are reviewable estimates

Qwen3-ASR 0.6B runs locally with Qwen3-ForcedAligner 0.6B for word timestamps. The optional SUPERB
wav2vec2 emotion classifier is an English IEMOCAP utterance-level baseline with only
neutral/happy/angry/sad categories. It is not validated for emergency/radio recordings.
Three-second audio contexts are classified and their labels assigned to words by midpoint;
this is not a model trained to identify independent emotion in each word. Classification
can use captured or processed speech. Scores below a configurable threshold are gray;
these scores are uncalibrated model outputs, not probabilities of a person's true emotion.

You can correct words and override emotion labels in the table. Human overrides are
identified in exports, with the original model label/score retained. Text corrections do
not realign timestamps. Hover over colored words for their timing and annotation source.
The optional emotion download is an explicit setup step; runtime loading remains local-only.
Qwen can use an existing local cache or `models/qwen3-asr` and `models/qwen3-aligner`.
For offline transfer, provision and copy both directories and `models/emotion` as needed. No audio is sent out.

The downloadable annotated transcript JSON is a separate sidecar. If transmitting both
text and acoustic tokens, count both payload sizes plus framing/security overhead. The
packet decoder still needs only acoustic tokens and matching codec weights. Neither
ASR accuracy nor consistency of classifier colors proves preserved emotional delivery.

## New neural backends and controlled comparisons

DeepFilterNet3 and MossFormer2 SE 48K use the selected MLX-Audio 0.6.2 inference
components retained under `textuplink/_vendor/mlx_audio`, with upstream MIT attribution.
Their neural math and weight loaders are preserved; package imports are namespaced and
unrelated TTS/STT loaders are omitted. File loading uses a local librosa wrapper.
These selected components need MLX but not the full SDK’s Transformers 5 dependency,
so they coexist with Qwen’s pinned Transformers 4.57.6 in one environment. These are MLX ports of the pretrained architectures, not the original
PyTorch runtimes. DeepFilterNet's original package requires a Rust extension without a
usable wheel for this environment; using MLX avoids changing Qwen or installing a compiler.
The lab first downmixes/resamples uploads to 24 kHz. The worker then resamples 24 → 48 kHz,
enhances, then resamples back and checks finite samples
and duration. Upsampling does not recover bandwidth lost in the initial conversion. It uses temporary `.npy` files with pickle disabled, cleans them after the
call, loads local weights and rejects socket connections. There is no silent fallback
if the selected enhancer fails. Model startup is included in worker time; enhancement
outputs are cached across Streamlit reruns. DeepFilterNet's upstream MLX implementation
clips its own waveform to [-1, 1]; our subsequent uniform boost has a separate -1 dBFS ceiling.

`models/deepfilternet3/v3` contains the DeepFilterNet3 weights/config. MossFormer2 uses the
FP32 checkpoint from `starkdmi/MossFormer2_SE_48K_MLX`; provisioning creates the default
configuration from the pinned MLX runtime and a relative `model.safetensors` alias to
`model_fp32.safetensors`. Preserve that relative link (or copy the target file as the alias)
when transferring weights. `--local-only` can finalize previously downloaded model files.

FocalCodec is installed from the pinned official source revision in `requirements/app.txt`.
The supported noncausal `lucadellalib/focalcodec_50hz` checkpoint encodes at 16 kHz with
one 8192-entry codebook, 13 bits/token and 50 tokens/s: nominal **0.65 kb/s**. Packet
metadata makes the actual rate larger, especially for short clips. Audio enters the lab
at 24 kHz, is converted to 16 kHz for FocalCodec, and returns as 24 kHz WAV for comparison.
Upsampling does not restore frequencies removed at the 16 kHz stage. No speaker-matching
reference waveform is transmitted. The receiver uses packet tokens and matching weights.
Provisioning pairs `model.json` with `model.safetensors` as required by the local loader.

Use **Compare codecs on this processed speech** to hear EnCodec at 3 kb/s, Mimi at eight
codebooks, and FocalCodec at 0.65 kb/s on identical processed samples. Their actual bitrates
are shown; this is not an equal-rate benchmark. Individual failures are retained as failures.
Download each packet for the Receiver tab and download the comparison report. Neither
comparison selects an emotion-preservation winner. There is no trained emotion-fidelity
objective in these codecs. Human listening against ground truth remains necessary.

Main signal metrics are sample-aligned and level-sensitive, so turning on amplification
can worsen waveform SNR despite louder playback. The enhancement comparison disables
boost for every model. Even then, low noise metrics alone do not establish preserved words,
voice identity or expression. Qwen stays the transcript backend for any selected enhancer.

## Packet format and bandwidth

`.tuplink` files contain `TUP1`, a big-endian uint32 JSON-header length, a bounded JSON
header, and a bit-packed code stream (10 bits/index for EnCodec, 11 for Mimi, 13 for FocalCodec). The header
includes codec family, model-file SHA-256 fingerprint, sample count, code tensor shape,
setting and payload checksum. Mono EnCodec 24 kHz uses no normalization scale; unsupported
normalizing/chunked checkpoints are rejected. Decoder padding is trimmed to declared length.
Model fingerprints must match between endpoints. Checkpoint format conversions can change
the fingerprint even if a model is numerically equivalent; copy identical model directories.

Actual reported bitrate is `8 * packet_bytes / audio_seconds / 1000`. It includes the
header and code padding, but not encryption, radio framing, retransmissions or model setup.
Compression ratio uses uncompressed 24 kHz mono PCM16 (384 kb/s), not the uploaded file's
format. Downloads use float WAV to expose, rather than silently clip, decoder overshoots.
The packet is an experimental format, not interoperable with `.ecdc` files.

Packets are **unencrypted**. SHA-256 checks detect accidental payload corruption and match
weights, but do not authenticate senders or protect metadata against tampering. No network
transport is implemented. A secure deployment still needs an established authenticated,
encrypted channel, peer identity, replay handling and key management. Do not treat this
file boundary as an implemented secure communications system.

## Evaluate expression and noise

Keep the same source, device and playback volume across codec settings. Start with calm,
urgent, questioning, whispered and shouted versions of the same words. Use speakers and
recordings held out of any future training set. Get blinded listener ratings of word
intelligibility, speaker similarity and expressive similarity; subjective labels alone
are not an emotion-preservation ground truth.

Controlled noise produces an aligned clean reference, with one shared gain applied to
clean/noisy copies to avoid clipping. Existing arbitrary noisy recordings have no known
clean ground truth. The two included samples are not automatically treated as a paired
dataset. You can explicitly choose the included clean sample as a reference or upload
your own, confirm that it is the same speech aligned in time and level, and compare against
it if the resampled lengths match. Length equality alone does not establish alignment.
Denoising defaults to DeepFilterNet3 at full enhanced mix. Neural enhancement can alter
breathiness, quiet speech and strain; spectral gating remains available at strength 0.5.
After denoising, a separate boost defaults to +9 dB. One constant gain is applied across
all samples to retain relative emphasis, pauses and timing. Peak protection reduces the
requested gain for the entire recording rather than clipping peaks or dynamically
compressing syllables. Silence is not boosted. The UI reports requested/applied gain,
RMS levels and whether headroom limited amplification. Residual noise is also amplified.
For the spectral baseline, choose steady mode for hiss or changing mode for varying noise.
Neural models do not use that noise-type selector. No method guarantees perfect isolation.
The processed waveform feeds both Qwen and the codec. New processing settings invalidate
old transcripts and reconstructions, and all settings are included in exports. A boost
makes playback louder but does not itself improve SNR or guarantee better ASR accuracy.
A codec ordinarily preserves noise as well as speech; it is not a denoiser.

Reports contain actual packet size/rate, preprocessing and encode/decode time, real-time
factor, waveform SNR, log-spectral distance and energy-envelope correlation. Signal
comparisons use the captured recording, the actual codec input, and the clean reference
when synthetically added noise or an explicitly confirmed aligned reference provides one. Metrics assume sample alignment; decoder
delay can affect them and no delay compensation is applied. These are not calibrated
emotion, intelligibility or speaker scores. Runtime excludes model loading and evaluation;
batch runtime is not streaming latency. Human listening remains necessary.

Neither model guarantees faithful preservation of emotional expression at a given bitrate.
First establish a rate/quality curve and find failures; then consider fine-tuning encoder,
quantizer and decoder together on expressive speech with noisy inputs and aligned clean
targets. Training from scratch requires a substantial dataset and compute. AudioCraft can
support that later stage; no unsupported training pipeline is claimed here.

## Validation

```sh
python -m unittest discover -s tests -v
```

Tests cover exact bit packing, corruption/invalid packet rejection, controlled SNR,
resampling, descriptive metrics and a real offline independent EnCodec receiver when
weights are provisioned. Neural inference is separately exercised on included samples.
An offline receiver test forbids socket connections during model loading and inference.

## Primary references

- Meta EnCodec: https://github.com/facebookresearch/encodec
- Transformers EnCodec: https://huggingface.co/docs/transformers/model_doc/encodec
- Kyutai Mimi: https://github.com/kyutai-labs/moshi
- Transformers Mimi: https://huggingface.co/docs/transformers/model_doc/mimi
- AudioCraft compression training: https://github.com/facebookresearch/audiocraft/blob/main/docs/ENCODEC.md
- AudioCraft MultiBand Diffusion: https://github.com/facebookresearch/audiocraft/blob/main/docs/MBD.md

- Local ASR and word timestamps: https://github.com/QwenLM/Qwen3-ASR
- Optional emotion baseline: https://huggingface.co/superb/wav2vec2-base-superb-er

- DeepFilterNet: https://github.com/Rikorose/DeepFilterNet
- MossFormer2 / ClearerVoice: https://github.com/modelscope/ClearerVoice-Studio
- Apple Silicon inference ports: https://github.com/Blaizzy/mlx-audio
- FocalCodec: https://github.com/lucadellalib/focalcodec

The enhancement worker sets a deterministic MLX seed and evaluation mode before inference.
Selected model components were compared with the original package before consolidation.
