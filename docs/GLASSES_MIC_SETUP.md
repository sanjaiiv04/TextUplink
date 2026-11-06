# Ray-Ban glasses → Mac → Brave → Streamlit microphone setup

Verified with this project on November 3, 2026. The microphone worked after both
macOS and Brave were explicitly set to the glasses. The tested Bluetooth device
name was **RB Meta 0180**; another pair will have a different name.

## 1. Complete initial phone setup

- Charge the glasses and case.
- Pair the glasses with the **Meta AI app** on your phone and finish firmware updates.
- The Meta Horizon / Quest developer app used for VR is separate.
- Pairing with the phone alone does not make the glasses available to a Mac browser.

## 2. Pair with the Mac

- Temporarily turn off Bluetooth in the iPhone's **Settings** to prevent it from
  reconnecting during the Mac test. Keep the existing pairing; do not factory reset.
- Power on the glasses, take them out of the case, and fold both arms.
- Hold the **capture button on top of the frame** for about **12 seconds**, until
  the light **inside the glasses blinks blue**.
- Open **Mac System Settings → Bluetooth**. Select the glasses under nearby devices
  and click **Connect**. Confirm their status becomes connected.

The rear case button did not produce the expected blue pairing light during this
setup; it showed white. The capture-button method above successfully enabled pairing.
Follow your model's instructions if the controls differ. Avoid holding buttons
indefinitely or performing a factory reset as a troubleshooting shortcut.

Official pairing reference:
https://www.ray-ban.com/usa/c/frequently-asked-questions-ray-ban-meta-smart-glasses

## 3. Select and verify the Mac microphone

1. Open **System Settings → Sound → Input**.
2. Select the Ray-Ban glasses, such as **RB Meta 0180**.
3. Speak near the glasses. Confirm that the **input-level meter moves**.

If the glasses are missing, check the Mac Bluetooth connection first. If they are
listed but the meter does not move, troubleshoot the hardware connection before
recording in Streamlit.

## 4. Set playback separately

Open **System Settings → Sound → Output** and select either:

- **MacBook Speakers**, to listen through the computer; or
- The **Ray-Ban glasses**, to listen through their speakers.

Raise the selected output's volume and check mute. Input and output are separate
settings: keep input on the glasses when changing playback output.

## 5. Explicitly select the microphone in Brave

1. Type **brave://settings/content/microphone** into Brave's **address bar** and press Enter.
   This is a browser settings address, not a terminal command or web search.
2. In the **Microphone** dropdown, explicitly select the Ray-Ban device.
   On the tested setup it appeared as **RB Meta 0180 (Bluetooth) (System default)**.
3. Confirm that **Sites can ask to use your microphone** is enabled.
4. Allow microphone access for the Streamlit address you are using, normally
   **http://localhost:8501**. If the site is blocked, change its microphone permission
   through the site's settings. Another address or port may need separate permission.
5. Return to Streamlit and **reload the page** before making a fresh recording.

**The issue we encountered:** macOS input was already set correctly and its input
meter moved, but Brave still had **MacBook Air Microphone (Built-in)** selected.
Changing Brave to the glasses, reloading Streamlit, and making a new recording
resolved the test. System input selection alone was not sufficient.

## 6. Record, play and save in Streamlit

From the project folder:

```sh
source .venv/bin/activate
streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

1. Open the local URL in Brave, then choose the **Microphone test** tab.
2. Start recording and allow microphone access if prompted.
3. Speak for roughly **10 seconds**, with the Mac a few feet away and your voice
   near the glasses. Disconnect other headsets to avoid confusing input routes.
4. Press **Stop recording**.
5. Check **Recorded peak** and **Average recording level**, then press Play.
6. Click **Save microphone recording as WAV** to download `microphone_test.wav`.
7. Optionally upload that WAV in **Expressive captions** for isolation/transcription.

This test records the browser-selected microphone. It does not automatically verify
its hardware identity. The WAV is the original browser recording, with no isolation
or AI processing applied in this tab.

## If the recording is silent

Check in this order:

1. **Bluetooth:** does the Mac show the glasses connected?
2. **Mac input:** are the glasses selected, and does the level meter move when speaking?
3. **Brave input:** is the glasses device explicitly selected, rather than the built-in mic?
4. **Permissions:** can Brave use the microphone in macOS Privacy & Security → Microphone,
   and can the Streamlit site use it in Brave?
5. **Fresh capture:** reload Streamlit, then record again. An older silent recording
   will not change when input settings change.
6. **Recorded levels:** “digital silence” means every recorded sample is zero.
   A very quiet signal can also be inaudible; nonzero audio does not prove speech was captured.
7. **Playback:** check output device, mute and volume. Try MacBook Speakers to distinguish
   playback routing from capture. Hearing a system sound through the glasses confirms
   their speakers work, but does not establish that their microphone is being recorded.

When testing another pair, repeat the Mac and browser selections using its new
Bluetooth name. Recheck after reconnecting devices or switching browsers/profiles.
For phone-only use, this Mac setup is not an audio bridge from the iPhone; a separate
phone capture path would be needed.

## Change log · November 3, 2026 · Live transcription latency

Keep adding future setup changes, debugging observations and verified outcomes to
this file so the workflow can be reproduced. Current working branch:
`codex/glasses-live-transcription`.

### Baseline and measured cause

The first live implementation finalized separate segments at up to 5 seconds and
refreshed the display once per second. Qwen used CPU float32. Its six-segment queue
could discard older audio under overload. These were explicit sources of latency
and potentially missing words. The installed Qwen Transformers backend does not
support its native streaming API; that API requires vLLM. No vLLM or cloud model
has been introduced.

On the included clean sample, measured November 3 on this Mac with local
Qwen3-ASR-0.6B float32, automatic language detection, timestamps disabled:

- CPU: first 2-second input 2.057 s; warmed repeat 1.102 s; 3 seconds 1.601 s;
  5 seconds 5.019 s; full 5.98-second clip 2.326 s.
- Apple GPU/MPS: first 2-second input 7.001 s; warmed repeat 0.883 s; 3 seconds
  1.363 s; 5 seconds 2.090 s; full clip 1.592 s.
- CPU and GPU returned identical text at each tested length. The full text was:
  “A tanker is a ship designed to carry large volumes of oil or other liquid cargo.”
- These are individual timings, not benchmark averages or a guaranteed speedup.
  Noisy speech, accents, thermal conditions and other workloads can change results.
- MPS appeared unavailable inside the tool sandbox, but was available when checked
  outside it. Check availability in the actual Streamlit runtime before diagnosing
  a missing GPU option.

### Implemented changes (verification recorded below)

- A separate live model loader omits the forced aligner. Recorded caption alignment
  retains its existing CPU path.
- **Qwen processing device** selects Apple GPU/MPS when available, with CPU available
  explicitly. Setup errors are visible; the app does not silently switch devices.
- **Prepare local Qwen3 microphone** loads the selected model and performs a dummy
  two-second silent inference before capture. The dummy result is discarded. Model
  load and warm-up durations are displayed and cached per device.
- A provisional draft is requested every second from the complete current phrase.
  The draft can change as context arrives. About 350 ms of quiet, or a 10-second
  continuous-speech limit, finalizes a phrase. Quiet detection remains heuristic.
- An older queued draft is replaced by the latest snapshot; its audio is still in
  the complete final phrase. Final phrases are processed in order. If six finals
  wait, capture stops and finishes accepted final audio instead of discarding it.
- Display refresh is 250 ms. The UI shows inference time, queue plus shared-model
  lock wait, and server-received-audio to result time. This last measure excludes
  Bluetooth/browser capture delay and up to 250 ms display refresh.
- Plain text and JSON downloads include finalized phrases; STOP finalizes the tail.
  A standalone `嗯。` remains filtered. Other speech is not paraphrased or summarized.

### Reproduce the next hardware test

1. Keep the Mac and Brave microphone set to the glasses as described above.
2. Restart/reload Streamlit after applying code changes. Finish any old session.
3. Open **Live transcription**, choose **Apple GPU (MPS)** if offered, and click
   **Prepare local Qwen3 microphone**. Wait for the displayed warm-up to finish.
4. Choose the glasses microphone in the WebRTC component and click START.
5. Speak a sentence continuously, then pause. Watch the draft appear and revise,
   followed by finalized text. Read the three timing measurements after an update.
6. STOP, wait for the last phrase, and export the finalized text and timed segments.
7. Repeat with CPU between sessions if GPU fails or seems slower. Use the same
   spoken sentence and compare actual words as well as delay.

The automatic timing does not measure physical mouth-to-screen latency. For that,
record a video with audible speech and the visible display and compare the word's
spoken time with its first appearance. This hardware measurement is still pending.

### Verification outcomes for the latency changes

- All **16 automated tests passed**, including revisable drafts, full final-phrase
  context, provisional coalescing, final-queue overload without silent dropping,
  STOP/tail flush, silence, and the standalone filler filter.
- Streamlit's prepare flow rendered the device selector, warm-up metadata and
  revised transcript panel successfully with mocked inference.
- Actual Apple GPU replay of `samples/audio-without-noise.wav`, fed in 20 ms frames
  at its real duration: ASR-only load 13.770 s; silent warm-up 1.751 s. Setup times
  vary with cache/other processes and are not part of live capture timing.
- First observed nonempty draft: **3.601 s from replay start**. A later draft
  appeared at 5.124 s. A one-second request cadence does not guarantee the first
  useful text arrives in one second.
- The 2-second draft inference took 1.020 s, with 0.093 s queue wait and 1.117 s
  last-server-audio-to-result delay. The 3-second draft took 1.506 s, with 0.115 s
  queue wait and 1.621 s last-server-audio-to-result delay.
- Final recognized words matched the full-recording Qwen reference after ignoring
  case and punctuation. The pause split produced “large volumes. Of oil…” whereas
  the reference kept one sentence. Punctuation/case equality was not established.
- This replay measures the Python capture/ASR pipeline with known audio, not the
  physical glasses/Bluetooth/browser path. New live hardware timing remains pending.
- Old in-memory live sessions are stopped when the engine version changes; the UI
  offers a text download from the previous engine. Prepare a new session afterward.

### Debug the new timing indicators

- Large **Last inference**: try the other processing device between sessions, ensure
  warm-up finished, and close other heavy workloads. Longer phrase snapshots cost
  more; this implementation re-decodes them and does not reuse a native streaming KV cache.
- Large **Queue + model wait**: the model is behind or another caption transcription
  holds the shared lock. Avoid running recorded-file ASR simultaneously.
- Small displayed timings but slow perceived text: the shown result may cover an older
  audio snapshot, useful text may need more context, or Bluetooth/browser transport
  may add delay. The metric is not a claim that every spoken word appears that quickly.
- Draft words changing is expected. Check the finalized text after a pause/STOP.
  Downloads intentionally omit unfinished drafts.
- No Apple GPU option: verify MPS availability in the running host environment;
  a sandboxed Python check can report false even when the normal Mac app can use it.
- GPU initialization error: select CPU and prepare again. The error remains visible;
  there is no automatic hidden fallback.

## Recording a demo with the glasses' built-in camera

Requested November 3, 2026: record from the glasses themselves, rather than using
Mac screen recording.

- Wear the powered-on glasses and look at the laptop's Live transcription tab.
- Press and hold the capture button on top of the right frame to start a video;
  release after recording begins. A short press normally takes a photo.
- Press the capture button again to stop. Keep the first test to about 10 seconds.
- After the test, reconnect the glasses to the iPhone if necessary, open Meta AI,
  and import the captured video through its media/gallery controls.
- Built-in camera recording while simultaneously sending the Bluetooth microphone
  to Brave has NOT been verified on this pair. Check that Streamlit's audio counter
  and microphone level continue changing and that the imported video includes sound.
  Do not assume the two capture paths can share the microphone. If live transcription
  stops or video recording is refused, record the demonstration externally instead.

Official camera control reference:
https://www.ray-ban.com/usa/c/frequently-asked-questions-ray-ban-meta-smart-glasses

## Prototype direction · November 4, 2026 · Social-cue audio feedback

Professor's intended experience: the glasses hear another person's speech and give
private audio cues to the wearer about possible emotional delivery. Streamlit is
an observability/demo screen; the final interaction must work without reading it.
The following is a proposed next milestone, NOT implemented or hardware-verified.

1. Verify the other person's voice first: wearer stays silent while another person
   speaks toward them from conversational distance. Save samples with the glasses
   input selected. Existing wearer-speech transcription does not establish that a
   conversation partner is captured well. Compare quiet and noisy surroundings.
2. Branch the existing live audio into Qwen transcription and speech-emotion analysis
   in parallel. Emotion analysis should not wait for ASR or use text as its sole input.
   Start with short phrase audio (roughly 2–4 seconds), benchmark actual delay and
   preserve captured audio for comparison with any enhancement.
3. Use the existing local four-class SUPERB classifier only as an integration baseline.
   It is an utterance classifier trained/evaluated on IEMOCAP, not validated on these
   glasses' Bluetooth audio or everyday social conversations. Review several speakers
   and identical sentences spoken with varied delivery before selecting a final model.
   Model scores are uncalibrated; low-confidence, silent, noisy or overlapping input
   should permit abstention. Do not describe predictions as known internal feelings.
4. Show estimated delivery, score/source, abstention reason, cue decision and timings
   in Streamlit. Check whether false confident labels are more disruptive than missing
   labels. Do not treat successful word transcription as evidence of emotion accuracy.
5. Test output independently with a button that plays one short, locally generated or
   prerecorded spoken cue through the glasses. This should verify routing/volume and
   whether cue audio is recaptured by the microphone before automating cue generation.
6. Gate automatic cues: require sustained evidence; suppress repeats and neutral cues;
   add a cooldown and a mute control. These initial parameters are tunable hypotheses.
   Pause/mark analysis during cue playback and measure residual echo instead of treating
   the cue as a new conversation partner. Check speech/cue interference with the wearer.
7. First closed-loop demo: one other speaker, wearer silent, clear turn-taking, quiet
   room, short cues for supported labels only. Later work must distinguish wearer vs
   partner, handle multiple/overlapping speakers, and test with intended users for
   usefulness. No native iPhone app is needed for this first Mac/Streamlit milestone.

Initial success criterion: another person's speech produces inspectable predictions
in Streamlit and an occasional audible cue in the glasses, with measurable delay,
visible abstentions, and no repeated feedback from the app's own cue. Accuracy and
social usefulness require separate evaluation with participant feedback.

Reference checked November 4:
https://huggingface.co/superb/wav2vec2-base-superb-er
The official model card specifies 16 kHz input and utterance-level emotion prediction
using IEMOCAP, reduced to four classes. Current app code maps neutral/happy/angry/sad.

## Stage 1 implementation · November 6, 2026 · Emotion comparison lab

User request: improve emotion robustness, including distinguishing an angry versus
surprised/excited “oh my god,” in stages. High pitch is NOT an anger rule. Facial
appearance is not used to assign emotions. Camera-based active-speaker assistance
would be a separate, later feature; no camera code is enabled here.

### What is installed

- Added **Emotion lab · Stage 1** to Streamlit. Existing live Qwen transcription
  remains separate. No automatic emotion cues or live emotion analysis are enabled.
- New candidate: `emotion2vec/emotion2vec_plus_large`, revision
  `6c303ba987b86b93193de93e34bb2b077a6bedc4`, stored in
  `models/emotion2vec_plus_large`. The nine labels are angry, disgusted, fearful,
  happy, neutral, other, sad, surprised and unknown. These are model outputs, not
  verified internal feelings or robust social-cue understanding.
- Weight SHA-256 observed on installation:
  `be501a01f26fcdc7663a062dff86af839afbaef7c4de32f5e42d7e1ad2784da4`.
- Runtime: **FunASR 1.4.22**, CPU, added to the existing single `.venv`. Its install
  added dependencies without replacing Qwen's existing packages. `uv pip check`
  reported all 153 installed packages compatible.
- The official model describes about 300M parameters and large-scale pseudo-labelled
  training. Its HF license tag is `model-license`; do not assume the source code's MIT
  license also licenses the model weights. Retain its model card with the weights.
- Setup command: `python scripts/provision_models.py emotion2vec`. A pinned snapshot
  downloads only the checkpoint, configuration, token list and model card. Existing
  artifacts can be checked with `--local-only`.
- Loading uses the local path, `disable_update=True`, and `trust_remote_code=False`.
  Official architecture code comes from the installed FunASR package. Inference
  was tested with socket connections blocked; no cloud inference is required.

### Why this stage is separate

The previous SUPERB classifier has only four labels, so cannot express surprise
or fear. The new candidate can. That expansion is NOT proof of better accuracy on
our glasses. An unlabelled clean reading sample received Sad ~0.800 vs Neutral ~0.195
from the new model, demonstrating that a large score alone cannot establish accuracy.
It must not be treated as a verified emotional label for that recording.

The lab optionally compares the previous local SUPERB model (default on). It cannot
veto newly supported surprise/fear classes just because it lacks those classes.
For shared categories, a confident disagreement causes abstention. In this prototype,
Angry additionally requires the old model also to choose Angry with score >=0.6.
This conservative consistency check is a heuristic, not an ensemble validated for
accuracy. Two models can still make the same mistake.

### Exact decision gates

- Clip must be 1–15 seconds, finite mono after downmix/resampling.
- Very quiet input (RMS <0.001) is not classified in the lab.
- User confirms one clear speaker. This is a human check; the app does not detect
  diarization, overlap, speaker identity or who is wearing the glasses.
- More than 1% of original input samples at absolute amplitude >=0.99 withholds the
  decision. This is a heuristic for clipping, not a complete audio-quality detector.
- Default top score >=0.70 and top-two margin >=0.15; both sliders are tunable.
- Other/Unknown abstain. Anger requires the comparison described above.
- Scores are not calibrated probabilities. No promise is made that an accepted
  result is correct. Changing thresholds recomputes the gate, not model inference.

### How to run the controlled test

1. Keep Mac and Brave input on the glasses. Stop live transcription during lab tests.
2. Open **Emotion lab · Stage 1**. Record one complete 1–15-second phrase, or upload one.
3. Enter the speaker's **intended delivery**. This is a test label, not verified ground
   truth; it is NOT given to either model as input.
4. Confirm one clear speaker and click **Analyze vocal delivery**. Inspect all nine
   alternatives, top-two gap, old model comparison, abstention reasons and timings.
5. Click **Add result to test log**. Download the report and audio, or download the
   cumulative session log after several recordings. Logs live only in this browser
   session; download before reloading. Re-adding identical audio updates its entry.
6. Record the SAME words (“oh my god”) as angry, surprised, happy and neutral, several
   times. Then repeat with another speaker, at conversational distance, wearer silent.
7. Compare intended labels with predictions and count accepted false anger estimates.
   Repeat in mild noise. Do not call results from one speaker/phrase general accuracy.

Reports include audio hash, actual model checkpoint hash, source revision, runtime,
all scores, comparison model result, thresholds, audio-quality measurements, timing,
intended test label and withheld/accepted decision. Raw recordings are only saved
when the user downloads them. No audio cue is automatically played.

### Verified so far / known debugging details

- Actual inference on the clean ~6-second sample ran offline: ~11.55 s initial load,
  ~0.43 s classification on this run. This is not a latency guarantee or an emotion
  accuracy result.
- The checkpoint's last label is `<unk>`, which is explicitly mapped to Unknown.
  The initial strict parser rejected it; that mapping was corrected and retested.
- FunASR warns about missing training decoder tensors. Its utterance-classification
  inference uses `features_only=True`; those decoder branches are not used in the
  tested classification path. Do not suppress or ignore unrelated load warnings.
- No glasses emotion dataset or wearer-feedback trial has been completed yet.

### Next stages after the controlled recordings

Stage 2: use the same emotion-analysis logic on finalized live phrase audio in an
independent worker, alongside Qwen. Measure latency and errors without sending cues.
Stage 3: add a manual spoken-cue output test, then stability/cooldown/mute and feedback
suppression before automatic cues. Speaker separation and multiple people require
separate validation. Camera capture, if added for active-speaker localization, needs
separate glasses hardware/access testing and does not provide emotion labels.

Sources checked November 6:
- https://huggingface.co/emotion2vec/emotion2vec_plus_large
- https://github.com/ddlBoJack/emotion2vec
- https://github.com/modelscope/FunASR

Stage 1 verification completed November 6:
- **22 automated tests passed**, including surprise versus anger, uncertainty/margin
  gates, corroboration, unconfirmed speakers, invalid scores and neutral suppression.
- The new tab's recording, nine-score display, intended label, test-log updates and
  abstention behavior passed Streamlit AppTest checks with controlled model outputs.
- A second end-to-end AppTest used the REAL downloaded emotion2vec+ and SUPERB models.
  The new model chose Sad (0.7999), the old model chose Happy (0.6127), and the lab
  displayed **Uncertain** because of shared-category disagreement. Report/log export
  succeeded. This unlabelled sample is not evidence for either emotion being correct.
- Full offline inference was separately verified with socket connections rejected.
- The stronger candidate, comparison model and heuristic gates are ready for controlled
  glasses recordings. Emotion accuracy, actual false-anger rate and usefulness to a
  wearer remain unmeasured. Do not enable automatic wearer cues based only on these tests.
- Qwen regression after installing FunASR: local CPU ASR on the first two seconds
  of the clean sample still returned “A tanker is a ship designed.” No Qwen package
  versions or existing live transcription logic were replaced for the emotion lab.

### User test update · November 6, 2026

The user reports Stage 1 worked for a simple test using their own voice. This is
an initial functional observation, not a measured accuracy result. No labelled
recordings, confusion counts, cue trials or partner-speech results were supplied.

Next test: wear the glasses silently while a partner speaks the same phrase in
neutral, happy, surprised and angry intended deliveries at conversational distance.
Run three recordings per delivery (12 clips), record abstentions/mismatches and
accepted anger predictions on non-angry intended deliveries, and download the lab
log. Check audibility first. If partner capture works, Stage 2 can add live emotion
estimates beside Qwen, still without automatic wearer cues; repeat the controlled
cases live and measure delay/stability. Do not treat the small test as general
validation or make a numerical accuracy claim from the user's informal observation.

### User test update · Video playback · November 6, 2026

While alone, the user played a YouTube video of someone saying “oh my god, what
is that” with delivery they described as surprised. They report the emotion lab
identified it as Surprised. No URL, recording, model scores, gate status or exported
report was supplied, so this is a user-reported qualitative result; it does not
establish the accepted-gate outcome or accuracy.

This adds an external-voice playback case beyond the user's own voice. It is not
yet a live partner-speech test: source editing/compression and playback speakers
can affect the captured signal. Useful next solo tests are contrasting surprised,
angry, happy and neutral clips, preferably identical wording and similar playback
level/distance. Include high-pitched non-angry delivery to check false anger.
Log failures and uncertain cases as well as matches, and export the session log.

## Cleanup and branch snapshot · November 6, 2026

User requested saving the current working prototype to a new feature branch before
removing earlier experiments. Saved and pushed **47f5ad9** on
**codex/social-cues-stage1** BEFORE deletion. The cleanup follows as a separate
commit on that branch, so the pre-cleanup code can be recovered from history.

Current app scope: Microphone test, Live transcription, Emotion lab · Stage 1.
Removed the old expressive-caption/upload workflow, pitch-shaped caption player,
voice-isolation implementations/vendor code, alignment code and the old samples.
Deleted local DeepFilterNet, MossFormer2, RE-USE and Qwen forced-aligner weights.
Kept the SUPERB baseline because the emotion lab uses it for corroboration, and
kept emotion2vec+ and local cached Qwen. Ignored model weights were NOT in the Git
snapshot; the historical setup script gives their sources if required again.

Historical sections above document earlier versions; commands mentioning removed
models, samples or tabs do not describe the cleaned app. For current setup, use
README.md and TECHNICAL.md. For old sample-based benchmark reproduction, recover
the sample from the saved commit into a temporary file, e.g.:

```sh
git show 47f5ad9:samples/audio-without-noise.wav > /tmp/textuplink-old-clean-sample.wav
```

Only active live/emotion verification remains in the local ignored tests folder.
Old tracked tests are removed along with the code they tested. The single `.venv`
is preserved. No new stage, automatic wearer cue or camera feature was added.

Cleanup verification:
- 11 active live/emotion tests passed after removing tests for retired features.
- AppTest confirmed exactly the three retained tabs, microphone WAV playback, and
  successful inference/comparison using both actual local emotion models.
- The simplified ASR-only Qwen loader transcribed the recovered two-second sample
  as “A tanker is a ship designed.”
- Removed enhancement-only packages `mlx`, `mlx-metal`, `noisereduce`; `uv pip check`
  verified the 150 remaining installed packages are compatible.
- The cleaned model folder contains emotion2vec+ and its SUPERB comparison model.
  Qwen uses the existing local Hugging Face cache; its provisioning command is kept
  so a new machine can store it in `models/qwen3-asr` instead.
