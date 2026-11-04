# TextUplink · Expressive captions

Local voice isolation and captions that show how words were spoken.

```sh
source .venv/bin/activate
streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

## Try it

1. Upload your audio recording (WAV, FLAC or OGG; 0.1–60 seconds).
2. Use **NVIDIA NemotronLabs RE-USE** at full enhanced mix. Listen to the isolated speech.
3. Click **Create expressive captions**. Qwen3 transcribes isolated audio and aligns word times.
4. Play the audio inside the caption viewer. Larger/bolder words indicate relative loudness;
   vertical position and arrows show pitch and pitch movement. Compare captured vs isolated audio
   as the source for expression measurements.
5. Optionally estimate phrase emotion, correct the transcript, and enable emotion colors.
6. Download caption JSON or an HTML player with its audio for offline playback.

Emotion estimates describe phrase contexts, not independently measured word emotions.
Pitch and loudness are measured relative to the recording, not each speaker. Check the
transcript and expression against playback. This is a recorded-audio prototype; speaker tracking and automatic audio cues are future work. The Live transcription tab
adds continuous microphone capture with local Qwen3 updates in short batches.

## Project folders

- `app.py` — the only Streamlit entry point.
- `textuplink/` — audio loading, isolation, transcription and caption rendering.
- `models/` — local isolation, Qwen alignment and optional emotion weights.
- `samples/` — clean and noisy test recordings.
- `scripts/provision_models.py` — one-time local model setup.
- `requirements/app.txt` — dependencies for the single `.venv`.
- `tests/` — verification code.
- `docs/TECHNICAL.md` — setup and limitations.

Only models used by the app's isolation choices, transcription, word alignment and
optional emotion estimates are retained. Setup tools, tests and license files support
these features.

```sh
python -m unittest discover -s tests -v
```

## Microphone test in Streamlit

Open **Microphone test** to record a short clip, play it and download the original
browser WAV. For glasses audio, first connect them as a Bluetooth input on the Mac
and select that input in the browser. An iPhone-only pairing does not send audio
to this page. This recorder does not verify the hardware input identity.

For the complete, verified Mac and Brave pairing workflow and silent-recording
troubleshooting, see [Glasses microphone setup](docs/GLASSES_MIC_SETUP.md).

## Live Qwen3 transcription

1. Open **Live transcription** and click **Prepare local Qwen3 microphone**.
2. Choose the Ray-Ban microphone in the component's device selector and Brave settings.
3. Click **START** and speak; text appears after a pause or about five seconds,
   plus processing time. Click **STOP** to finish the last segment.
4. Download plain text or timed segment data if needed.

The local Transformers model transcribes short batches; this is not native token-by-token
streaming. Audio capture continues in a separate path during inference. The UI shows input
level, pending segments and any discarded audio if the model cannot keep up. Updates can
lag on CPU. Live audio is not enhanced or emotion-labelled. Use `localhost` on this Mac;
no external STUN/TURN relay is configured. Silence suppression is a simple level heuristic.
