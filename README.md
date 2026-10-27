# TextUplink

Run **`app.py`**. It is the only app entry point.

```sh
source .venv/bin/activate
streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

The local models and the Python environment are already set up on this Mac.
Restart Streamlit after pulling or applying code changes.

## Test your recordings

1. Select **Included noisy sample**, or upload your noisy recording. Leave added noise off.
2. Choose **DeepFilterNet3** or **MossFormer2** and listen to the processed speech. Disable
   voice boost when comparing noise-removal quality.
3. Optionally transcribe using **Qwen3-ASR** and generate reviewable emotion-color estimates.
4. Select **EnCodec**, **Mimi**, or **FocalCodec**, then encode and reconstruct.
5. Listen to the reconstructed WAV. Supply your matching aligned clean recording above
   for ground-truth comparisons.

The comparison buttons let you compare enhancement models or codecs on the same input.
The Receiver tab reconstructs from a downloaded `.tuplink` packet and local model weights.
Emotion colors are estimates; listen to recordings to assess preservation of expression.
Packet downloads are currently unencrypted.

## Project folders

- `app.py` — run this with Streamlit.
- `textuplink/` — supporting audio, transcript, codec and worker code.
- `samples/` — the included clean and noisy recordings.
- `models/` — pretrained local weights; needed for offline inference.
- `scripts/` — one-time model setup, not the app.
- `requirements/` — dependencies for the single environment.
- `tests/` — verification code.
- `docs/TECHNICAL.md` — detailed setup, model behavior and research limitations.

Only **`.venv`** is used. Both neural enhancement and Qwen run from this environment.

## Verify

```sh
python -m unittest discover -s tests -v
```

For a new machine or model setup, follow [the technical guide](docs/TECHNICAL.md).
