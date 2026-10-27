"""Local audio loading and WAV export for the caption prototype."""
import io
import numpy as np
import soundfile as sf

def read_audio(raw):
    import librosa
    with sf.SoundFile(io.BytesIO(raw)) as file:
        if file.frames/file.samplerate > 60:
            raise ValueError('Use a recording of at most 60 seconds')
        y = file.read(dtype='float32', always_2d=True).mean(axis=1)
        sr = file.samplerate
    if not y.size or not np.isfinite(y).all():
        raise ValueError('Audio is empty or contains non-finite samples')
    return librosa.resample(y, orig_sr=sr, target_sr=24000).astype(np.float32) if sr != 24000 else y


def wav_bytes(y):
    buffer = io.BytesIO()
    # Float WAV preserves the processed waveform without export clipping.
    sf.write(buffer, y, 24000, format='WAV', subtype='FLOAT')
    return buffer.getvalue()

