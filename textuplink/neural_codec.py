"""Local-only neural codecs. Receiver reconstructs using packet + matching weights."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import hashlib
import io
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from textuplink.codec_packet import pack_codes, unpack_codes

MODEL_DIRS = {'encodec': 'models/encodec_24khz', 'mimi': 'models/mimi', 'focalcodec':'models/focalcodec'}


def model_fingerprint(path):
    root = Path(path)
    weights = sorted(root.glob('*.safetensors')) or sorted(root.glob('pytorch_model*.bin')) or sorted(root.glob('model.pt'))
    files = [root/('model.json' if (root/'model.json').is_file() else 'config.json'), *weights]
    if len(files) < 2 or not all(p.is_file() for p in files):
        raise ValueError('Provision the model config and weights first; see README.md')
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        with file.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


class LocalCodec:
    def __init__(self, family, path, device='cpu'):
        import torch
        from transformers import EncodecModel, MimiModel
        if family not in MODEL_DIRS:
            raise ValueError('Unsupported codec')
        self.family, self.device = family, device
        self.fingerprint = model_fingerprint(path)
        if family=='focalcodec':
            from focalcodec import FocalCodec
            config=Path(path)/'model.json'
            if not config.is_file():
                raise ValueError('Provision FocalCodec using python scripts/provision_models.py focalcodec')
            self.model=FocalCodec.from_pretrained(str(config)).to(device).eval()
            if self.model.sample_rate_input!=16000 or self.model.sample_rate_output!=16000 or self.model.quantizer_config.get('codebook_size')!=8192:
                raise ValueError('Use the supported 50 Hz, 8192-entry FocalCodec checkpoint')
            return
        cls = EncodecModel if family == 'encodec' else MimiModel
        self.model = cls.from_pretrained(str(path), local_files_only=True).to(device).eval()
        config = self.model.config
        if config.sampling_rate != 24000 or config.audio_channels != 1:
            raise ValueError('Only 24 kHz mono models are supported')
        if config.codebook_size != (1024 if family == 'encodec' else 2048):
            raise ValueError('Unsupported codebook cardinality')
        if family == 'encodec' and (config.normalize or config.chunk_length_s is not None):
            raise ValueError('Use the non-normalizing, unchunked EnCodec 24 kHz checkpoint')

    def encode(self, audio, setting):
        import torch
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not 1 <= len(audio) <= 24_000*60 or not np.isfinite(audio).all():
            raise ValueError('Use finite mono audio up to 60 seconds')
        x = torch.from_numpy(audio.copy())[None,None,:].to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            if self.family=='focalcodec':
                import librosa
                if float(setting)!=.65:
                    raise ValueError('This FocalCodec checkpoint operates at nominal 0.65 kb/s')
                native=librosa.resample(audio,orig_sr=24000,target_sr=16000)
                tokens=self.model.sig_to_toks(torch.from_numpy(native.copy())[None,:].to(self.device))
                codes=tokens[:,None,:].cpu().numpy()
            elif self.family == 'encodec':
                if float(setting) not in self.model.config.target_bandwidths:
                    raise ValueError('Unsupported EnCodec bandwidth')
                result = self.model.encode(x, bandwidth=float(setting))
            else:
                if int(setting) not in [4, 8, 16, 32]:
                    raise ValueError('Choose 4, 8, 16 or 32 Mimi codebooks')
                result = self.model.encode(x, num_quantizers=int(setting))
        if self.family!='focalcodec':
            codes = result.audio_codes.cpu().numpy()
        packet = pack_codes(codes, family=self.family, model_fingerprint=self.fingerprint,
                            samples=len(audio), setting=setting)
        return packet, time.perf_counter()-started

    def decode(self, packet):
        import torch
        header, codes = unpack_codes(packet)
        if header['family'] != self.family or header['fingerprint'] != self.fingerprint:
            raise ValueError('Receiver needs the same codec and exact model weights as the sender')
        books = codes.shape[-2]
        allowed = ([int(b*1000/(75*10)) for b in self.model.config.target_bandwidths]
                   if self.family == 'encodec' else ([1] if self.family=='focalcodec' else [4,8,16,32]))
        if books not in allowed:
            raise ValueError('Unsupported number of codebooks')
        tensor = torch.from_numpy(codes).to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            if self.family=='focalcodec':
                import math
                native_samples=math.ceil(header['samples']*16000/24000)
                native=self.model.toks_to_sig(tensor[:,0,:],output_length=native_samples)
                import librosa
                decoded=librosa.resample(native.detach().cpu().numpy().reshape(-1),orig_sr=16000,target_sr=24000)
            elif self.family == 'encodec':
                result = self.model.decode(tensor, audio_scales=[None]*codes.shape[0])
            else:
                result = self.model.decode(tensor)
        if self.family!='focalcodec':
            decoded = result.audio_values.detach().cpu().numpy().reshape(-1)
        if len(decoded) < header['samples'] or not np.isfinite(decoded).all():
            raise ValueError('Decoder returned invalid or insufficient audio')
        return decoded[:header['samples']], time.perf_counter()-started


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
    # Float WAV avoids hiding reconstruction clipping in evaluation downloads.
    sf.write(buffer, y, 24000, format='WAV', subtype='FLOAT')
    return buffer.getvalue()


def add_noise(clean, snr_db, seed=7):
    noise = np.random.default_rng(seed).normal(size=len(clean)).astype(np.float32)
    rms = float(np.sqrt(np.mean(np.asarray(clean, dtype=np.float64)**2)))
    if rms < 1e-8:
        raise ValueError('Cannot set a meaningful SNR for a silent recording')
    noise *= rms / (10**(snr_db/20) * np.sqrt(np.mean(noise.astype(np.float64)**2)))
    noisy = clean + noise
    gain = min(1., .99/max(float(np.max(np.abs(noisy))), 1e-8))
    return (clean*gain).astype(np.float32), (noisy*gain).astype(np.float32)


def compare_audio(reference, reconstruction):
    """Descriptive metrics for aligned audio, not emotion or intelligibility scores."""
    import librosa
    n = min(len(reference), len(reconstruction))
    ref, rec = np.asarray(reference[:n], dtype=np.float64), np.asarray(reconstruction[:n], dtype=np.float64)
    if n < 1024:
        raise ValueError('Use at least 1024 samples for comparison')
    eps=1e-12
    error = np.mean((ref-rec)**2)
    rms_ref, rms_rec = np.sqrt(np.mean(ref**2)), np.sqrt(np.mean(rec**2))
    spectral_ref = np.maximum(np.abs(librosa.stft(ref, n_fft=1024, hop_length=240)), 1e-5)
    spectral_rec = np.maximum(np.abs(librosa.stft(rec, n_fft=1024, hop_length=240)), 1e-5)
    lsd = np.mean(np.sqrt(np.mean((20*np.log10(spectral_ref/spectral_rec))**2, axis=0)))
    energy_ref = librosa.feature.rms(y=ref, frame_length=1024,hop_length=240)[0]
    energy_rec = librosa.feature.rms(y=rec, frame_length=1024,hop_length=240)[0]
    corr = float(np.corrcoef(energy_ref,energy_rec)[0,1]) if min(np.std(energy_ref),np.std(energy_rec))>1e-8 else None
    return dict(snr_db=float(10*np.log10((rms_ref**2+eps)/(error+eps))),
                log_spectral_distance_db=float(lsd), energy_contour_correlation=corr,
                level_change_db=float(20*np.log10((rms_rec+eps)/(rms_ref+eps))),
                output_peak=float(np.max(np.abs(rec))), compared_samples=n)
