"""Local voice isolation for expressive captions; no gain processing."""
from pathlib import Path
import subprocess
import os
import sys
import tempfile
import numpy as np


ENHANCERS={'reuse':'NVIDIA NemotronLabs RE-USE', 'spectral':'Spectral gating (baseline)', 'deepfilter':'DeepFilterNet3', 'mossformer':'MossFormer2 SE'}
MODEL_PATHS={'reuse':'reuse','deepfilter':'deepfilternet3','mossformer':'mossformer2'}


def neural_enhance(audio,backend):
    root=Path(__file__).resolve().parent.parent
    runtime=Path(sys.executable)
    model=root/'models'/MODEL_PATHS[backend]
    if not runtime.is_file() or not model.is_dir():
        raise ValueError(f'Set up the enhancement runtime and run python scripts/provision_models.py {backend}; see README.md')
    with tempfile.TemporaryDirectory(prefix='textuplink-enhance-') as work:
        incoming=Path(work)/'input.npy'
        outgoing=Path(work)/'output.npy'
        np.save(incoming,audio,allow_pickle=False)
        result=subprocess.run([str(runtime),'-m','textuplink.enhancement_worker',backend,str(incoming),str(outgoing),str(model)],
                              capture_output=True,text=True,timeout=300,cwd=str(root),env=dict(os.environ,PYTHONHASHSEED='0'))
        if result.returncode:
            raise RuntimeError(f'{ENHANCERS[backend]} failed: {result.stderr[-2000:] or result.stdout[-2000:]}')
        if not outgoing.is_file():
            raise RuntimeError('Enhancement worker produced no audio')
        return np.load(outgoing,allow_pickle=False)


def enhance_speech(audio, reduce_noise=True, strength=1., backend='reuse'):
    """Suppress noise without changing recording gain or relative delivery."""
    if backend not in ENHANCERS:
        raise ValueError('Unknown enhancement backend')
    y=np.asarray(audio,dtype=np.float32)
    if y.ndim!=1 or not y.size or not np.isfinite(y).all():
        raise ValueError('Use non-empty finite mono audio')
    if not np.isfinite(strength) or not 0<=strength<=1:
        raise ValueError('Invalid enhancement mix')
    cleaned=y.copy()
    if reduce_noise and strength>0:
        if backend=='spectral':
            import noisereduce as nr
            cleaned=np.asarray(nr.reduce_noise(y=cleaned,sr=24000,stationary=False,
                                              prop_decrease=strength),dtype=np.float32)
        else:
            enhanced=neural_enhance(cleaned,backend)
            if enhanced.shape!=y.shape or not np.isfinite(enhanced).all():
                raise ValueError('Neural enhancer returned invalid audio')
            cleaned=((1-strength)*y+strength*enhanced).astype(np.float32)
    if not np.isfinite(cleaned).all():
        raise ValueError('Noise reduction returned invalid audio')
    runtime=('SpeechLens CLI v1.0 / MLX Metal (strict), NemotronLabs RE-USE 31f82f2' if backend=='reuse'
             else 'spectral' if backend=='spectral' else 'MLX-Audio 0.6.2 inference subset / MLX 0.32.4 (Apple Silicon)')
    stats=dict(enhancement_backend=backend,enhancement_runtime=runtime,
               noise_reduction=bool(reduce_noise),
               enhanced_mix=float(strength) if reduce_noise else None)
    return cleaned,stats
