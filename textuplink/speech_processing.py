"""Local noise suppression followed by transparent, peak-safe speech-level gain."""
from pathlib import Path
import subprocess
import os
import sys
import tempfile
import numpy as np


ENHANCERS={'spectral':'Spectral gating (baseline)', 'deepfilter':'DeepFilterNet3', 'mossformer':'MossFormer2 SE'}
MODEL_PATHS={'deepfilter':'deepfilternet3','mossformer':'mossformer2'}


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


def enhance_speech(audio, reduce_noise=True, strength=.5, boost_db=9., steady_noise=False, ceiling_dbfs=-1., backend='spectral'):
    """Use one gain for the recording to preserve relative emphasis and timing.

    The selected backend suppresses noise; it does not guarantee isolation. Gain
    also raises any remaining noise. Avoid clipping by reducing the applied gain
    for the entire recording rather than compressing individual peaks.
    """
    if backend not in ENHANCERS:
        raise ValueError('Unknown enhancement backend')
    y=np.asarray(audio,dtype=np.float32)
    if y.ndim!=1 or not y.size or not np.isfinite(y).all():
        raise ValueError('Use non-empty finite mono audio')
    if not np.isfinite([strength,boost_db,ceiling_dbfs]).all() or not 0<=strength<=1 or not 0<=boost_db<=24 or not -12<=ceiling_dbfs<=0:
        raise ValueError('Invalid noise suppression or gain settings')
    cleaned=y.copy()
    if reduce_noise and strength>0:
        if backend=='spectral':
            import noisereduce as nr
            cleaned=np.asarray(nr.reduce_noise(y=cleaned,sr=24000,stationary=steady_noise,
                                              prop_decrease=strength),dtype=np.float32)
        else:
            enhanced=neural_enhance(cleaned,backend)
            if enhanced.shape!=y.shape or not np.isfinite(enhanced).all():
                raise ValueError('Neural enhancer returned invalid audio')
            cleaned=((1-strength)*y+strength*enhanced).astype(np.float32)
    if not np.isfinite(cleaned).all():
        raise ValueError('Noise reduction returned invalid audio')
    ceiling=10**(ceiling_dbfs/20)
    peak=float(np.max(np.abs(cleaned)))
    requested=10**(boost_db/20)
    gain=min(requested,ceiling/peak) if peak>1e-8 else 1.
    processed=(cleaned*gain).astype(np.float32)
    stats=dict(enhancement_backend=backend,enhancement_runtime='spectral' if backend=='spectral' else 'MLX-Audio 0.6.2 inference subset / MLX 0.32.4 (Apple Silicon)',noise_reduction=bool(reduce_noise),noise_reduction_strength=float(strength) if reduce_noise else None,
               noise_mode='steady' if steady_noise else 'changing',requested_boost_db=float(boost_db),
               applied_gain_db=float(20*np.log10(gain)),peak_ceiling_dbfs=float(ceiling_dbfs),
               peak_limited=bool(gain<requested and peak>1e-8),
               input_rms_dbfs=float(20*np.log10(max(float(np.sqrt(np.mean(y.astype(np.float64)**2))),1e-12))),
               cleaned_rms_dbfs=float(20*np.log10(max(float(np.sqrt(np.mean(cleaned.astype(np.float64)**2))),1e-12))),
               output_rms_dbfs=float(20*np.log10(max(float(np.sqrt(np.mean(processed.astype(np.float64)**2))),1e-12))))
    return processed,stats
