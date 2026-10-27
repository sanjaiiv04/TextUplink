"""Local Apple Silicon enhancement worker; no inference-time network access."""
import argparse
import os
from pathlib import Path
import socket

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'


def forbid_network(*args,**kwargs):
    raise RuntimeError('Network access is disabled during enhancement; provision weights first')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend',choices=['reuse','deepfilter','mossformer'])
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('model_dir')
    args=parser.parse_args()
    # Keep local-only behavior even if an upstream loader changes its defaults.
    socket.socket.connect=forbid_network
    socket.create_connection=forbid_network
    import numpy as np
    import librosa
    root=Path(args.model_dir).resolve()
    if not root.is_dir():
        raise ValueError(f'Model directory is missing: {root}')
    audio=np.load(args.input,allow_pickle=False)
    if audio.ndim!=1 or not 2400<=len(audio)<=24000*60 or not np.isfinite(audio).all():
        raise ValueError('Worker expects finite 24 kHz mono audio from 0.1 to 60 seconds')
    if args.backend=='reuse':
        import hashlib
        import json
        import subprocess
        import tempfile
        import soundfile as sf
        manifest=json.loads((root/'manifest.json').read_text())
        required=['model_mlx.safetensors','runtime/speechlens-cli','runtime/mlx.metallib','runtime/ffmpeg','runtime/ffprobe']
        for name in required:
            path=root/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256'].get(name):
                raise ValueError(f'RE-USE artifact missing or changed: {name}; provision it again')
        with tempfile.TemporaryDirectory(prefix='textuplink-reuse-') as work:
            incoming=Path(work)/'input.wav'
            outgoing=Path(work)/'output.wav'
            sf.write(incoming,audio,24000,subtype='FLOAT')
            # Deny network to the native process too; Python socket guards alone
            # cannot constrain a subprocess. No automatic downloads or telemetry.
            command=['/usr/bin/sandbox-exec','-p','(version 1)(allow default)(deny network*)',
                     str(root/'runtime'/'speechlens-cli'),'--weights',str(root/'model_mlx.safetensors'),
                     '--input',str(incoming),'--output',str(outgoing),'--chunk-seconds','5','--strict']
            result=subprocess.run(command,capture_output=True,text=True,timeout=280)
            if result.returncode or not outgoing.is_file():
                raise RuntimeError(f'RE-USE native inference failed: {result.stderr[-2000:] or result.stdout[-2000:]}')
            enhanced,sample_rate=sf.read(outgoing,dtype='float32')
            if sample_rate!=24000 or enhanced.shape!=audio.shape or not np.isfinite(enhanced).all():
                raise ValueError('RE-USE returned invalid audio, sample rate or duration')
            np.save(args.output,enhanced,allow_pickle=False)
        return
    fullband=librosa.resample(audio,orig_sr=24000,target_sr=48000).astype(np.float32)
    import mlx.core as mx
    mx.random.seed(0)
    if args.backend=='deepfilter':
        from textuplink._vendor.mlx_audio.sts.models.deepfilternet.model import DeepFilterNetModel
        model=DeepFilterNetModel.from_pretrained(str(root),subfolder='v3')
        model.eval()
        enhanced=model.enhance_array(fullband)
    else:
        from textuplink._vendor.mlx_audio.sts.models.mossformer2_se.model import MossFormer2SEModel
        model=MossFormer2SEModel.from_pretrained(str(root))
        model.eval()
        enhanced=model.enhance(fullband)
    enhanced=np.asarray(enhanced,dtype=np.float32).reshape(-1)
    if not np.isfinite(enhanced).all() or abs(len(enhanced)-len(fullband))>480:
        raise ValueError('Enhancement returned invalid audio or changed duration unexpectedly')
    enhanced=librosa.resample(enhanced,orig_sr=48000,target_sr=24000)
    if len(enhanced)<len(audio):
        enhanced=np.pad(enhanced,(0,len(audio)-len(enhanced)))
    np.save(args.output,enhanced[:len(audio)].astype(np.float32),allow_pickle=False)


if __name__=='__main__':
    main()
