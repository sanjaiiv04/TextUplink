"""Explicit one-time download step; runtime inference does not access the network."""
import argparse
import hashlib
import json
from pathlib import Path

REUSE_REVISION='31f82f2efad1095034492f236e8dac9239ba713b'
REUSE_CLI_URL='https://github.com/faraday/SpeechLens/releases/download/v1.0/speechlens-cli-1.0.zip'
REUSE_CLI_SHA256='39cbf274ba93b805eabfa0bb1dd651e8e1a835a101b400fa0ab46304b4b76a04'


def provision_reuse(destination,local_only=False):
    """Pin NVIDIA weights and SpeechLens CLI; convert tensor layout offline."""
    import platform
    import shutil
    import tempfile
    import urllib.request
    import zipfile
    import numpy as np
    from safetensors.numpy import load_file,save_file
    from huggingface_hub import snapshot_download
    if platform.system()!='Darwin' or platform.machine()!='arm64':
        raise ValueError('This RE-USE runtime requires an Apple Silicon Mac')
    if local_only and (destination/'manifest.json').is_file():
        manifest=json.loads((destination/'manifest.json').read_text())
        for name,expected_hash in manifest['sha256'].items():
            if hashlib.sha256((destination/name).read_bytes()).hexdigest()!=expected_hash:
                raise ValueError(f'RE-USE artifact changed: {name}')
        return
    snapshot_download('nvidia/NVIDIA-NemotronLabs-RE-USE',revision=REUSE_REVISION,
                      local_dir=str(destination),allow_patterns=['config.json','model.safetensors','LICENSE','README.md'],
                      local_files_only=local_only)
    config=json.loads((destination/'config.json').read_text())
    expected=dict(num_tfmamba=30,d_state=16,d_conv=4,expand=4,hid_feature=64,
                  compress_factor='relu_log1p',mapping=True,inner_mamba_nlayer=1)
    if any(config['model_cfg'].get(k)!=v for k,v in expected.items()) or config['stft_cfg']!=dict(hop_size=40,n_fft=320,sampling_rate=8000,sfi=True,win_size=320):
        raise ValueError('Checkpoint architecture does not match the pinned SpeechLens runtime')
    # Tensor/key conversion adapted from SpeechLens WeightPort.swift (Apache-2.0),
    # Copyright 2026 Çağatay Çallı, revision f6c640fce1d265e8b4b7564ec81dff1f08d67200.
    # See docs/licenses/SpeechLens-Apache-2.0.txt. Runtime third-party licenses
    # are also retained beside the CLI.
    renames=[('dense_encoder.','denseEncoder.'),('mask_decoder.','maskDecoder.'),
             ('phase_decoder.','phaseDecoder.'),('TSMamba.','tfMamba.'),
             ('time_mamba.','timeMamba.'),('freq_mamba.','freqMamba.'),
             ('forward_blocks.','forward.'),('backward_blocks.','backward.'),
             ('output_proj.','outputProj.'),('in_proj.','inProj.'),('out_proj.','outProj.'),
             ('x_proj.','xProj.'),('dt_proj.','dtProj.'),('final_conv.','finalConv.'),
             ('phase_conv_r.','phaseConvR.'),('phase_conv_i.','phaseConvI.')]
    mapped={}
    for key,tensor in load_file(str(destination/'model.safetensors')).items():
        name=key.removeprefix('model.')
        for i in range(10):
            for j in range(10):
                name=name.replace(f'dense_block.dense_block.{i}.{j}.',f'denseBlock.layers.{i}.layers.{j}.')
        for old,new in renames:
            name=name.replace(old,new)
        for old,new in zip(['dense_conv_1','dense_conv_2','up_conv1','up_conv2'],['conv1','conv2','upConv1','upConv2']):
            for i in range(10):
                name=name.replace(f'{old}.{i}.',f'{new}.layers.{i}.')
        if 'conv1d' in key and key.endswith('weight') and tensor.ndim==3:
            tensor=tensor.transpose(0,2,1)
        elif key.endswith('weight') and tensor.ndim==4:
            tensor=tensor.transpose(0,2,3,1)
        elif 'dt_proj' in key and key.endswith('bias'):
            tensor=tensor.astype(np.float32)
        if name in mapped:
            raise ValueError(f'Duplicate converted tensor: {name}')
        mapped[name]=np.ascontiguousarray(tensor)
    save_file(mapped,str(destination/'model_mlx.safetensors'))
    # Source tensors are only needed during conversion, not app inference.
    (destination/'model.safetensors').unlink()
    runtime=destination/'runtime'
    if not (runtime/'speechlens-cli').is_file():
        if local_only:
            raise ValueError('RE-USE CLI is missing; run provisioning once without --local-only')
        with tempfile.TemporaryDirectory(prefix='textuplink-reuse-setup-') as tmp:
            archive=Path(tmp)/'runtime.zip'
            urllib.request.urlretrieve(REUSE_CLI_URL,archive)
            if hashlib.sha256(archive.read_bytes()).hexdigest()!=REUSE_CLI_SHA256:
                raise ValueError('RE-USE CLI archive checksum mismatch')
            with zipfile.ZipFile(archive) as bundle:
                # Copy only top-level runtime artifacts, never arbitrary archive paths.
                runtime.mkdir(parents=True,exist_ok=True)
                for member in bundle.infolist():
                    path=Path(member.filename)
                    if len(path.parts)==2 and path.parts[0]=='speechlens-cli' and not path.name.startswith('.') and not member.is_dir():
                        with bundle.open(member) as source,(runtime/path.name).open('wb') as target:
                            shutil.copyfileobj(source,target)
            for name in ['speechlens-cli','ffmpeg','ffprobe']:
                (runtime/name).chmod(0o755)
    artifacts={name:hashlib.sha256((destination/name).read_bytes()).hexdigest()
               for name in ['model_mlx.safetensors','runtime/speechlens-cli','runtime/mlx.metallib','runtime/ffmpeg','runtime/ffprobe']}
    (destination/'manifest.json').write_text(json.dumps(dict(repo='nvidia/NVIDIA-NemotronLabs-RE-USE',
        revision=REUSE_REVISION,runtime='SpeechLens CLI v1.0 (strict)',runtime_archive_sha256=REUSE_CLI_SHA256,sha256=artifacts),indent=2))

REPOS = {'emotion': ('superb/wav2vec2-base-superb-er','emotion'),
         'qwen': ('Qwen/Qwen3-ASR-0.6B','qwen3-asr'),
         'aligner': ('Qwen/Qwen3-ForcedAligner-0.6B','qwen3-aligner'),
         'deepfilter': ('mlx-community/DeepFilterNet-mlx','deepfilternet3'),
         'mossformer': ('starkdmi/MossFormer2_SE_48K_MLX','mossformer2'),
         'reuse': ('nvidia/NVIDIA-NemotronLabs-RE-USE','reuse')}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=REPOS)
    parser.add_argument('--local-only',action='store_true',help='Finalize models already downloaded without network access')
    args=parser.parse_args()
    from huggingface_hub import snapshot_download
    repo,directory=REPOS[args.model]
    destination=Path(__file__).resolve().parent.parent/'models'/directory
    if args.model=='reuse':
        provision_reuse(destination,args.local_only)
        print(f'RE-USE ready for offline inference at {destination}')
        return
    patterns=['config.json','*.safetensors','*.safetensors.index.json']
    if args.model=='emotion':
        from huggingface_hub import list_repo_files
        files=list_repo_files(repo)
        patterns=['config.json','preprocessor_config.json',
                  'model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin']
    if args.model in ['qwen','aligner']:
        patterns=['*.json','*.txt','*.safetensors']
    if args.model=='deepfilter':
        patterns=['v3/config.json','v3/model.safetensors']
    if args.model=='mossformer':
        patterns=['model_fp32.safetensors']
    snapshot_download(repo_id=repo,local_dir=str(destination),allow_patterns=patterns,local_files_only=args.local_only)
    if args.model=='mossformer':
        import json
        # Default MossFormer2SEConfig from the pinned mlx-audio 0.6.2 runtime.
        config=dict(sample_rate=48000,win_len=1920,win_inc=384,fft_len=1920,win_type='hamming',
                    num_mels=60,preemphasis=.97,one_time_decode_length=20,decode_window=4,
                    chunk_seconds=4.,chunk_overlap=.25,auto_chunk_threshold=60.,
                    in_channels=180,out_channels=512,out_channels_final=961,num_blocks=24)
        (destination/'config.json').write_text(json.dumps(config,indent=2))
        alias=destination/'model.safetensors'
        if not alias.exists():
            alias.symlink_to('model_fp32.safetensors')
    print(f'Model provisioned at {destination}. Keep this directory for offline use.')

if __name__=='__main__':
    main()
