"""Explicit one-time download step; runtime inference does not access the network."""
import argparse
from pathlib import Path

REPOS = {'encodec': ('facebook/encodec_24khz', 'encodec_24khz'), 'mimi': ('kyutai/mimi','mimi'),
         'emotion': ('superb/wav2vec2-base-superb-er','emotion'),
         'qwen': ('Qwen/Qwen3-ASR-0.6B','qwen3-asr'),
         'aligner': ('Qwen/Qwen3-ForcedAligner-0.6B','qwen3-aligner'),
         'deepfilter': ('mlx-community/DeepFilterNet-mlx','deepfilternet3'),
         'mossformer': ('starkdmi/MossFormer2_SE_48K_MLX','mossformer2'),
         'focalcodec': ('lucadellalib/focalcodec_50hz','focalcodec')}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('codec', choices=REPOS)
    parser.add_argument('--local-only',action='store_true',help='Finalize models already downloaded without network access')
    args=parser.parse_args()
    from huggingface_hub import snapshot_download
    repo,directory=REPOS[args.codec]
    destination=Path(__file__).resolve().parent.parent/'models'/directory
    patterns=['config.json','*.safetensors','*.safetensors.index.json']
    if args.codec=='emotion':
        from huggingface_hub import list_repo_files
        files=list_repo_files(repo)
        patterns=['config.json','preprocessor_config.json',
                  'model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin']
    if args.codec in ['qwen','aligner']:
        patterns=['*.json','*.txt','*.safetensors']
    if args.codec=='deepfilter':
        patterns=['v3/config.json','v3/model.safetensors']
    if args.codec=='focalcodec':
        patterns=['config.json','model.safetensors']
    if args.codec=='mossformer':
        patterns=['model_fp32.safetensors']
    snapshot_download(repo_id=repo,local_dir=str(destination),allow_patterns=patterns,local_files_only=args.local_only)
    if args.codec=='focalcodec':
        import shutil
        # Upstream local loader pairs model.json with model.safetensors/model.pt.
        shutil.copyfile(destination/'config.json',destination/'model.json')
    if args.codec=='mossformer':
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
    print(f'Model provisioned at {destination}. Copy this directory to the receiver for offline use.')

if __name__=='__main__':
    main()
