"""One-time setup for the three local model roles used by the current app."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

REPOS={
    'qwen':('Qwen/Qwen3-ASR-0.6B','qwen3-asr',None),
    'emotion':('superb/wav2vec2-base-superb-er','emotion',None),
    'emotion2vec':('emotion2vec/emotion2vec_plus_large','emotion2vec_plus_large',
                   '6c303ba987b86b93193de93e34bb2b077a6bedc4'),
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',choices=REPOS)
    parser.add_argument('--local-only',action='store_true')
    args=parser.parse_args()
    repo,directory,revision=REPOS[args.model]
    destination=Path(__file__).resolve().parent.parent/'models'/directory
    patterns={'qwen':['*.json','*.txt','*.safetensors'],
              'emotion':['config.json','preprocessor_config.json','pytorch_model.bin'],
              'emotion2vec':['config.yaml','configuration.json','model.pt','tokens.txt','README.md']}[args.model]
    snapshot_download(repo_id=repo,local_dir=str(destination),revision=revision,
                      allow_patterns=patterns,local_files_only=args.local_only)
    print(f'Model ready for offline inference: {destination}')

if __name__=='__main__':main()
