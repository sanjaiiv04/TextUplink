"""Local Qwen3 ASR and the four-label comparison emotion model."""
from pathlib import Path
import numpy as np

MODEL_LABELS = {'neu':'Neutral', 'hap':'Happy', 'ang':'Angry', 'sad':'Sad'}


def load_transcriber(root, *, device='cpu'):
    import torch
    from qwen_asr import Qwen3ASRModel
    from huggingface_hub import snapshot_download
    asr=Path(root)/'models/qwen3-asr'
    source=str(asr) if asr.is_dir() else snapshot_download('Qwen/Qwen3-ASR-0.6B',local_files_only=True)
    return Qwen3ASRModel.from_pretrained(source,dtype=torch.float32,device_map=device,
        local_files_only=True,max_inference_batch_size=1,max_new_tokens=1024)


class LocalEmotion:
    def __init__(self,path):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        if not (Path(path)/'config.json').is_file():
            raise ValueError('Provision the optional emotion model: python scripts/provision_models.py emotion')
        self.extractor=Wav2Vec2FeatureExtractor.from_pretrained(str(path),local_files_only=True)
        self.model=Wav2Vec2ForSequenceClassification.from_pretrained(str(path),local_files_only=True).eval()

    def estimate(self,audio):
        import torch
        import librosa
        speech=librosa.resample(np.asarray(audio,dtype=np.float32),orig_sr=24000,target_sr=16000)
        if len(speech)<16000 or np.sqrt(np.mean(speech.astype(np.float64)**2))<1e-5:
            return 'Uncertain',None
        inputs=self.extractor(speech,sampling_rate=16000,return_tensors='pt')
        with torch.inference_mode():
            scores=self.model(**inputs).logits.softmax(dim=-1)[0]
        idx=int(scores.argmax())
        return MODEL_LABELS.get(self.model.config.id2label[idx],'Uncertain'),float(scores[idx])
