"""Local transcript timing and reviewable phrase-context emotion annotations."""
from pathlib import Path
import numpy as np

LABELS = ['Unlabelled', 'Uncertain', 'Neutral', 'Happy', 'Angry', 'Sad']
COLORS = {'Unlabelled':'#64748b', 'Uncertain':'#64748b', 'Neutral':'#475569',
          'Happy':'#15803d', 'Angry':'#c2410c', 'Sad':'#1d4ed8'}
MODEL_LABELS = {'neu':'Neutral', 'hap':'Happy', 'ang':'Angry', 'sad':'Sad'}


def load_transcriber(root):
    import torch
    from qwen_asr import Qwen3ASRModel
    from huggingface_hub import snapshot_download
    root=Path(root)
    asr=root/'models/qwen3-asr'
    aligner=root/'models/qwen3-aligner'
    return Qwen3ASRModel.from_pretrained(
        str(asr) if asr.is_dir() else snapshot_download('Qwen/Qwen3-ASR-0.6B',local_files_only=True),
        dtype=torch.float32,device_map='cpu',local_files_only=True,
        max_inference_batch_size=1,max_new_tokens=1024,
        forced_aligner=str(aligner) if aligner.is_dir() else snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B',local_files_only=True),
        forced_aligner_kwargs=dict(dtype=torch.float32,device_map='cpu',local_files_only=True))


def transcribe_words(audio, model):
    import librosa
    speech = librosa.resample(np.asarray(audio,dtype=np.float32),orig_sr=24000,target_sr=16000)
    results=model.transcribe(audio=(speech,16000),language=None,return_time_stamps=True)
    if not results:
        return [],'Unknown'
    result=results[0]
    rows=[]
    for unit in result.time_stamps or []:
        if not unit.text.strip():
            continue
        start,end=float(unit.start_time),float(unit.end_time)
        if not np.isfinite([start,end]).all() or start<0 or start>len(audio)/24000 or end<start or end>len(audio)/24000+.1:
            raise ValueError('Qwen returned invalid alignment times; no word colors were generated')
        rows.append(dict(word=unit.text.strip(),start_s=start,end_s=min(end,len(audio)/24000),
                         emotion='Unlabelled',annotation_source='None',model_score=None))
    if result.text.strip() and not rows:
        raise ValueError('Qwen returned text without word alignment. Check the forced aligner and language support')
    return rows,str(result.language)


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


def annotate_context(rows,audio,estimator,threshold=.6,window_s=3.):
    """Assign a phrase-window estimate to words by midpoint, never classify isolated words."""
    duration=len(audio)/24000
    windows={}
    annotated=[]
    for row in rows:
        midpoint=(float(row['start_s'])+float(row['end_s']))/2
        bucket=max(0,int(midpoint//window_s))
        if bucket not in windows:
            start=max(0.,min(bucket*window_s,max(0.,duration-window_s)))
            end=min(duration,start+window_s)
            label,score=estimator.estimate(audio[round(start*24000):round(end*24000)])
            windows[bucket]=(label if score is not None and score>=threshold else 'Uncertain',score,start,end,label)
        label,score,start,end,model_label=windows[bucket]
        annotated.append(dict(row,emotion=label,annotation_source='Phrase-context model estimate',
                              model_score=score,model_emotion=model_label,context_start_s=start,context_end_s=end))
    return annotated


def review_annotations(edited,baseline):
    """Keep model estimates and human overrides distinguishable in exports."""
    result=[]
    for row,original in zip(edited,baseline):
        entry=dict(original)
        entry.setdefault('model_emotion',original['emotion'] if original.get('annotation_source')=='Phrase-context model estimate' else None)
        entry['word']=str(row['word'])
        label=row['emotion'] if row['emotion'] in LABELS else 'Unlabelled'
        entry['emotion']=label
        if label!=original['emotion']:
            entry['annotation_source']='Human override'
        result.append(entry)
    return result
