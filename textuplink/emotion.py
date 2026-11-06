"""Local utterance-emotion evaluation, with explicit quality and ambiguity gates."""
import hashlib
from pathlib import Path
import time
import numpy as np

EMOTION_REPO='emotion2vec/emotion2vec_plus_large'
EMOTION_REVISION='6c303ba987b86b93193de93e34bb2b077a6bedc4'
LABEL_MAP={name.lower():name for name in ['Angry','Disgusted','Fearful','Happy','Neutral','Other','Sad','Surprised','Unknown']}


def gate_prediction(scores, *, duration_s, rms, clipping_fraction, one_speaker,
                    minimum_score=.7, minimum_margin=.15, reference_label=None, reference_score=None):
    """Conservative prototype gates, not a calibrated confidence/accuracy claim."""
    if not 0<=minimum_score<=1 or not 0<=minimum_margin<=1:
        raise ValueError('Invalid decision thresholds')
    if not np.isfinite([duration_s,rms,clipping_fraction]).all() or duration_s<=0 or rms<0 or not 0<=clipping_fraction<=1:
        raise ValueError('Invalid audio quality measurements')
    if reference_score is not None and (not np.isfinite(reference_score) or not 0<=reference_score<=1):
        raise ValueError('Invalid comparison-model score')
    if len(scores)<2 or any(not np.isfinite(v) or not 0<=v<=1 for v in scores.values()):
        raise ValueError('Invalid emotion scores')
    ranked=sorted(scores.items(),key=lambda item:item[1],reverse=True)
    (top,score),(second,runner_up)=ranked[:2]
    reasons=[]
    if duration_s<1.:reasons.append('Less than one second of context')
    if rms<.001:reasons.append('Very quiet or silent input')
    if clipping_fraction>.01:reasons.append('More than 1% of samples near clipping')
    if not one_speaker:reasons.append('Single clear speaker has not been confirmed')
    if top in ['Other','Unknown']:reasons.append('Model chose an unsupported/unknown category')
    if score<minimum_score:reasons.append('Top score is below the selected threshold')
    shared={'Angry','Happy','Sad','Neutral'}
    if top=='Angry' and (reference_label!='Angry' or reference_score is None or reference_score<.6):
        reasons.append('Anger is not corroborated by the comparison model')
    elif top in shared and reference_label in shared and reference_score is not None and reference_score>=.6 and reference_label!=top:
        reasons.append('The two models disagree on a shared category')
    margin=float(score-runner_up)
    if margin<minimum_margin:reasons.append('Top two alternatives are too close')
    return dict(estimate='Uncertain' if reasons else top,top_label=top,top_score=float(score),
                second_label=second,second_score=float(runner_up),margin=margin,
                abstained=bool(reasons),reasons=reasons,
                hypothetical_cue=f'Possibly {top.lower()}' if not reasons and top!='Neutral' else None)


class Emotion2Vec:
    def __init__(self,path):
        self.path=Path(path).resolve()
        required=['model.pt','config.yaml','configuration.json','tokens.txt']
        if not all((self.path/name).is_file() for name in required):
            raise ValueError('Install the local model: python scripts/provision_models.py emotion2vec')
        started=time.perf_counter()
        from funasr import AutoModel
        # A local directory and disable_update prevent automatic hub/update requests.
        # Native FunASR architecture is used; no downloaded Python model code is trusted.
        self.model=AutoModel(model=str(self.path),device='cpu',hub='hf',
                             disable_update=True,disable_pbar=True,disable_log=True,
                             trust_remote_code=False,ncpu=4,log_level='ERROR')
        self.load_s=time.perf_counter()-started
        with (self.path/'model.pt').open('rb') as handle:
            digest=hashlib.file_digest(handle,'sha256').hexdigest()
        self.provenance=dict(model=EMOTION_REPO,source_revision=EMOTION_REVISION,
                             checkpoint_sha256=digest,device='cpu',runtime='FunASR 1.4.22')

    def distribution(self,audio,sample_rate=24000):
        import librosa
        import torch
        y=np.asarray(audio,dtype=np.float32)
        if y.ndim!=1 or not y.size or not np.isfinite(y).all():
            raise ValueError('Use finite mono speech')
        if not 1<=len(y)/sample_rate<=15:
            raise ValueError('Use one complete phrase, between 1 and 15 seconds')
        y=librosa.resample(y,orig_sr=sample_rate,target_sr=16000) if sample_rate!=16000 else y
        with torch.inference_mode():
            results=self.model.generate(input=y,fs=16000,granularity='utterance',extract_embedding=False,
                                        disable_pbar=True)
        if not results or 'labels' not in results[0] or 'scores' not in results[0]:
            raise ValueError('Emotion model returned no label distribution')
        row=results[0]
        if len(row['labels'])!=len(row['scores']):raise ValueError('Emotion labels and scores do not match')
        scores={}
        for label,score in zip(row['labels'],row['scores']):
            name=str(label).split('/')[-1].strip().lower()
            if name=='<unk>':name='unknown'
            if name not in LABEL_MAP:raise ValueError(f'Unsupported model label: {label}')
            canonical=LABEL_MAP[name]
            if canonical in scores:raise ValueError('Duplicate emotion model label')
            scores[canonical]=float(score)
        if set(scores)!=set(LABEL_MAP.values()) or not np.isfinite(list(scores.values())).all() or not np.isclose(sum(scores.values()),1.,atol=.01):
            raise ValueError('Invalid nine-class model score distribution')
        return scores


def render_emotion_lab(root,get_baseline):
    """Stage 1: explicit phrase tests; does not capture camera or send wearer cues."""
    import io
    import json
    import threading
    import pandas as pd
    import soundfile as sf
    import streamlit as st
    from textuplink.audio import read_audio

    @st.cache_resource
    def load_model():return Emotion2Vec(Path(root)/'models/emotion2vec_plus_large'),threading.Lock()

    st.subheader('Stage 1 · Test vocal emotion on one phrase')
    st.write('Record the same words with different intended deliveries. Start with “oh my god”: angry, surprised, happy and neutral. Keep one speaker in each clip.')
    st.caption('Stop live transcription before recording a lab clip, so microphone use and timing are easier to compare.')
    st.caption('These are estimates of vocal delivery, not facts about internal feelings. Pitch alone never determines anger. No camera or automatic spoken cues are used in this stage.')
    method=st.radio('Emotion test input',['Record microphone','Upload a clip'],horizontal=True,key='emotion_input_method')
    if method=='Record microphone':
        recording=st.audio_input('Record one complete phrase · 1–15 seconds',sample_rate=None,key='emotion_test_mic')
    else:
        recording=st.file_uploader('One phrase · 1–15 seconds',type=['wav','flac','ogg'],key='emotion_test_upload')
    raw=recording.getvalue() if recording is not None else None
    intended=st.selectbox('Intended delivery for this test',['Not labelled',*LABEL_MAP.values()],key='emotion_test_intended',
        help='Your intended delivery is a test label, not verified ground truth. It is never sent to the model.')
    compare=st.checkbox('Compare with previous four-label model',value=True,key='emotion_compare_baseline')
    st.caption('The comparison model cannot recognize surprise or fear. It does not veto those new categories. Anger requires agreement from both models in this prototype.')
    one_speaker=st.checkbox('I confirm this clip contains one clear speaker',key='emotion_single_speaker')
    with st.expander('Decision thresholds (experimental)'):
        minimum_score=st.slider('Minimum top model score',.3,.95,.7,.05,key='emotion_min_score')
        minimum_margin=st.slider('Minimum gap between top two scores',0.,.5,.15,.05,key='emotion_min_margin')
        st.caption('These are tunable abstention rules, not calibrated probabilities. We need test recordings to choose them responsibly.')
    if raw is None:
        st.info('Record or upload a short phrase to test. For the partner test, the wearer stays silent and another person speaks toward the glasses.')
        return
    digest=hashlib.sha256(raw).hexdigest()
    if st.session_state.get('emotion_lab_input')!=digest:
        st.session_state.pop('emotion_lab_result',None)
        st.session_state.emotion_lab_input=digest
    st.audio(raw)
    try:
        info=sf.info(io.BytesIO(raw))
        if not 1<=info.duration<=15:raise ValueError('Use a complete phrase between 1 and 15 seconds; record again or trim your clip.')
        audio=read_audio(raw)
        duration=len(audio)/24000
        rms=float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
        original_samples,_=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True)
        clipping=float(np.mean(np.abs(original_samples)>=.99))
    except Exception as exc:
        st.error(str(exc));return
    if st.button('Analyze vocal delivery',key='analyze_emotion',type='primary'):
        if rms<.001:
            st.warning('This recording is too quiet for a useful test. Check the microphone and record again.')
            st.session_state.pop('emotion_lab_result',None)
        else:
            try:
                with st.spinner('Loading/running local emotion2vec+ large (first load takes longer)…'):
                    model,lock=load_model()
                    started=time.perf_counter()
                    with lock:scores=model.distribution(audio)
                    elapsed=time.perf_counter()-started
                baseline=None
                if compare:
                    with st.spinner('Checking the previous local model for disagreement…'):
                        label,score=get_baseline().estimate(audio)
                        baseline=dict(label=label,score=score,model='superb/wav2vec2-base-superb-er')
                st.session_state.emotion_lab_result=dict(scores=scores,baseline=baseline,inference_s=elapsed,
                    model_load_s=model.load_s,provenance=model.provenance)
            except Exception as exc:
                st.error(f'Emotion analysis failed: {exc}')
    result=st.session_state.get('emotion_lab_result')
    if result is None:return
    baseline=result.get('baseline') if compare else None
    if baseline:st.caption(f"Comparison model: {baseline['label']} · score {baseline['score'] if baseline['score'] is not None else 'unavailable'}. Scores from different models are not directly comparable.")
    if compare and baseline is None:st.info('Run Analyze again to add the comparison model to this result.')
    decision=gate_prediction(result['scores'],duration_s=duration,rms=rms,clipping_fraction=clipping,
                             one_speaker=one_speaker,minimum_score=minimum_score,minimum_margin=minimum_margin,
                             reference_label=baseline['label'] if baseline else None,reference_score=baseline['score'] if baseline else None)
    st.metric('Vocal delivery estimate',decision['estimate'])
    st.write(f"Top alternative: **{decision['top_label']}** ({decision['top_score']:.3f}); next: **{decision['second_label']}** ({decision['second_score']:.3f}).")
    if decision['abstained']:st.warning('Withheld: '+'; '.join(decision['reasons']))
    else:st.info('The prototype gate accepted this model estimate. This does not establish that it is correct.')
    if intended!='Not labelled':
        st.write(f"Your intended test label: **{intended}**. {'Top model label matches.' if decision['top_label']==intended else 'Top model label differs — save this test as a failure case.'}")
    st.dataframe(pd.DataFrame(sorted(result['scores'].items(),key=lambda item:item[1],reverse=True),columns=['Model label','Score']),hide_index=True)
    st.caption(f"Inference {result['inference_s']:.2f}s · initial load {result['model_load_s']:.2f}s (cached) · CPU · no isolation or transcription before classification.")
    st.caption('No cue is played. First collect contrasting recordings and inspect mistakes, including false anger and uncertain cases.')
    report=dict(schema='textuplink.emotion-test.v1',source_sha256=digest,duration_s=duration,
        intended_delivery=intended,test_label_source='User intended delivery, unverified',comparison_enabled=compare,
        one_speaker_confirmed=one_speaker,input_rms=rms,clipping_fraction=clipping,
        thresholds=dict(minimum_score=minimum_score,minimum_margin=minimum_margin),
        decision=decision,**result)
    st.download_button('Save emotion test report',json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),
                       f'emotion_test_{digest[:8]}.json','application/json')
    st.download_button('Save this test audio',raw,'emotion_test_audio'+('.wav' if info.format=='WAV' else '.flac' if info.format=='FLAC' else '.ogg'),
                       'audio/wav' if info.format=='WAV' else 'application/octet-stream')

    if st.button('Add result to test log',key='log_emotion_test'):
        history=st.session_state.setdefault('emotion_test_history',[])
        history[:]=[entry for entry in history if entry['source_sha256']!=digest]
        history.append(report)
    history=st.session_state.get('emotion_test_history',[])
    if history:
        st.subheader('Session test log')
        labelled=[entry for entry in history if entry['intended_delivery']!='Not labelled']
        false_anger=sum(entry['decision']['estimate']=='Angry' and entry['intended_delivery']!='Angry' for entry in labelled)
        st.caption(f'{len(history)} distinct recordings · {len(labelled)} labelled tests · {false_anger} accepted anger estimates that differ from the intended label. Intended delivery is not verified ground truth.')
        st.dataframe(pd.DataFrame([dict(intended=entry['intended_delivery'],top_model_label=entry['decision']['top_label'],
            gated_estimate=entry['decision']['estimate'],score=entry['decision']['top_score'],
            clip=entry['source_sha256'][:8]) for entry in history]),hide_index=True)
        st.download_button('Download session test log',json.dumps(history,ensure_ascii=False,indent=2,allow_nan=False),
                           'emotion_test_log.json','application/json')
        st.caption('The log lives in this Streamlit session. Download it before reloading. Re-adding the same recording updates its entry.')
