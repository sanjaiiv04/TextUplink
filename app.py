"""TextUplink: local voice isolation and expressive caption prototype."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import hashlib
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from textuplink.audio import read_audio,wav_bytes
from textuplink.speech_processing import ENHANCERS,enhance_speech
from textuplink.transcript import load_transcriber,transcribe_words,LocalEmotion,annotate_context,LABELS,review_annotations
from textuplink.captions import measure_delivery,caption_player

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='TextUplink · Expressive captions',layout='wide')
st.title('TextUplink · Expressive captions')
st.write('Isolate speech, read the words, and see how they were spoken. All model inference stays on this Mac.')
st.caption('Recording prototype for a future wearable caption experience. Speaker tracking and live streaming are not implemented.')

@st.cache_resource
def transcriber():return load_transcriber(ROOT)

@st.cache_resource
def emotion_model():return LocalEmotion(ROOT/'models/emotion')

@st.cache_data(show_spinner=False)
def process(audio,enabled,mix,backend):
    return enhance_speech(audio,reduce_noise=enabled,strength=mix,backend=backend)

@st.cache_data(show_spinner=False)
def delivery(rows,audio):return measure_delivery(rows,audio)

st.subheader('1. Upload your audio')
upload=st.file_uploader('Audio · 0.1–60 seconds',type=['wav','flac','ogg'])
raw=upload.getvalue() if upload is not None else None

if raw is None:
    st.info('Upload an audio file to begin.')
else:
    try:
        captured=read_audio(raw)
        if len(captured)<2400:raise ValueError('Use at least 0.1 seconds of audio')
        st.audio(wav_bytes(captured),format='audio/wav')
        st.subheader('2. Isolate speech')
        enabled=st.checkbox('Reduce background noise',value=True)
        backend=st.selectbox('Voice isolation model',['reuse','deepfilter','mossformer','spectral'],format_func=ENHANCERS.get,disabled=not enabled)
        mix=st.slider('Enhanced audio mix',0.,1.,1.,.05,disabled=not enabled)
        st.caption('Full mix uses only the enhanced output. Lower values reintroduce the original recording. Amplification is off; complete isolation is not guaranteed.')
        with st.spinner('Isolating speech locally…'):prepared,processing=process(captured,enabled,mix,backend)
        st.audio(wav_bytes(prepared),format='audio/wav')
        st.download_button('Download isolated speech',wav_bytes(prepared),'isolated_speech.wav','audio/wav')
        identity=hashlib.sha256(raw+prepared.tobytes()+str((enabled,backend,mix)).encode()).hexdigest()
        if st.session_state.get('audio_id')!=identity:
            for key in ['rows','language','emotion_settings','revision']:st.session_state.pop(key,None)
            st.session_state.audio_id=identity
        st.subheader('3. Read expressive captions')
        prosody_source=st.radio('Audio used to measure vocal expression',['Captured voice','Isolated voice'],horizontal=True)
        st.caption('Captured audio retains the original delivery but noise can distort measurements. Isolation can remove vocal detail. Compare both. Pitch and loudness describe delivery; they do not establish emotion.')
        if st.button('Create expressive captions',type='primary'):
            with st.spinner('Transcribing and aligning words with local Qwen3…'):
                rows,language=transcribe_words(prepared,transcriber())
            st.session_state.rows=rows
            st.session_state.language=language
            st.session_state.revision=st.session_state.get('revision',0)+1
            st.session_state.pop('emotion_settings',None)
        rows=st.session_state.get('rows')
        if rows is not None:
            if not rows:st.info('No aligned words returned. Try less isolation or check the recording.')
            else:
                measured_audio=captured if prosody_source=='Captured voice' else prepared
                with st.expander('Optional emotion estimates and transcript corrections'):
                    threshold=st.slider('Minimum emotion model score',.25,.95,.6,.05)
                    if st.button('Estimate phrase emotion locally'):
                        with st.spinner('Estimating phrase-context emotion…'):
                            rows=annotate_context(rows,measured_audio,emotion_model(),threshold)
                        st.session_state.rows=rows
                        st.session_state.revision=st.session_state.get('revision',0)+1
                        st.session_state.emotion_settings=dict(audio=prosody_source,threshold=threshold)
                    st.caption('English baseline: happy, angry, sad, neutral. Predictions describe short contexts and are shared across their words; they are not independent word-level emotions. Scores are not calibrated certainty.')
                    if st.session_state.get('emotion_settings'):
                        saved=st.session_state.emotion_settings
                        st.caption(f"Stored estimates: {saved['audio']}, score threshold {saved['threshold']:.2f}. Run estimation again to apply different settings.")
                    frame=pd.DataFrame(rows)
                    edited=st.data_editor(frame,hide_index=True,key=f"review_{identity}_{st.session_state.get('revision',0)}",
                        disabled=[c for c in frame.columns if c not in ['word','emotion']],
                        column_config={'emotion':st.column_config.SelectboxColumn('Emotion',options=LABELS,required=True)})
                    reviewed=review_annotations(edited.to_dict('records'),rows)
                    st.caption('Editing words does not realign their times. Human emotion corrections remain identified in exports.')
                colors=st.checkbox('Color words by emotion annotation',value=False)
                if colors:st.caption('Green: happy · orange: angry · blue: sad · slate: neutral · gray: uncertain or unlabelled. Playback highlight is separate from emotion color.')
                with st.spinner('Measuring pitch and emphasis…'):expressive=delivery(reviewed,measured_audio)
                st.iframe(caption_player(expressive,measured_audio,colors),height=440,alt="Expressive captions with synchronized audio playback")
                st.caption(f"Language: {st.session_state.language}. Relative pitch and loudness use this recording’s aligned speech, not a known speaker baseline. No speaker identities are inferred.")
                payload=dict(schema='textuplink.captions.v1',language=st.session_state.language,
                    asr='Qwen/Qwen3-ASR-0.6B',alignment='Qwen/Qwen3-ForcedAligner-0.6B',
                    source_sha256=hashlib.sha256(raw).hexdigest(),processing=processing,
                    prosody_audio=prosody_source,emotion_settings=st.session_state.get('emotion_settings'),words=expressive)
                st.download_button('Download caption data',json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),'expressive_captions.json','application/json')
                st.download_button('Download caption player',caption_player(expressive,measured_audio,colors),'expressive_captions.html','text/html')
                st.caption('The HTML export includes this recording for synchronized offline playback. Check words against the audio; noise and transcription errors can affect the display.')
    except Exception as exc:
        st.error(f'Could not complete this step: {exc}')
