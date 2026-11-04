"""TextUplink: local voice isolation and expressive caption prototype."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import hashlib
import io
import json
import threading
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import soundfile as sf
from textuplink.audio import read_audio,wav_bytes
from textuplink.speech_processing import ENHANCERS,enhance_speech
from textuplink.transcript import load_transcriber,transcribe_words,LocalEmotion,annotate_context,LABELS,review_annotations
from textuplink.captions import measure_delivery,caption_player

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='TextUplink · Expressive captions',layout='wide')
st.title('TextUplink · Expressive captions')
st.write('Isolate speech, read the words, and see how they were spoken. All model inference stays on this Mac.')
st.caption('Recorded captions and live microphone transcription for a future wearable experience. Speaker tracking and automatic audio cues are not implemented.')

@st.cache_resource
def transcriber():return load_transcriber(ROOT)

@st.cache_resource
def emotion_model():return LocalEmotion(ROOT/'models/emotion')

@st.cache_data(show_spinner=False)
def process(audio,enabled,mix,backend):
    return enhance_speech(audio,reduce_noise=enabled,strength=mix,backend=backend)

@st.cache_data(show_spinner=False)
def delivery(rows,audio):return measure_delivery(rows,audio)

@st.cache_resource
def transcription_lock():return threading.Lock()

captions_tab,microphone_tab,live_tab=st.tabs(['Expressive captions','Microphone test','Live transcription'])

with live_tab:
    from textuplink.live_transcription import render_live_tab
    render_live_tab(transcriber,transcription_lock)

with microphone_tab:
    st.subheader('Record, listen, and save a WAV')
    st.write('Select your microphone in the browser, record a short clip, then stop to play or download it.')
    st.info('For the glasses: they must appear as an input microphone on the computer running your browser. Pairing with your iPhone alone does not connect its microphone to this page.')
    with st.expander('Choose the glasses microphone'):
        st.markdown('1. Connect the glasses to the Mac over Bluetooth.\n'
                    '2. In **System Settings → Sound → Input**, check whether the glasses appear and select them.\n'
                    '3. Allow microphone access for this page in your browser and select the glasses as its microphone.\n'
                    '4. Disconnect other headsets. Record with the Mac a few feet away and speak near the glasses.')
        st.caption('If the glasses are absent from the input list, this page cannot access them yet. A working recording alone does not prove the glasses were used; it may be the Mac microphone. This widget does not report the hardware input name.')
    recording=st.audio_input('Record a microphone test · aim for 10 seconds',sample_rate=None,key='microphone_test',
        help='Click the microphone to start and stop. Browser microphone permission is required. The recording stays local to this Streamlit server.')
    if recording is not None:
        try:
            recorded_wav=recording.getvalue()
            with sf.SoundFile(io.BytesIO(recorded_wav)) as audio_file:
                duration=audio_file.frames/audio_file.samplerate
                rate=audio_file.samplerate
                channels=audio_file.channels
                if audio_file.format!='WAV' or duration<=0:
                    raise ValueError('No valid WAV audio was captured. Please record again.')
                samples=audio_file.read(dtype='float32',always_2d=True)
                if not np.isfinite(samples).all():
                    raise ValueError('The recording contains invalid audio samples. Please record again.')
            peak=float(np.max(np.abs(samples)))
            rms=float(np.sqrt(np.mean(samples.astype(np.float64)**2)))
            peak_db=20*np.log10(peak) if peak>0 else None
            rms_db=20*np.log10(rms) if rms>0 else None
            levels=st.columns(2)
            levels[0].metric('Recorded peak','Silent' if peak_db is None else f'{peak_db:.1f} dBFS')
            levels[1].metric('Average recording level','Silent' if rms_db is None else f'{rms_db:.1f} dBFS')
            if peak==0:
                st.error('The WAV contains digital silence: every sample is zero. This is a capture problem, not a playback volume problem.')
            elif rms<.001:
                st.warning('The WAV contains very quiet audio. Check the browser microphone selection and the Mac input meter, then make a fresh recording.')
            else:
                st.info('The WAV contains an audio signal. Listen to confirm speech; this does not verify which microphone was used.')
            st.caption('To check capture: open Mac System Settings → Sound → Input and speak. The input meter should move. Then explicitly select the glasses in your browser microphone settings and reload this page before recording again.')
            st.audio(recorded_wav,format='audio/wav')
            st.caption(f'{duration:.1f} seconds · {rate:,} Hz · {channels} channel(s). Original browser recording; no isolation or AI processing applied.')
            st.download_button('Save microphone recording as WAV',recorded_wav,'microphone_test.wav','audio/wav',key='download_microphone_test')
            st.caption('Listen for clear speech, then save the WAV. You can upload it in Expressive captions for the next step.')
        except Exception as exc:
            st.error(f'Could not read the microphone recording: {exc}')
    else:
        st.caption('Nothing is recorded until you start the microphone. Stop the recording to enable playback and WAV download.')

with captions_tab:
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
                    with transcription_lock():
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
