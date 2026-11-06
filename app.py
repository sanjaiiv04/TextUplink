"""Local glasses microphone, Qwen transcription and staged vocal-emotion tests."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import io
import threading
import time
from pathlib import Path
import numpy as np
import streamlit as st
import soundfile as sf
from textuplink.transcript import load_transcriber,LocalEmotion

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='TextUplink · Social cue prototype',layout='wide')
st.title('TextUplink · Social cue prototype')
st.write('Test the glasses microphone, transcribe speech locally, and evaluate vocal-emotion estimates.')
st.caption('All inference stays on this Mac. Emotion testing is Stage 1; automatic wearer cues and speaker tracking are not implemented.')

@st.cache_resource
def emotion_model():return LocalEmotion(ROOT/'models/emotion')

@st.cache_resource
def transcription_lock():return threading.Lock()

@st.cache_resource(max_entries=2)
def live_transcriber(device):
    import torch
    started=time.perf_counter()
    model=load_transcriber(ROOT,device=device)
    loaded=time.perf_counter()
    with transcription_lock():
        model.transcribe(audio=(np.zeros(32000,dtype=np.float32),16000),language='English',return_time_stamps=False)
        if device=='mps':torch.mps.synchronize()
    return model,dict(device=device,model_load_s=loaded-started,warmup_s=time.perf_counter()-loaded)

microphone_tab,live_tab,emotion_tab=st.tabs(['Microphone test','Live transcription','Emotion lab · Stage 1'])

with emotion_tab:
    from textuplink.emotion import render_emotion_lab
    render_emotion_lab(ROOT,emotion_model)

with live_tab:
    from textuplink.live_transcription import render_live_tab
    render_live_tab(live_transcriber,transcription_lock)

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
            st.caption('Listen for clear speech, then save the WAV. Test a short phrase in Emotion lab, or use Live transcription for continuous speech.')
        except Exception as exc:
            st.error(f'Could not read the microphone recording: {exc}')
    else:
        st.caption('Nothing is recorded until you start the microphone. Stop the recording to enable playback and WAV download.')
