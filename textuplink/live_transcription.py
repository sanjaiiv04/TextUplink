"""Continuous browser capture with local, short-batch Qwen3 transcription."""
from collections import deque
import threading
import time
import numpy as np


class LiveTranscript:
    """Session-local bounded audio queue; inference never blocks audio capture."""
    def __init__(self,recognize,model_lock):
        self.recognize=recognize
        self.model_lock=model_lock
        self.condition=threading.Condition()
        self.pending=deque()
        self.parts=[]
        self.part_samples=0
        self.captured_samples=0
        self.quiet_samples=0
        self.segments=[]
        self.dropped=0
        self.error=None
        self.processing=False
        self.closed=False
        self.ended=False
        self.thread=None
        self.last_audio=time.monotonic()
        self.resampler=None
        self.level_db=None

    def receive_frame(self,frame):
        import av
        if self.resampler is None:
            self.resampler=av.AudioResampler(format='fltp',layout='mono',rate=16000)
        try:
            for converted in self.resampler.resample(frame):
                self.push(converted.to_ndarray().reshape(-1))
        except Exception as exc:
            with self.condition:self.error=f'Microphone conversion failed: {exc}'

    def push(self,audio):
        y=np.asarray(audio,dtype=np.float32).reshape(-1)
        if not len(y) or not np.isfinite(y).all():return
        rms=float(np.sqrt(np.mean(y.astype(np.float64)**2)))
        with self.condition:
            if self.closed:return
            self.last_audio=time.monotonic()
            self.level_db=20*np.log10(max(rms,1e-8))
            if self.thread is None:
                self.thread=threading.Thread(target=self._work,daemon=True,name='local-qwen-live')
                self.thread.start()
            self.parts.append(y.copy())
            self.part_samples+=len(y)
            self.captured_samples+=len(y)
            self.quiet_samples=self.quiet_samples+len(y) if rms<.003 else 0
            if self.part_samples>=80000 or (self.part_samples>=16000 and self.quiet_samples>=8000):
                self._enqueue()

    def _enqueue(self):
        if not self.parts:return
        y=np.concatenate(self.parts)
        start=(self.captured_samples-len(y))/16000
        end=self.captured_samples/16000
        self.parts=[]
        self.part_samples=0
        self.quiet_samples=0
        if len(y)<1600 or np.sqrt(np.mean(y.astype(np.float64)**2))<.0001:return
        # At most six waiting segments; flag discarded audio rather than hiding gaps.
        if len(self.pending)>=6:
            self.pending.popleft()
            self.dropped+=1
        self.pending.append((start,end,y))
        self.condition.notify_all()

    def finish(self):
        with self.condition:
            if not self.closed:
                self._enqueue()
                self.closed=True
                if self.thread is None:self.ended=True
                self.condition.notify_all()

    def _work(self):
        while True:
            with self.condition:
                while not self.pending:
                    if self.closed:
                        self.ended=True
                        return
                    self.condition.wait(timeout=1)
                    if time.monotonic()-self.last_audio>30:
                        self._enqueue()
                        self.closed=True
                start,end,audio=self.pending.popleft()
                self.processing=True
            began=time.monotonic()
            try:
                with self.model_lock:
                    results=self.recognize(audio)
                text=str(results[0].text).strip() if results else ''
                if text == '嗯。':
                    text = ''
                language=str(results[0].language) if results else 'Unknown'
                with self.condition:
                    if text:
                        self.segments.append(dict(start_s=start,end_s=end,text=text,language=language,
                            processing_s=time.monotonic()-began))
                        self.segments=self.segments[-1000:]
            except Exception as exc:
                with self.condition:self.error=f'Local Qwen transcription failed: {exc}'
            finally:
                with self.condition:self.processing=False

    def snapshot(self):
        with self.condition:
            return dict(segments=[dict(s) for s in self.segments],queued=len(self.pending),
                        processing=self.processing,dropped=self.dropped,error=self.error,
                        closed=self.closed,ended=self.ended,level_db=self.level_db,
                        captured_s=self.captured_samples/16000)


def render_live_tab(get_model,get_model_lock):
    import json
    import streamlit as st
    from streamlit_webrtc import webrtc_streamer,WebRtcMode
    from streamlit_webrtc.sink import CallbackSinkTrack

    st.subheader('Speak and transcribe with Qwen3')
    st.write('Choose the Ray-Ban microphone, press START, and speak. The transcript updates as short speech segments are processed locally.')
    st.caption('Updates arrive after a pause or about five seconds of speech, plus model processing time. This is chunked transcription, not word-by-word native streaming. No isolation, emotion analysis or word alignment runs in this tab.')
    session=st.session_state.get('live_qwen_session')
    if session is None or session.snapshot()['ended']:
        if st.button('Prepare local Qwen3 microphone',key='prepare_live_qwen'):
            with st.spinner('Loading local Qwen3-ASR…'):
                model=get_model()
            session=LiveTranscript(lambda audio:model.transcribe(audio=(audio,16000),language=None,return_time_stamps=False),get_model_lock())
            st.session_state.live_qwen_session=session
            st.session_state.live_qwen_sink=CallbackSinkTrack(session.receive_frame,kind='audio')
            st.session_state.live_qwen_was_playing=False
            st.session_state.live_qwen_number=st.session_state.get('live_qwen_number',0)+1
        elif session is None:
            st.info('Prepare the model first, then allow microphone access when you press START.')
            return
    if session is None:return
    snap=session.snapshot()
    context=webrtc_streamer(key=f"live-qwen-{st.session_state.get('live_qwen_number',0)}",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration={'iceServers':[]},
        server_rtc_configuration={'iceServers':[]},
        media_stream_constraints={'video':False,'audio':{'echoCancellation':False,'noiseSuppression':False,'autoGainControl':False}},
        sink_audio_track=st.session_state.live_qwen_sink,
        desired_playing_state=False if snap['closed'] else None,
        sendback_audio=False)
    st.caption('Use localhost on this Mac. Select RB Meta in the microphone device menu and Brave settings. Camera capture and audio echo are disabled. No external STUN/TURN relay is configured.')
    if context.state.playing:
        st.session_state.live_qwen_was_playing=True
    elif st.session_state.get('live_qwen_was_playing'):
        session.finish()
        st.session_state.live_qwen_was_playing=False

    @st.fragment(run_every=1)
    def transcript_panel():
        state=session.snapshot()
        if (state['ended'] and not snap['ended']) or (state['closed'] and context.state.playing and not snap['closed']):
            st.rerun(scope='app')
        if st.session_state.get('live_qwen_was_playing') and not context.state.playing:
            session.finish()
            st.session_state.live_qwen_was_playing=False
            state=session.snapshot()
        metrics=st.columns(3)
        metrics[0].metric('Audio received',f"{state['captured_s']:.1f} s")
        metrics[1].metric('Microphone level','Waiting' if state['level_db'] is None else f"{state['level_db']:.1f} dBFS")
        metrics[2].metric('Segments waiting',state['queued'])
        if state['error']:st.error(state['error'])
        if state['dropped']:st.warning(f"Qwen could not keep up: {state['dropped']} queued audio segment(s) were discarded. The transcript has gaps.")
        if state['processing']:
            st.info('Qwen3 is transcribing. Microphone capture continues while the model processes audio.')
        elif state['closed'] and not state['ended']:
            st.info('Microphone stopped. Finishing the remaining speech…')
        elif state['ended']:
            st.success('Transcription finished. Use Prepare local Qwen3 microphone to start a new session.')
        elif not state['segments']:
            st.info('Waiting for speech. Check that the microphone level changes when you talk.')
        visible_segments=[segment for segment in state['segments'] if segment['text'].strip() != '嗯。']
        text='\n\n'.join(segment['text'] for segment in visible_segments)
        st.subheader('Transcript')
        if text:
            st.write(text)
            st.download_button('Download live transcript',text,'live_transcript.txt','text/plain',key='live_transcript_text')
            st.download_button('Download timed segments',json.dumps(visible_segments,ensure_ascii=False,indent=2),
                'live_transcript_segments.json','application/json',key='live_transcript_segments')
            st.caption('Times mark captured audio segments, not individual words. Chunk boundaries can split words; verify the text against speech.')
        else:
            st.write('Your words will appear here.')
    transcript_panel()
