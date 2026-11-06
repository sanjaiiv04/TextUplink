"""Local Qwen3 live drafts with phrase context and measured processing delay."""
from collections import deque
import threading
import time
import numpy as np


class LiveTranscript:
    """Coalesce provisional snapshots; keep complete final phrase audio in order."""
    version=2
    SAMPLE_RATE=16000
    DRAFT_INTERVAL=16000
    MAX_PHRASE=160000
    PAUSE=5600

    def __init__(self,recognize,model_lock):
        self.recognize=recognize
        self.model_lock=model_lock
        self.condition=threading.Condition()
        self.pending=deque()
        self.pending_draft=None
        self.parts=[]
        self.part_samples=0
        self.captured_samples=0
        self.quiet_samples=0
        self.last_draft_samples=0
        self.phrase_id=0
        self.segments=[]
        self.draft=None
        self.last_timing=None
        self.last_final_id=-1
        self.coalesced=0
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
        try:
            if self.resampler is None:
                self.resampler=av.AudioResampler(format='fltp',layout='mono',rate=16000)
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
            final=self.part_samples>=self.MAX_PHRASE or (self.part_samples>=8000 and self.quiet_samples>=self.PAUSE)
            if final:
                self._enqueue(final=True)
            elif self.part_samples-self.last_draft_samples>=self.DRAFT_INTERVAL:
                self._enqueue(final=False)

    def _enqueue(self,final=True):
        if not self.parts:return
        y=np.concatenate(self.parts)
        samples=len(y)
        job=dict(id=self.phrase_id,start_s=(self.captured_samples-samples)/16000,
                 end_s=self.captured_samples/16000,audio=y,final=final,
                 queued_at=time.monotonic(),audio_end_received_at=self.last_audio)
        self.last_draft_samples=samples
        if len(y)>=1600 and np.sqrt(np.mean(y.astype(np.float64)**2))>=.0001:
            if final:
                if self.pending_draft is not None and self.pending_draft['id']==self.phrase_id:
                    self.pending_draft=None
                self.pending.append(job)
                if len(self.pending)>=6:
                    # Stop accepting further capture; never discard accepted final audio.
                    self.closed=True
                    self.error=(f'Qwen is falling behind. Capture stopped at {job["end_s"]:.1f}s; '
                                'finishing all accepted audio. Try CPU/GPU selection or shorter phrases.')
            else:
                if self.pending_draft is not None:self.coalesced+=1
                self.pending_draft=job
            self.condition.notify_all()
        if final:
            self.parts=[]
            self.part_samples=0
            self.quiet_samples=0
            self.last_draft_samples=0
            self.phrase_id+=1

    def finish(self):
        with self.condition:
            if not self.closed:
                self._enqueue(final=True)
                self.closed=True
                if self.thread is None:self.ended=True
                self.condition.notify_all()

    def _work(self):
        while True:
            with self.condition:
                while not self.pending and self.pending_draft is None:
                    if self.closed:
                        self.ended=True
                        return
                    self.condition.wait(timeout=1)
                    if time.monotonic()-self.last_audio>30:
                        self._enqueue(final=True)
                        self.closed=True
                if self.pending:
                    job=self.pending.popleft()
                else:
                    job=self.pending_draft
                    self.pending_draft=None
                self.processing=True
            dequeued=time.monotonic()
            try:
                with self.model_lock:
                    inference_start=time.monotonic()
                    results=self.recognize(job['audio'])
                    done=time.monotonic()
                text=str(results[0].text).strip() if results else ''
                if text=='嗯。':text=''
                language=str(results[0].language) if results else 'Unknown'
                timing=dict(queue_wait_s=dequeued-job['queued_at'],
                            model_lock_wait_s=inference_start-dequeued,
                            inference_s=done-inference_start,
                            received_to_result_s=done-job['audio_end_received_at'],
                            audio_duration_s=len(job['audio'])/16000)
                entry=dict(id=job['id'],start_s=job['start_s'],end_s=job['end_s'],text=text,
                           language=language,final=job['final'],**timing)
                with self.condition:
                    self.last_timing=timing
                    if job['final']:
                        self.last_final_id=job['id']
                        if text:
                            self.segments.append(entry)
                            self.segments=self.segments[-1000:]
                        if self.draft is not None and self.draft['id']<=job['id']:self.draft=None
                    elif job['id']>self.last_final_id and (self.draft is None or self.draft['id']<=job['id']):
                        self.draft=entry if text else None
            except Exception as exc:
                with self.condition:
                    self.error=f'Local Qwen transcription failed: {exc}'
                    if job['final']:
                        self.closed=True
                        self.pending.clear()
                        self.pending_draft=None
                        self.parts=[]
                        self.part_samples=0
                        self.error+=' Capture stopped; this phrase and remaining unprocessed audio have no transcript.'
            finally:
                with self.condition:self.processing=False

    def snapshot(self):
        with self.condition:
            return dict(segments=[dict(s) for s in self.segments],draft=dict(self.draft) if self.draft else None,
                        queued=len(self.pending)+(self.pending_draft is not None),
                        processing=self.processing,coalesced=self.coalesced,error=self.error,
                        closed=self.closed,ended=self.ended,level_db=self.level_db,
                        captured_s=self.captured_samples/16000,
                        last_timing=dict(self.last_timing) if self.last_timing else None)


def render_live_tab(get_model,get_model_lock):
    import json
    import torch
    import streamlit as st
    from streamlit_webrtc import webrtc_streamer,WebRtcMode
    from streamlit_webrtc.sink import CallbackSinkTrack

    st.subheader('Speak and transcribe with Qwen3')
    st.write('Choose the Ray-Ban microphone, press START, and speak. The transcript updates as short speech segments are processed locally.')
    st.caption('A revisable draft starts after about one second of speech plus inference time. Qwen re-reads the growing phrase; a 350 ms quiet pause or 10-second limit finalizes it. No isolation or emotion analysis runs here.')
    session=st.session_state.get('live_qwen_session')
    if session is not None and getattr(session,'version',None)!=2:
        session.finish()
        st.session_state.live_qwen_previous_session=session
        st.session_state.live_qwen_session=None
        st.session_state.live_qwen_was_playing=False
        session=None
        st.info('The live engine changed. The previous capture was stopped; prepare a new session to use the lower-delay engine.')
    previous=st.session_state.get('live_qwen_previous_session')
    if previous is not None:
        previous_text='\n\n'.join(row['text'] for row in previous.snapshot()['segments'] if row['text'].strip()!='嗯。')
        if previous_text:
            st.download_button('Save transcript from previous engine',previous_text,'previous_live_transcript.txt','text/plain')
    available_devices=['mps','cpu'] if torch.backends.mps.is_available() else ['cpu']
    active=session is not None and not session.snapshot()['ended']
    device=st.selectbox('Qwen processing device',available_devices,
        format_func=lambda value:'Apple GPU (MPS)' if value=='mps' else 'CPU',disabled=active,
        help='GPU is selected when available. Change only between sessions; preparing warms up the selected model before capture.')
    if session is None or session.snapshot()['ended']:
        if st.button('Prepare local Qwen3 microphone',key='prepare_live_qwen'):
            try:
                with st.spinner('Loading Qwen3 and warming up before microphone capture…'):
                    model,runtime=get_model(device)
            except Exception as exc:
                st.error(f'Qwen setup failed on {device}: {exc}. Select CPU and prepare again if the Apple GPU fails.')
                return
            st.session_state.live_qwen_runtime=runtime
            session=LiveTranscript(lambda audio:model.transcribe(audio=(audio,16000),language=None,return_time_stamps=False),get_model_lock())
            st.session_state.live_qwen_session=session
            st.session_state.live_qwen_sink=CallbackSinkTrack(session.receive_frame,kind='audio')
            st.session_state.live_qwen_was_playing=False
            st.session_state.live_qwen_number=st.session_state.get('live_qwen_number',0)+1
        elif session is None:
            st.info('Prepare the model first, then allow microphone access when you press START.')
            return
    if session is None:return
    runtime=st.session_state.get('live_qwen_runtime',{})
    st.caption(f"Prepared device: {runtime.get('device','unknown')} · model load {runtime.get('model_load_s',0):.2f}s · warm-up {runtime.get('warmup_s',0):.2f}s. Setup happens before capture and is cached.")
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

    @st.fragment(run_every=.25)
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
        timing=state['last_timing']
        if timing:
            timings=st.columns(3)
            timings[0].metric('Last inference',f"{timing['inference_s']:.2f} s")
            timings[1].metric('Queue + model wait',f"{timing['queue_wait_s']+timing['model_lock_wait_s']:.2f} s")
            timings[2].metric('Received audio → result',f"{timing['received_to_result_s']:.2f} s")
            st.caption('Delay starts when the last sample of that update reaches the server. Add up to 250 ms for display refresh; Bluetooth/browser capture delay is not measured here.')
        if state['processing']:
            st.info('Qwen3 is updating the transcript. Draft words may change as more of the phrase arrives.')
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
        draft=state['draft']
        if draft and draft['text']:
            st.caption('Draft — may be corrected as you continue speaking')
            st.write(draft['text'])
        if not text and not draft:
            st.write('Your words will appear here.')
        if text:
            st.download_button('Download live transcript',text,'live_transcript.txt','text/plain',key='live_transcript_text')
            st.download_button('Download timed segments',json.dumps(visible_segments,ensure_ascii=False,indent=2),
                'live_transcript_segments.json','application/json',key='live_transcript_segments')
            st.caption('Downloads include finalized phrases only. STOP finalizes the remaining draft. Segment times are not word timings; check the text against speech.')
        with st.expander('How these updates work'):
            st.write('Every one second, Qwen receives the phrase captured so far. A newer pending draft replaces an older pending draft of that same phrase; the audio remains in the final phrase. Completed phrases take priority. If six final phrases are waiting, capture stops and all accepted final audio is processed instead of silently dropping it.')
            st.caption(f"Provisional recalculations coalesced: {state['coalesced']}. Quiet detection is a level heuristic and can fail in noise. Continuous speech is finalized after 10 seconds, which can still split a word at that boundary.")
    transcript_panel()
