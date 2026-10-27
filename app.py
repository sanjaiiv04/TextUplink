"""TextUplink: local low-bandwidth speech reconstruction experiment."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st
from textuplink.codec_packet import unpack_codes
from textuplink.neural_codec import LocalCodec, MODEL_DIRS, read_audio, wav_bytes, add_noise, compare_audio
from textuplink.speech_processing import ENHANCERS

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='TextUplink · Speech Codec Lab', layout='wide')
st.title('TextUplink · Speech Codec Lab')
st.caption('Speech → compact acoustic tokens → reconstructed voice. All model inference stays on this device.')
st.write('Captured voice → reduce noise → encode acoustic detail → small packet → decode voice. A transcript with reviewable emotion colors runs alongside this audio path.')


@st.cache_resource(show_spinner=False, max_entries=3)
def get_codec(family):
    return LocalCodec(family, ROOT/MODEL_DIRS[family])


@st.cache_data(show_spinner=False)
def prepare_speech(audio,reduce,strength,boost_db,steady_noise,backend):
    from textuplink.speech_processing import enhance_speech
    return enhance_speech(audio,reduce_noise=reduce,strength=strength,boost_db=boost_db,steady_noise=steady_noise,backend=backend)


@st.cache_resource(show_spinner=False)
def get_transcriber():
    from textuplink.transcript import load_transcriber
    return load_transcriber(ROOT)


@st.cache_resource(show_spinner=False)
def get_emotion_model():
    from textuplink.transcript import LocalEmotion
    return LocalEmotion(ROOT/'models/emotion')


def show_metrics(metrics):
    cols = st.columns(3)
    cols[0].metric('Waveform SNR', f"{metrics['snr_db']:.2f} dB")
    cols[1].metric('Log-spectral distance', f"{metrics['log_spectral_distance_db']:.2f} dB")
    corr = metrics['energy_contour_correlation']
    cols[2].metric('Energy contour correlation', 'N/A' if corr is None else f'{corr:.3f}')
    st.caption('These are descriptive, sample-aligned signal measurements. They are not emotion scores. SNR is sensitive to codec delay; no automatic time alignment is applied.')


sender, receiver, protocol = st.tabs(['Sender & experiment', 'Receiver', 'Evaluation guide'])
with sender:
    st.subheader('1. Captured voice in noise')
    source = st.selectbox('Recording', ['Upload a recording', 'Included clean sample', 'Included noisy sample'])
    uploaded = st.file_uploader('Source audio · maximum 60 seconds', type=['wav','flac','ogg'], key='source') if source == 'Upload a recording' else None
    raw = uploaded.getvalue() if uploaded else None
    if source != 'Upload a recording':
        raw = (ROOT/'samples'/('audio-without-noise.wav' if source == 'Included clean sample' else 'audio-with-noise.wav')).read_bytes()
    if raw is None:
        st.info('Upload your noisy file or select Included noisy sample. You can decode a packet in Receiver without the source recording.')
    else:
        try:
            original = read_audio(raw)
            if len(original)<2400:
                raise ValueError('Use at least 0.1 seconds of audio')
            noisy = st.checkbox('Add controlled white noise', value=False)
            snr = st.slider('Added noise SNR (dB)', -5, 30, 10, disabled=not noisy)
            ground_truth, captured = add_noise(original,snr) if noisy else (None,original)
            reference_digest='synthetic' if noisy else 'none'
            if not noisy:
                with st.expander('Optional: provide the matching clean speech for evaluation'):
                    reference_choice=st.selectbox('Clean reference',['None','Upload clean reference','Included clean sample'])
                    reference_file=st.file_uploader('Aligned clean reference',type=['wav','flac','ogg'],key='clean_reference') if reference_choice=='Upload clean reference' else None
                    reference_raw=reference_file.getvalue() if reference_file else None
                    if reference_choice=='Included clean sample':
                        reference_raw=(ROOT/'samples/audio-without-noise.wav').read_bytes()
                    confirmed=st.checkbox('I confirm the reference contains the same speech, aligned in time and level',disabled=reference_raw is None)
                    if reference_raw is not None and confirmed:
                        candidate=read_audio(reference_raw)
                        if len(candidate)!=len(captured):
                            st.warning('Reference duration differs. Align the recordings before using them for ground-truth metrics; this reference will not be used.')
                        else:
                            ground_truth=candidate
                            reference_digest=hashlib.sha256(reference_raw).hexdigest()
            st.audio(wav_bytes(captured),format='audio/wav')
            st.caption(f'Captured recording · {len(captured)/24000:.2f} s · 24 kHz mono. For your already noisy file, leave added noise off.')

            st.subheader('2. Reduce background noise and raise the voice level')
            denoise=st.checkbox('Apply noise reduction before encoding',value=True)
            enhancer=st.selectbox('Speech enhancement model',['deepfilter','mossformer','spectral'],format_func=lambda key:ENHANCERS[key],disabled=not denoise)
            if enhancer=='spectral':
                strength=st.slider('Noise reduction strength',0.,1.,.5,.05,disabled=not denoise)
                noise_mode=st.selectbox('Background noise type',['Changing background noise','Steady hiss / white noise'],disabled=not denoise)
            else:
                strength=st.slider('Enhanced audio mix',0.,1.,1.,.05,disabled=not denoise,
                    help='1 uses the neural model output; lower values blend back the original recording. This is a mix, not calibrated noise attenuation.')
                noise_mode='Neural enhancement'
                st.caption('Pretrained neural enhancement runs locally at 48 kHz using the same Python environment, then returns to the 24 kHz pipeline. No silent fallback to spectral gating.')
            boost=st.checkbox('Boost voice level after noise reduction',value=True)
            boost_db=st.slider('Requested voice boost (dB)',0.,24.,9.,1.,disabled=not boost)
            with st.spinner('Reducing noise and raising speech level…'):
                started=time.perf_counter()
                prepared,processing=prepare_speech(captured,denoise,strength,boost_db if boost else 0.,noise_mode=='Steady hiss / white noise',enhancer)
                preprocessing_s=time.perf_counter()-started
            st.audio(wav_bytes(prepared),format='audio/wav')
            level_columns=st.columns(3)
            level_columns[0].metric('Applied voice boost',f"{processing['applied_gain_db']:.1f} dB")
            level_columns[1].metric('Level after noise reduction',f"{processing['cleaned_rms_dbfs']:.1f} dBFS")
            level_columns[2].metric('Final processed level',f"{processing['output_rms_dbfs']:.1f} dBFS")
            if processing['peak_limited']:
                st.info('Applied gain was limited to keep peaks below −1 dBFS. Lower the requested boost if you want less amplification.')
            st.download_button('Download processed speech',wav_bytes(prepared),'processed_speech.wav','audio/wav')
            st.caption('Noise suppression comes first, then one uniform gain raises the remaining voice and any residual noise. Peaks stay below −1 dBFS without dynamic compression. Relative emphasis and timing are preserved by the gain stage. This is noise suppression, not perfect speaker isolation.')

            enhancement_id=hashlib.sha256(raw+captured.tobytes()).hexdigest()+reference_digest
            if st.session_state.get('enhancement_compare_id')!=enhancement_id:
                st.session_state.pop('enhancement_comparison',None)
                st.session_state.enhancement_compare_id=enhancement_id
            if st.button('Compare enhancement models (boost off)'):
                candidates=[]
                for backend in ['spectral','deepfilter','mossformer']:
                    try:
                        with st.spinner(f'Comparing {ENHANCERS[backend]}…'):
                            start=time.perf_counter()
                            signal,details=prepare_speech(captured,True,.5 if backend=='spectral' else 1.,0.,False,backend)
                            elapsed=time.perf_counter()-start
                            measured=compare_audio(ground_truth,signal) if ground_truth is not None else None
                            candidates.append(dict(model=ENHANCERS[backend],status='Completed',seconds=elapsed,
                                wav=wav_bytes(signal),processing=details,source_sha256=hashlib.sha256(raw).hexdigest(),clean_reference_kind=reference_digest,clean_reference=measured))
                    except Exception as exc:
                        candidates.append(dict(model=ENHANCERS[backend],status=f'Failed: {exc}'))
                st.session_state.enhancement_comparison=candidates
            comparison=st.session_state.get('enhancement_comparison')
            if comparison:
                st.caption('Same captured input, boost off. Spectral baseline strength 0.5; neural outputs at full mix. Times include worker/model startup unless cached. Listen for changes to words and delivery.')
                for candidate in comparison:
                    st.write(candidate['model'])
                    if candidate['status']=='Completed':
                        st.audio(candidate['wav'],format='audio/wav')
                        if candidate['clean_reference']:
                            show_metrics(candidate['clean_reference'])
                    else:
                        st.warning(candidate['status'])
                summary=[{k:v for k,v in item.items() if k!='wav'} for item in comparison]
                st.download_button('Download enhancement comparison report',json.dumps(summary,indent=2,allow_nan=False),'enhancement_comparison.json','application/json')

            audio_id=hashlib.sha256(raw+captured.tobytes()).hexdigest()+str((denoise,strength if denoise else None,boost_db if boost else 0.,noise_mode if denoise else None,enhancer if denoise else None))
            if st.session_state.get('pipeline_audio_id')!=audio_id:
                for key in ['transcript_rows','transcript_language','transcript_revision','emotion_settings']:
                    st.session_state.pop(key,None)
                st.session_state.pipeline_audio_id=audio_id
            st.subheader('3. Optional transcript and emotion colors')
            st.write('The words help you read and review the speech. The audio continues directly to the codec in stage 4; it is not regenerated from this transcript.')
            st.caption('Transcription can invent or miss words in noise. Check the words against playback before treating them as accurate.')
            if st.button('Transcribe processed speech locally'):
                try:
                    from textuplink.transcript import transcribe_words
                    with st.spinner('Transcribing with local Qwen3-ASR and aligning word times…'):
                        rows,language=transcribe_words(prepared,get_transcriber())
                    st.session_state.transcript_rows=rows
                    st.session_state.transcript_language=language
                    st.session_state.transcript_revision=st.session_state.get('transcript_revision',0)+1
                    st.session_state.pop('emotion_settings',None)
                except Exception as exc:
                    st.error(f'Transcription failed: {exc}. Provision Qwen with python scripts/provision_models.py qwen and python scripts/provision_models.py aligner.')
            rows=st.session_state.get('transcript_rows')
            if rows is not None:
                if not rows:
                    st.info('No words returned. Listen to the processed audio and reduce denoising strength if speech was lost.')
                else:
                    from textuplink.transcript import LABELS, LocalEmotion, review_annotations, colored_html
                    emotion_input=st.radio('Voice used for emotion estimates',['Captured voice','Processed voice'],horizontal=True)
                    threshold=st.slider('Minimum model score for a colored estimate',.25,.95,.6,.05)
                    if st.button('Estimate phrase emotion locally'):
                        try:
                            from textuplink.transcript import annotate_context
                            with st.spinner('Estimating emotion in short speech contexts…'):
                                rows=annotate_context(rows,captured if emotion_input=='Captured voice' else prepared,get_emotion_model(),threshold)
                            st.session_state.transcript_rows=rows
                            st.session_state.transcript_revision=st.session_state.get('transcript_revision',0)+1
                            st.session_state.emotion_settings=dict(input=emotion_input,threshold=threshold)
                        except Exception as exc:
                            st.error(f'Emotion estimates unavailable: {exc}')
                    st.caption('English utterance-emotion baseline with four categories; not validated for emergency/radio speech. Orange: angry · green: happy · blue: sad · slate: neutral · gray: uncertain/unlabelled. Automatic colors are phrase-context predictions assigned to words by timing, not independent word-level emotion measurements. Model scores are not calibrated probabilities of true emotion.')
                    saved=st.session_state.get('emotion_settings')
                    if saved:
                        st.caption(f"Displayed model estimates: {saved['input']}, threshold {saved['threshold']:.2f}. Click Estimate again to apply changed settings.")
                    frame=pd.DataFrame(rows)
                    edited=st.data_editor(frame,key=f"transcript_editor_{audio_id}_{st.session_state.get('transcript_revision',0)}",hide_index=True,
                        disabled=[c for c in frame.columns if c not in ['word','emotion']],
                        column_config={'emotion':st.column_config.SelectboxColumn('Emotion annotation',options=LABELS,required=True)})
                    reviewed=review_annotations(edited.to_dict('records'),rows)
                    st.markdown(colored_html(reviewed),unsafe_allow_html=True)
                    st.caption('Hover over words for timing and annotation source. You can correct words and assign emotion labels manually. Corrections do not realign word times. Colors annotate the display; they do not control the decoder.')
                    sidecar=json.dumps(dict(schema='textuplink.transcript.v1',asr_model='Qwen/Qwen3-ASR-0.6B',alignment_model='Qwen/Qwen3-ForcedAligner-0.6B',source_sha256=hashlib.sha256(raw).hexdigest(),
                        processed_audio_sha256=hashlib.sha256(prepared.tobytes()).hexdigest(),noise_snr_db=snr if noisy else None,
                        language=st.session_state.transcript_language,denoised=denoise,noise_reduction_strength=strength if denoise else None,
                        processing=processing,emotion_settings=saved,words=reviewed),ensure_ascii=False,indent=2,allow_nan=False)
                    st.download_button('Download annotated transcript JSON',sidecar,'annotated_transcript.json','application/json')
                    st.caption(f'Transcript sidecar: {len(sidecar.encode()):,} bytes. It is separate from the audio packet; transmitting both costs the sum of their sizes.')

            st.subheader('4. Encode processed voice into a small packet')
            family = st.selectbox('Neural codec', ['encodec','mimi','focalcodec'], format_func=lambda f:{'encodec':'Meta EnCodec · 24 kHz','mimi':'Kyutai Mimi · 24 kHz','focalcodec':'FocalCodec · 16 kHz / 0.65 kb/s'}[f])
            if family == 'encodec':
                setting = st.select_slider('Codec target bitrate (kb/s)', options=[1.5,3.,6.,12.,24.],value=3.)
                nominal = setting
            elif family=='mimi':
                setting = st.select_slider('Mimi codebooks', options=[4,8,16,32],value=8)
                nominal = setting*12.5*11/1000
                st.caption(f'Nominal code rate: {nominal:.3f} kb/s before packet overhead.')
            else:
                setting=.65
                nominal=.65
                st.caption('FocalCodec uses 13-bit tokens at 50 Hz. This checkpoint is fixed at nominal 0.65 kb/s. Audio is resampled to 16 kHz internally; the WAV is returned at 24 kHz for comparison, without restoring lost high frequencies.')
            st.write('The encoder compresses the processed waveform, including the acoustic cues to delivery. No emotion label is required by the codec. Lower rates can lose speech and expression.')
            identity=audio_id+str((family,setting,reference_digest))
            if st.session_state.get('experiment_id')!=identity:
                st.session_state.pop('experiment',None)
                st.session_state.pop('rate_sweep',None)
                st.session_state.pop('codec_comparison',None)
                st.session_state.experiment_id=identity
            if st.button('Encode packet → reconstruct voice',type='primary'):
                try:
                    with st.spinner('Encoding and decoding locally…'):
                        model=get_codec(family)
                        packet,encode_s=model.encode(prepared,setting)
                        reconstructed,decode_s=model.decode(packet)
                        seconds=len(captured)/24000
                        header,_=unpack_codes(packet)
                        report=dict(codec=family,setting=setting,model_fingerprint=header['fingerprint'],source_sha256=hashlib.sha256(raw).hexdigest(),
                            duration_s=seconds,noise_snr_db=snr if noisy else None,denoised=denoise,noise_reduction_strength=strength if denoise else None,
                            packet_bytes=len(packet),packet_kbps=len(packet)*8/seconds/1000,
                            nominal_kbps=nominal,pcm16_kbps=384.,compression_vs_pcm16=(len(captured)*2)/len(packet),
                            processing=processing,preprocessing_s=preprocessing_s,encode_s=encode_s,decode_s=decode_s,
                            realtime_factor=(encode_s+decode_s)/seconds,
                            captured_reference=compare_audio(captured,reconstructed),
                            codec_input_reference=compare_audio(prepared,reconstructed),
                            clean_reference=compare_audio(ground_truth,reconstructed) if ground_truth is not None else None,
                            clean_reference_kind=reference_digest,
                            caveats='Offline batch file experiment; not streaming latency. No encryption or transport overhead. No automatic delay correction. No validated emotion score. Transcription and colors are a separate sidecar, excluded from audio packet bitrate.')
                        st.session_state.experiment=dict(packet=packet,wav=wav_bytes(reconstructed),report=report)
                except Exception as exc:
                    st.error(f'Codec experiment failed: {exc}')
                    st.info(f'Provision {family} locally first: python scripts/provision_models.py {family}.')
            if st.button('Compare bitrate settings'):
                try:
                    with st.spinner('Comparing codec settings on the same processed speech…'):
                        model=get_codec(family)
                        sweep=[]
                        for value in ([1.5,3.,6.,12.,24.] if family=='encodec' else ([.65] if family=='focalcodec' else [4,8,16,32])):
                            packet,enc=model.encode(prepared,value)
                            reconstructed,dec=model.decode(packet)
                            reference=ground_truth if ground_truth is not None else prepared
                            measured=compare_audio(reference,reconstructed)
                            sweep.append({'Setting':value,'Actual kb/s':len(packet)*8/(len(captured)/24000)/1000,
                                'Packet bytes':len(packet),'Encode s':enc,'Decode s':dec,
                                'Reference':'Clean ground truth' if ground_truth is not None else 'Processed codec input',
                                'Log-spectral distance (dB)':measured['log_spectral_distance_db'],
                                'Energy contour correlation':measured['energy_contour_correlation']})
                        st.session_state.rate_sweep=sweep
                except Exception as exc:
                    st.error(f'Bitrate comparison failed: {exc}')
            if st.button('Compare codecs on this processed speech'):
                outcomes=[]
                for backend,value in [('encodec',3.),('mimi',8),('focalcodec',.65)]:
                    try:
                        with st.spinner(f'Running {backend} on the same processed speech…'):
                            model=get_codec(backend)
                            packet,enc=model.encode(prepared,value)
                            signal,dec=model.decode(packet)
                            measured=compare_audio(ground_truth if ground_truth is not None else prepared,signal)
                            outcomes.append(dict(codec=backend,setting=value,status='Completed',packet_kbps=8*len(packet)/(len(prepared)/24000)/1000,
                                packet_bytes=len(packet),encode_s=enc,decode_s=dec,metrics=measured,
                                reference='Clean ground truth' if ground_truth is not None else 'Processed codec input',
                                source_sha256=hashlib.sha256(raw).hexdigest(),processing=processing,
                                model_fingerprint=unpack_codes(packet)[0]['fingerprint'],wav=wav_bytes(signal),packet=packet))
                    except Exception as exc:
                        outcomes.append(dict(codec=backend,status=f'Failed: {exc}'))
                st.session_state.codec_comparison=outcomes
            outcomes=st.session_state.get('codec_comparison')
            if outcomes:
                st.caption('These settings have different bitrates. Compare actual size and listening quality; the app does not rank emotional fidelity. FocalCodec is band-limited to 16 kHz sampling.')
                for item in outcomes:
                    st.write(item['codec'])
                    if item['status']=='Completed':
                        st.audio(item['wav'],format='audio/wav')
                        st.caption(f"{item['packet_kbps']:.3f} kb/s · {item['packet_bytes']:,} bytes")
                        st.download_button(f"Download {item['codec']} packet",item['packet'],f"{item['codec']}.tuplink",'application/octet-stream')
                    else:
                        st.warning(item['status'])
                summary=[{k:v for k,v in item.items() if k not in ['wav','packet']} for item in outcomes]
                st.download_button('Download codec comparison report',json.dumps(summary,indent=2,allow_nan=False),'codec_comparison.json','application/json')
            sweep=st.session_state.get('rate_sweep')
            if sweep:
                frame=pd.DataFrame(sweep)
                st.scatter_chart(frame,x='Actual kb/s',y='Log-spectral distance (dB)')
                st.dataframe(frame,hide_index=True)
                st.caption('Lower spectral distance is closer under this metric, not proof of preserved emotion. Select individual settings to listen.')
                st.download_button('Download bitrate comparison',frame.to_csv(index=False),'bitrate_comparison.csv','text/csv')
            st.subheader('5. Decode packet → reconstructed voice')
            result=st.session_state.get('experiment')
            if result:
                report=result['report']
                st.audio(result['wav'],format='audio/wav')
                cols=st.columns(3)
                cols[0].metric('Actual audio packet bitrate',f"{report['packet_kbps']:.3f} kb/s")
                cols[1].metric('Audio packet size',f"{report['packet_bytes']:,} bytes")
                cols[2].metric('Compression vs 24 kHz mono PCM16',f"{report['compression_vs_pcm16']:.1f}×")
                st.caption(f"Encode {report['encode_s']:.2f}s · decode {report['decode_s']:.2f}s · combined real-time factor {report['realtime_factor']:.2f}. Transcript sidecar and network/security overhead are excluded.")
                st.write('Compared with processed speech: isolates what compression changed.')
                show_metrics(report['codec_input_reference'])
                if report['clean_reference']:
                    st.write('Compared with clean ground truth: evaluates the complete noise-reduction and compression pipeline.')
                    st.audio(wav_bytes(ground_truth),format='audio/wav')
                    show_metrics(report['clean_reference'])
                else:
                    st.info('Add an aligned clean reference above to measure closeness to clean speech. Without it, we can only compare against the captured or processed recording.')
                st.download_button('Download sender packet',result['packet'],'speech.tuplink','application/octet-stream')
                st.download_button('Download reconstructed WAV',result['wav'],'reconstructed.wav','audio/wav')
                st.download_button('Download experiment report',json.dumps(report,indent=2,allow_nan=False),'experiment.json','application/json')
            else:
                st.info('Click Encode packet → reconstruct voice in stage 4. Then compare this output with stages 1 and 2 and your clean reference.')
        except Exception as exc:
            st.error(f'Could not prepare source audio: {exc}')
with receiver:
    st.subheader('Reconstruct from a sender packet')
    st.write('The receiver only needs the packet and the matching preinstalled model. It does not need the source recording or a transcript.')
    incoming=st.file_uploader('Upload .tuplink packet',type=['tuplink'],key='receiver_packet')
    if incoming is not None:
        packet=incoming.getvalue()
        digest=hashlib.sha256(packet).hexdigest()
        if st.session_state.get('receiver_id') != digest:
            st.session_state.pop('receiver_wav',None)
            st.session_state.receiver_id=digest
        try:
            header,_=unpack_codes(packet)
            st.write(f"Codec: {header['family']} · duration: {header['samples']/24000:.2f}s · packet: {len(packet):,} bytes")
            if st.button('Decode received packet'):
                with st.spinner('Reconstructing locally…'):
                    audio,elapsed=get_codec(header['family']).decode(packet)
                    st.session_state.receiver_wav=wav_bytes(audio)
                    st.success(f'Decoded in {elapsed:.2f}s')
            if 'receiver_wav' in st.session_state:
                st.audio(st.session_state.receiver_wav,format='audio/wav')
                st.download_button('Save receiver WAV',st.session_state.receiver_wav,'received.wav','audio/wav')
        except Exception as exc:
            st.error(f'Cannot decode packet: {exc}')
    st.caption('Packets are currently unencrypted. Integrity checks detect accidental payload corruption; they do not authenticate the sender. This tab simulates the transport boundary using a file.')
with protocol:
    st.subheader('Test the premise, not just transcription')
    st.markdown('''1. Record the same words with different delivery: calm, urgent, questioning, whispered and shouted.
2. Keep clean recordings as ground truth. Add controlled noise to create aligned test inputs.
3. Compare codec rates on identical inputs, with and without noise reduction.
4. Ask listeners to rate intelligibility, speaker similarity and expressive similarity without seeing the codec setting.
5. Report actual packet bitrate, encode/decode time and failures alongside listener scores.''')
    st.write('Text is a compact content baseline, but it omits much of the original delivery. Acoustic tokens provide a tunable tradeoff between bandwidth and reconstruction fidelity. Neither codec guarantees that emotion is preserved.')
    st.write('Start with pretrained encoder/quantizer/decoder systems. Train or fine-tune only after the benchmark reveals specific failures. A standalone vocoder still needs an encoder and a compact conditioning representation.')
    st.write('AudioCraft offers EnCodec training and MultiBand Diffusion decoders for compatible EnCodec tokens. These are follow-up experiments; this app uses the Transformers EnCodec and Mimi implementations.')
    st.caption('Codecs use local CPU batch inference; neural enhancement uses Apple Silicon Metal. Maximum 60 seconds. WAV/FLAC/OGG supported by libsndfile. Stereo is downmixed. No real-time radio, packet-loss recovery, encryption or training loop is implemented. The optional emotion classifier is an exploratory phrase-level baseline, not a validated measure of emotional state.')
