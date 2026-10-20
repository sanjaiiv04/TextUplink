import io
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import matplotlib.pyplot as plt
import importlib.util

st.set_page_config(page_title="Local Audio Triage — Phase 1", layout="wide")
st.title("Local Audio Triage — Phase 1")
st.caption("Local acoustic inspection prototype. Features describe the signal; they do not detect stress, emotion, or intent.")

uploaded = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])
if uploaded is None:
    st.info("Upload a recording to inspect its waveform, energy, pitch, and spectral features.")
    st.stop()

@st.cache_data(show_spinner=False)
def analyze_audio(file_bytes: bytes, filename: str):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True)
    if y.size == 0:
        raise ValueError("The uploaded audio appears to be empty.")
    duration = len(y) / sr
    # Short-time features. Keep FFT window suitable for narrow-band 8 kHz radio recordings.
    n_fft = 1024 if sr >= 16000 else 512
    hop = max(1, int(sr * 0.02))  # 20 ms hop
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop, center=True)[0]
    # Spectral flatness should be computed from the MAGNITUDE spectrogram.
    # Passing power spectrum with power=1 compresses the dynamic range and can
    # make values look almost uniformly close to zero.
    magnitude = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=True))
    flatness = librosa.feature.spectral_flatness(S=np.maximum(magnitude, 1e-10), power=2.0)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop, center=True)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    # YIN pitch estimates; treat only values within range as candidates, not guaranteed valid pitch.
    try:
        f0 = librosa.yin(y, fmin=65, fmax=min(500, sr / 2 - 1), sr=sr,
                         frame_length=n_fft, hop_length=hop)
        f0 = f0[:len(times)]
    except Exception:
        f0 = np.full(len(times), np.nan)
    if len(f0) < len(times):
        f0 = np.pad(f0, (0, len(times)-len(f0)), constant_values=np.nan)
    # Exclude low-energy frames from pitch display to reduce meaningless estimates.
    rms_db = 20 * np.log10(np.maximum(rms, 1e-8))
    active_threshold = max(np.percentile(rms_db, 20), np.max(rms_db) - 45)
    pitch_valid = np.isfinite(f0) & (f0 >= 65) & (f0 <= min(500, sr / 2 - 1)) & (rms_db > active_threshold)
    f0_plot = np.where(pitch_valid, f0, np.nan)
    # A smoothed flatness trace helps show local changes without hiding raw values.
    flat_smooth = pd.Series(flatness).rolling(window=5, center=True, min_periods=1).median().to_numpy()
    # Relative low-energy intervals; these are review candidates, not detected events.
    quiet_threshold_db = float(np.max(rms_db) - 35)
    quiet = rms_db < quiet_threshold_db
    return {"y": y, "sr": sr, "duration": duration, "n_fft": n_fft, "hop": hop,
            "times": times, "rms": rms, "rms_db": rms_db, "flatness": flatness,
            "flat_smooth": flat_smooth, "zcr": zcr, "f0": f0_plot,
            "pitch_valid": pitch_valid, "quiet": quiet, "quiet_threshold_db": quiet_threshold_db,
            "active_threshold": active_threshold}

try:
    with st.spinner("Analyzing audio features…"):
        d = analyze_audio(uploaded.getvalue(), uploaded.name)
except Exception as e:
    st.error(f"Could not analyze this audio: {e}")
    st.stop()

y, sr, times = d["y"], d["sr"], d["times"]
c1, c2, c3 = st.columns(3)
c1.metric("Duration", f"{d['duration']:.1f} s")
c2.metric("Sample rate", f"{sr:,} Hz")
c3.metric("Pitch frames retained", f"{100*np.mean(d['pitch_valid']):.1f}%")
st.audio(uploaded.getvalue())

st.subheader("1. Waveform and short-time energy")
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
wave_t = np.arange(len(y)) / sr
axes[0].plot(wave_t, y, linewidth=0.5)
axes[0].set_ylabel("Amplitude")
axes[0].grid(alpha=.25)
axes[1].plot(times, d["rms_db"], linewidth=1)
axes[1].axhline(d["quiet_threshold_db"], linestyle="--", label=f"Relative low-energy threshold ({d['quiet_threshold_db']:.1f} dB)")
axes[1].set_ylabel("RMS level (dB, relative)")
axes[1].set_xlabel("Time (s)")
axes[1].legend(loc="best")
axes[1].grid(alpha=.25)
fig.tight_layout(); st.pyplot(fig); plt.close(fig)

st.subheader("2. Pitch estimate (YIN; exploratory)")
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(times, d["f0"], ".", markersize=2, alpha=.65)
ax.set_ylim(60, min(520, sr / 2))
ax.set_xlabel("Time (s)"); ax.set_ylabel("Estimated F0 (Hz)")
ax.set_title("Low-energy frames hidden; remaining estimates are not confidence-calibrated")
ax.grid(alpha=.25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

st.subheader("3. Spectral flatness — interpretable view")
st.write("Spectral flatness compares the geometric and arithmetic means of the magnitude spectrum. Values nearer 0 indicate a more peaked/tonal spectrum; values nearer 1 indicate a flatter, more noise-like spectrum. It is a descriptor, not a standalone noise detector.")
flat = d["flatness"]
# Avoid using near-silent frames to set the zoom scale or the review threshold.
active = d["rms_db"] > d["active_threshold"]
flat_active = flat[active & np.isfinite(flat)]
if flat_active.size:
    q10, q50, q90, q99 = np.percentile(flat_active, [10, 50, 90, 99])
else:
    q10 = q50 = q90 = q99 = np.nan
m1, m2, m3, m4 = st.columns(4)
m1.metric("Active-frame median", f"{q50:.4f}" if np.isfinite(q50) else "N/A")
m2.metric("10th percentile", f"{q10:.4f}" if np.isfinite(q10) else "N/A")
m3.metric("90th percentile", f"{q90:.4f}" if np.isfinite(q90) else "N/A")
m4.metric("99th percentile", f"{q99:.4f}" if np.isfinite(q99) else "N/A")
fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
axes[0].plot(times, flat, linewidth=.7, alpha=.65, label="All frames")
axes[0].plot(times, d["flat_smooth"], linewidth=1.1, label="5-frame rolling median")
axes[0].set_ylabel("Flatness (0–1)"); axes[0].set_ylim(0, 1)
axes[0].set_title("Full scale — includes low-energy frames")
axes[0].legend(loc="upper right"); axes[0].grid(alpha=.25)
# Robust zoom: use active-frame 1st–99th percentiles and add margin. Keep a minimum
# y-span so near-constant data is honestly shown rather than magnified infinitely.
if flat_active.size:
    lo, hi = np.percentile(flat_active, [1, 99])
    span = max(float(hi - lo), 0.02)
    zoom_bottom = max(0.0, float(lo) - 0.15 * span)
    zoom_top = min(1.0, float(hi) + 0.15 * span)
    if zoom_top - zoom_bottom < 0.02:
        zoom_top = min(1.0, zoom_bottom + 0.02)
else:
    zoom_bottom, zoom_top = 0.0, 1.0
axes[1].plot(times, np.where(active, flat, np.nan), linewidth=.65, alpha=.55, label="Active frames")
axes[1].plot(times, np.where(active, d["flat_smooth"], np.nan), linewidth=1.2, label="Smoothed active frames")
axes[1].set_ylim(zoom_bottom, zoom_top)
axes[1].set_ylabel("Flatness (zoomed)"); axes[1].set_xlabel("Time (s)")
axes[1].set_title(f"Active-frame detail — y-axis {zoom_bottom:.3f} to {zoom_top:.3f}")
axes[1].legend(loc="upper right"); axes[1].grid(alpha=.25)
if flat_active.size:
    axes[2].hist(flat_active, bins=30)
    axes[2].axvline(q50, linestyle="--", label=f"Median {q50:.4f}")
    axes[2].set_title("Distribution across active frames")
    axes[2].set_xlabel("Spectral flatness"); axes[2].set_ylabel("Frame count")
    axes[2].legend(); axes[2].grid(alpha=.2)
else:
    axes[2].text(.5, .5, "No active frames under the current energy threshold", ha="center", va="center")
    axes[2].set_axis_off()
st.pyplot(fig); plt.close(fig)
# Threshold is computed only from active frames; inactive frames are excluded from flags.
flat_cutoff = float(np.percentile(flat_active, 90)) if flat_active.size else 1.0
high_flat = active & (flat >= flat_cutoff)
st.caption(f"High-flatness review cue uses the active-frame 90th percentile ({flat_cutoff:.4f}); it highlights relative outliers in this recording, not confirmed background noise. Corrected calculation uses magnitude spectrum with librosa's standard power=2 definition.")

st.subheader("4. Zero-crossing rate")
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(times, d["zcr"], linewidth=.8)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Zero-crossing rate")
ax.grid(alpha=.25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

st.subheader("5. Candidate low-energy intervals")
quiet = d["quiet"]
# Merge adjacent frames and keep intervals at least 0.2 seconds.
intervals = []
idx = np.flatnonzero(quiet)
if idx.size:
    starts = [idx[0]]; ends = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b > a + 1:
            ends.append(a); starts.append(b)
    ends.append(idx[-1])
    for a, b in zip(starts, ends):
        start_t = float(times[a]); end_t = float(times[b] + d["hop"] / sr)
        if end_t - start_t >= 0.2:
            intervals.append({"start_s": round(start_t, 2), "end_s": round(end_t, 2), "duration_s": round(end_t-start_t, 2)})
if intervals:
    interval_df = pd.DataFrame(intervals)
    st.dataframe(interval_df, use_container_width=True, hide_index=True)
else:
    interval_df = pd.DataFrame(columns=["start_s", "end_s", "duration_s"])
    st.info("No intervals met the current threshold and minimum-duration rule.")
st.caption("These are relative low-energy candidates only, not confirmed silence, speech gaps, or important events.")

st.subheader("6. Optional local transcription with noise reduction")
st.write("The app can optionally suppress background noise before sending audio to Whisper. The original remains untouched. Noise reduction can also remove quiet consonants or create artifacts, so compare the original and enhanced audio and treat the transcript as a hypothesis.")
if importlib.util.find_spec("faster_whisper") is None:
    st.info("Optional dependencies not installed. Run `pip install -r requirements-transcription.txt` in the same environment, then restart Streamlit.")
else:
    tcol1, tcol2 = st.columns([1, 2])
    with tcol1:
        whisper_size = st.selectbox("Local Whisper model", ["tiny", "base", "small", "medium"], index=1,
                                    help="Start with base for a CPU-friendly demo. Larger models are slower and use more memory.")
        language_choice = st.selectbox("Transcription language", ["Auto-detect", "English"], index=0)
        reduce_noise = st.checkbox("Reduce background noise before transcription", value=True)
        noise_strength = st.slider("Noise reduction strength", min_value=0.2, max_value=1.0, value=0.65, step=0.05,
                                   help="Higher values suppress more noise but may remove quiet speech details. Start around 0.5–0.7.")
        transcribe_clicked = st.button("Transcribe locally", type="primary")
    with tcol2:
        st.caption("Noise reduction uses a spectral-gating method. It does not intentionally shift pitch or translate speech, but artifacts are possible. Compare with the original, especially for radio speech and quiet consonants.")
    if reduce_noise and importlib.util.find_spec("noisereduce") is None:
        st.warning("Noise reduction package is missing. Install the updated optional requirements with `pip install -r requirements-transcription.txt`, then restart Streamlit.")
    if transcribe_clicked:
        try:
            from faster_whisper import WhisperModel
            whisper_source = y.astype(np.float32).copy()
            noise_reduction_applied = False
            if reduce_noise:
                if importlib.util.find_spec("noisereduce") is None:
                    st.error("Cannot apply noise reduction because `noisereduce` is not installed. Install requirements-transcription.txt and retry.")
                    st.stop()
                import noisereduce as nr
                with st.spinner("Reducing background noise while preserving the original sample rate…"):
                    whisper_source = nr.reduce_noise(y=whisper_source, sr=sr, stationary=False,
                                                     prop_decrease=float(noise_strength))
                noise_reduction_applied = True
                # Preview/download enhanced audio at its original sample rate.
                enhanced_buffer = io.BytesIO()
                import soundfile as sf
                sf.write(enhanced_buffer, whisper_source, sr, format="WAV")
                enhanced_bytes = enhanced_buffer.getvalue()
                st.markdown("**Enhanced audio preview**")
                st.audio(enhanced_bytes, format="audio/wav")
                st.download_button("Download noise-reduced WAV", enhanced_bytes,
                                   file_name="noise_reduced_audio.wav", mime="audio/wav")
            with st.spinner(f"Loading Whisper {whisper_size} and transcribing locally… first use may take a while"):
                model = WhisperModel(whisper_size, device="cpu", compute_type="int8")
                # Resampling changes sample rate for model input, not the underlying pitch.
                whisper_audio = librosa.resample(whisper_source, orig_sr=sr, target_sr=16000) if sr != 16000 else whisper_source
                kwargs = {"beam_size": 5, "vad_filter": False}
                if language_choice == "English":
                    kwargs["language"] = "en"
                segments, info = model.transcribe(whisper_audio, **kwargs)
                rows = [{"start_s": round(seg.start, 2), "end_s": round(seg.end, 2), "text": seg.text.strip()} for seg in segments]
                # Materialize lazy generator inside spinner.
                rows = list(rows)
            mode_label = "noise-reduced audio" if noise_reduction_applied else "original audio"
            st.success(f"Transcription of {mode_label} complete · detected language: {info.language} (probability {info.language_probability:.2f})")
            if rows:
                transcript_df = pd.DataFrame(rows)
                st.dataframe(transcript_df, use_container_width=True, hide_index=True)
                st.download_button("Download timestamped transcript (CSV)", transcript_df.to_csv(index=False).encode("utf-8"),
                                   file_name="local_transcript.csv", mime="text/csv")
                st.text_area("Transcript", "\n".join(f"[{r['start_s']:.2f}–{r['end_s']:.2f}] {r['text']}" for r in rows), height=220)
            else:
                st.warning("No speech segments were returned. Try a lower noise-reduction strength, enable/disable noise reduction to compare, or inspect the audio quality.")
        except Exception as e:
            st.error(f"Local transcription failed: {e}")
            st.caption("If this is the first run, check internet access for model download, available disk space, and the terminal traceback.")

feature_df = pd.DataFrame({"time_s": times, "rms_db_relative": d["rms_db"], "pitch_hz": d["f0"],
                           "pitch_frame_retained": d["pitch_valid"], "spectral_flatness": flat,
                           "spectral_flatness_smoothed": d["flat_smooth"], "active_energy_frame": active,
                           "high_spectral_flatness_flag": high_flat,
                           "zero_crossing_rate": d["zcr"]})
st.download_button("Download frame-level features (CSV)", feature_df.to_csv(index=False).encode("utf-8"),
                   file_name="audio_triage_features.csv", mime="text/csv")
st.download_button("Download candidate intervals (CSV)", interval_df.to_csv(index=False).encode("utf-8"),
                   file_name="candidate_review_intervals.csv", mime="text/csv")
with st.expander("Analysis settings and caveats"):
    st.write(f"FFT window: {d['n_fft']} samples ({1000*d['n_fft']/sr:.1f} ms); hop: {1000*d['hop']/sr:.1f} ms.")
    st.write(f"Active-frame threshold: {d['active_threshold']:.1f} dB relative RMS level. Quiet threshold: {d['quiet_threshold_db']:.1f} dB relative RMS level. High-flatness cue: active-frame 90th percentile ({flat_cutoff:.4f}).")
    st.write("Spectral flatness depends on window size, frequency content, and recording conditions. Validate with known clean speech, silence, and noise before using it as a triage cue.")
