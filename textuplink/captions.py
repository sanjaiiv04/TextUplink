"""Measured vocal delivery and audio-synchronized expressive captions."""
import base64
import html
import json
import numpy as np
from textuplink.audio import wav_bytes
from textuplink.transcript import COLORS


def measure_delivery(rows,audio):
    """Recording-relative prosody; no inferred speakers or emotion from pitch."""
    import librosa
    y=np.asarray(audio,dtype=np.float32)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('Use finite mono audio')
    hop=240
    f0,voiced,probability=librosa.pyin(y,sr=24000,fmin=65,fmax=500,frame_length=1024,hop_length=hop)
    rms=librosa.feature.rms(y=y,frame_length=1024,hop_length=hop)[0]
    times=np.arange(len(f0))*hop/24000
    valid=np.isfinite(f0)&voiced&(probability>=.5)
    # Use aligned speech frames, not noisy pauses, as the loudness reference.
    speech=np.zeros(len(times),dtype=bool)
    for row in rows:
        speech|=(times>=float(row['start_s']))&(times<float(row['end_s']))
    pitch_base=float(np.median(f0[valid&speech])) if np.any(valid&speech) else None
    levels=20*np.log10(np.maximum(rms,1e-8))
    level_base=float(np.median(levels[speech])) if np.any(speech) else -160.
    result=[]
    for row in rows:
        start,end=float(row['start_s']),float(row['end_s'])
        if not 0<=start<=end<=len(y)/24000+.1:
            raise ValueError('Word timing is outside this recording')
        mask=(times>=start)&(times<end)
        pitched=mask&valid
        pitch=float(np.median(f0[pitched])) if np.any(pitched) else None
        relative=float(12*np.log2(pitch/pitch_base)) if pitch is not None and pitch_base else None
        indices=np.flatnonzero(pitched)
        movement=None
        if len(indices)>=4:
            n=max(1,len(indices)//3)
            movement=float(12*np.log2(np.median(f0[indices[-n:]])/np.median(f0[indices[:n]])))
        level=float(np.median(levels[mask])-level_base) if np.any(mask) else None
        result.append(dict(row,duration_s=end-start,pitch_hz=pitch,pitch_relative_semitones=relative,
                           pitch_movement_semitones=movement,loudness_relative_db=level,
                           prosody_reference='Recording-relative; not speaker-normalized'))
    return result


def caption_player(rows,audio,emotion_colors=False):
    """Standalone synchronized playback; untrusted text is escaped before HTML."""
    tokens=[]
    for i,row in enumerate(rows):
        level=float(np.clip(row.get('loudness_relative_db') or 0,-9,9))
        pitch=row.get('pitch_relative_semitones')
        movement=row.get('pitch_movement_semitones')
        size=24+level*.7
        offset=float(np.clip(pitch or 0,-6,6))*-1.5
        color=COLORS.get(row.get('emotion'),'#334155') if emotion_colors else '#172b4d'
        arrow=' ↗' if movement is not None and movement>=2 else ' ↘' if movement is not None and movement<=-2 else ''
        title=f"{row['start_s']:.2f}–{row['end_s']:.2f}s · pitch {round(pitch,1) if pitch is not None else 'unavailable'} semitones · {row.get('emotion','Unlabelled')} · {row.get('annotation_source','None')}"
        tokens.append(f'<span class="word" id="w{i}" style="font-size:{size:.1f}px;font-weight:{700 if level>3 else 450};transform:translateY({offset:.1f}px);color:{color}" title="{html.escape(title,quote=True)}">{html.escape(str(row["word"]))}{arrow}</span>')
    timing=json.dumps([[float(r['start_s']),float(r['end_s'])] for r in rows],allow_nan=False)
    audio64=base64.b64encode(wav_bytes(audio)).decode('ascii')
    return '''<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:system-ui,sans-serif;margin:0;color:#172b4d;background:#f4f6fa;padding:18px;border-radius:16px}
#words{line-height:3.1;max-height:260px;overflow-y:auto;padding:14px;background:white;border-radius:12px;margin:12px 0}
.word{display:inline-block;border-radius:6px;padding:0 5px;margin:0 2px;transition:background .12s}
.word.active{background:#ffe2a8;outline:2px solid #b66a17}
audio{width:100%}p{font-size:13px;color:#475569;margin:6px 0}
</style></head><body><strong>Expressive captions</strong><p>Bigger / bolder = louder · Higher = higher pitch · ↗ / ↘ = pitch movement. Relative to this recording.</p>
<audio id="audio" controls src="data:audio/wav;base64,'''+audio64+'''"></audio><div id="words">'''+ ' '.join(tokens)+'''</div><p>Highlight follows playback. Missing pitch stays on the baseline; it does not establish whispering. Hover for timing and annotation source.</p>
<script>
const audio=document.getElementById('audio'), times='''+timing+''';
let current=-1;
function update(){const t=audio.currentTime;const next=times.findIndex(([s,e])=>s<=t&&t<e);
if(next!==current){if(current>=0)document.getElementById('w'+current).classList.remove('active');
if(next>=0){const word=document.getElementById('w'+next);word.classList.add('active');word.scrollIntoView({block:'nearest'});}current=next;}}
audio.addEventListener('timeupdate',update);audio.addEventListener('seeked',update);audio.addEventListener('ended',update);
</script></body></html>'''
