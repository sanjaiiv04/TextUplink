"""Local file loader for the selected enhancement components."""
def load_audio(path,target_sr=48000):
    import librosa
    import numpy as np
    audio,sr=librosa.load(path,sr=target_sr,mono=True)
    return np.asarray(audio,dtype=np.float32),sr
