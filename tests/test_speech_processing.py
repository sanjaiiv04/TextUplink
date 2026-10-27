import unittest
from unittest.mock import patch
import numpy as np
from textuplink.speech_processing import enhance_speech


class SpeechProcessingTests(unittest.TestCase):
    def test_disabled_isolation_preserves_the_recording_exactly(self):
        speech=np.array([0.,.01,-.02,1.2,-1.1],dtype=np.float32)
        result,stats=enhance_speech(speech,reduce_noise=False)
        np.testing.assert_array_equal(result,speech)
        self.assertFalse(stats['noise_reduction'])

    def test_spectral_option_reduces_noise(self):
        y=np.random.default_rng(11).normal(0,.03,72000).astype(np.float32)
        result,_=enhance_speech(y,backend='spectral',strength=.7)
        self.assertEqual(result.shape,y.shape)
        self.assertLess(np.sqrt(np.mean(result**2)),np.sqrt(np.mean(y**2)))





    def test_neural_wet_dry_mix_and_failure_never_silently_fall_back(self):
        y=np.full(2400,.1,dtype=np.float32)
        with patch('textuplink.speech_processing.neural_enhance',return_value=y*.2):
            mixed,stats=enhance_speech(y,backend='deepfilter',strength=.5)
            np.testing.assert_allclose(mixed,y*.6)
            self.assertEqual(stats['enhancement_backend'],'deepfilter')
        with patch('textuplink.speech_processing.neural_enhance',side_effect=RuntimeError('missing weights')):
            with self.assertRaisesRegex(RuntimeError,'missing weights'):
                enhance_speech(y,backend='mossformer')
        with patch('textuplink.speech_processing.neural_enhance',return_value=np.zeros(1)):
            with self.assertRaises(ValueError):enhance_speech(y,backend='deepfilter')

    def test_invalid_input_rejected(self):
        for audio in [np.array([]),np.array([np.nan]),np.zeros((2,100))]:
            with self.assertRaises(ValueError):enhance_speech(audio)

    def test_reuse_rejects_changed_artifacts_before_native_execution(self):
        import hashlib
        import json
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as work:
            root=Path(work)
            np.save(root/'input.npy',np.zeros(2400,dtype=np.float32),allow_pickle=False)
            (root/'model_mlx.safetensors').write_bytes(b'changed weights')
            (root/'manifest.json').write_text(json.dumps({'sha256':{
                'model_mlx.safetensors':hashlib.sha256(b'expected weights').hexdigest()}}))
            result=subprocess.run([sys.executable,'-B','-m','textuplink.enhancement_worker',
                'reuse',str(root/'input.npy'),str(root/'output.npy'),str(root)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('RE-USE artifact missing or changed',result.stderr)
            self.assertFalse((root/'output.npy').exists())


if __name__=='__main__':unittest.main()
