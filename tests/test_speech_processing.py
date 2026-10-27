import unittest
from unittest.mock import patch
import numpy as np
from textuplink.speech_processing import enhance_speech


class SpeechProcessingTests(unittest.TestCase):
    def test_gain_raises_voice_without_changing_relative_delivery(self):
        speech=np.array([0.,.01,-.02,.04,-.08],dtype=np.float32)
        result,stats=enhance_speech(speech,reduce_noise=False,boost_db=12.)
        np.testing.assert_allclose(result,speech*10**(12/20),rtol=1e-6)
        self.assertAlmostEqual(stats['output_rms_dbfs']-stats['cleaned_rms_dbfs'],12.,places=5)
        self.assertFalse(stats['peak_limited'])
        self.assertEqual(len(result),len(speech))

    def test_headroom_limits_gain_uniformly_instead_of_clipping(self):
        speech=np.array([.01,.2,-.8,.03],dtype=np.float32)
        result,stats=enhance_speech(speech,reduce_noise=False,boost_db=24.)
        ceiling=10**(-1/20)
        self.assertLessEqual(float(np.max(np.abs(result))),ceiling+1e-7)
        np.testing.assert_allclose(result/speech,result[0]/speech[0],rtol=1e-6)
        self.assertTrue(stats['peak_limited'])
        self.assertLess(stats['applied_gain_db'],24.)

    def test_silence_does_not_get_extreme_gain(self):
        result,stats=enhance_speech(np.zeros(24000),reduce_noise=False,boost_db=24.)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(stats['applied_gain_db'],0)
        self.assertTrue(np.all(result==0))

    def test_noise_suppression_precedes_measurable_gain(self):
        rng=np.random.default_rng(11)
        y=(.03*rng.normal(size=72000)).astype(np.float32)
        y[24000:48000]+=(.08*np.sin(2*np.pi*220*np.arange(24000)/24000)).astype(np.float32)
        cleaned,_=enhance_speech(y,strength=.7,boost_db=0.,steady_noise=True)
        boosted,stats=enhance_speech(y,strength=.7,boost_db=9.,steady_noise=True)
        self.assertLess(np.sqrt(np.mean(cleaned[:12000]**2)),np.sqrt(np.mean(y[:12000]**2)))
        np.testing.assert_allclose(boosted,cleaned*10**(stats['applied_gain_db']/20),rtol=1e-5,atol=1e-7)
        self.assertEqual(len(boosted),len(y))
        self.assertGreater(stats['output_rms_dbfs'],stats['cleaned_rms_dbfs'])

    def test_neural_wet_dry_mix_and_failure_never_silently_fall_back(self):
        y=np.full(2400,.1,dtype=np.float32)
        with patch('textuplink.speech_processing.neural_enhance',return_value=y*.2):
            mixed,stats=enhance_speech(y,backend='deepfilter',strength=.5,boost_db=0.)
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


if __name__=='__main__':unittest.main()
