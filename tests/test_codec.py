import io
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from textuplink.codec_packet import pack_codes, unpack_codes, MAGIC
from textuplink.neural_codec import LocalCodec, add_noise, compare_audio, read_audio

FINGERPRINT='a'*64


class PacketTests(unittest.TestCase):
    def test_bitpacked_indices_exactly_roundtrip(self):
        for family,bits,shape in [('encodec',10,(1,1,2,75)),('mimi',11,(1,8,13)),('focalcodec',13,(1,1,50))]:
            codes=np.random.default_rng(9).integers(0,2**bits,shape,dtype=np.int64)
            codes.reshape(-1)[0]=2**bits-1
            packet=pack_codes(codes,family=family,model_fingerprint=FINGERPRINT,samples=24000,setting=3)
            header,result=unpack_codes(packet)
            np.testing.assert_array_equal(result,codes)
            size=struct.unpack('>I',packet[4:8])[0]
            self.assertEqual(len(packet)-8-size,(codes.size*bits+7)//8)

    def test_corruption_and_truncation(self):
        packet=pack_codes(np.zeros((1,1,2,75),dtype=int),family='encodec',model_fingerprint=FINGERPRINT,samples=24000,setting=1.5)
        for broken in [b'',packet[:7],packet[:-1],packet[:-1]+b'\xff',b'NOPE'+packet[4:]]:
            with self.assertRaises(ValueError): unpack_codes(broken)

    def test_reject_unsafe_or_inconsistent_shape(self):
        packet=pack_codes(np.zeros((1,1,2,75),dtype=int),family='encodec',model_fingerprint=FINGERPRINT,samples=24000,setting=1.5)
        size=struct.unpack('>I',packet[4:8])[0]
        header=json.loads(packet[8:8+size])
        for key,value in [('samples',10**10),('shape',[1,1,32,10**10]),('shape',[1,1,2,-1]),('sample_rate',48000),('fingerprint','wrong')]:
            changed=dict(header);changed[key]=value
            raw=json.dumps(changed).encode()
            with self.assertRaises(ValueError): unpack_codes(MAGIC+struct.pack('>I',len(raw))+raw+packet[8+size:])

    def test_invalid_indices_rejected(self):
        for value in [-1,1024]:
            with self.assertRaises(ValueError):
                pack_codes(np.full((1,1,2,75),value),family='encodec',model_fingerprint=FINGERPRINT,samples=24000,setting=1.5)


class AudioTests(unittest.TestCase):
    def test_noise_has_requested_snr_and_shared_gain(self):
        clean=(.2*np.sin(np.arange(24000)*.05)).astype(np.float32)
        ref,noisy=add_noise(clean,10)
        measured=10*np.log10(np.mean(ref**2)/np.mean((noisy-ref)**2))
        self.assertAlmostEqual(float(measured),10,places=4)
        self.assertLessEqual(float(np.max(np.abs(noisy))),1)
        with self.assertRaises(ValueError): add_noise(np.zeros(100),10)

    def test_resampling_and_duration_limit(self):
        data=io.BytesIO();sf.write(data,np.zeros((8000,2)),8000,format='WAV')
        self.assertEqual(len(read_audio(data.getvalue())),24000)
        data=io.BytesIO();sf.write(data,np.zeros(61*1000),1000,format='WAV')
        with self.assertRaises(ValueError):read_audio(data.getvalue())

    def test_identity_signal_metrics(self):
        audio=.2*np.sin(np.arange(24000)*.05)
        result=compare_audio(audio,audio.copy())
        self.assertAlmostEqual(result['log_spectral_distance_db'],0)
        self.assertAlmostEqual(result['energy_contour_correlation'],1)
        self.assertAlmostEqual(result['level_change_db'],0)


@unittest.skipUnless(Path('models/encodec_24khz/config.json').exists(),'Provision EnCodec first')
class OfflineCodecTests(unittest.TestCase):
    def test_independent_receiver_needs_only_packet_and_weights(self):
        with patch('socket.socket.connect',side_effect=AssertionError('Network forbidden during inference')):
            sender=LocalCodec('encodec','models/encodec_24khz')
            source=(.1*np.sin(np.arange(24137)*2*np.pi*220/24000)).astype(np.float32)
            packet,_=sender.encode(source,1.5)
            del sender,source
            receiver=LocalCodec('encodec','models/encodec_24khz')
            decoded,_=receiver.decode(packet)
            self.assertEqual(len(decoded),24137)
            self.assertTrue(np.isfinite(decoded).all())
            receiver.fingerprint='b'*64
            with self.assertRaises(ValueError):receiver.decode(packet)

    @unittest.skipUnless(Path('models/focalcodec/model.json').exists(),'Provision FocalCodec first')
    def test_focalcodec_independent_offline_receiver(self):
        with patch('socket.socket.connect',side_effect=AssertionError('Network forbidden during inference')):
            sender=LocalCodec('focalcodec','models/focalcodec')
            source=(.1*np.sin(np.arange(24137)*2*np.pi*220/24000)).astype(np.float32)
            packet,_=sender.encode(source,.65)
            del sender,source
            receiver=LocalCodec('focalcodec','models/focalcodec')
            decoded,_=receiver.decode(packet)
            self.assertEqual(len(decoded),24137)
            self.assertTrue(np.isfinite(decoded).all())

    @unittest.skipUnless(Path('models/mimi/config.json').exists(),'Provision Mimi first')
    def test_mimi_offline_receiver_and_partial_frame(self):
        with patch('socket.socket.connect',side_effect=AssertionError('Network forbidden during inference')):
            sender=LocalCodec('mimi','models/mimi')
            source=(.1*np.sin(np.arange(24137)*2*np.pi*220/24000)).astype(np.float32)
            packet,_=sender.encode(source,8)
            del sender,source
            receiver=LocalCodec('mimi','models/mimi')
            decoded,_=receiver.decode(packet)
            self.assertEqual(len(decoded),24137)
            self.assertTrue(np.isfinite(decoded).all())


if __name__=='__main__':unittest.main()
