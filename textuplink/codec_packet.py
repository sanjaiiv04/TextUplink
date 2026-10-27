"""Versioned, bit-packed codec packets. No pickle or executable serialization."""
import hashlib
import json
import math
import struct
import numpy as np

MAGIC = b'TUP1'
MAX_PACKET = 8_000_000
MAX_SAMPLES = 24_000 * 60
FAMILIES = {'encodec': (10, 32), 'mimi': (11, 32), 'focalcodec': (13, 1)}


def pack_codes(codes, *, family, model_fingerprint, samples, setting):
    bits, _ = FAMILIES[family]
    codes = np.asarray(codes)
    if codes.dtype.kind not in 'iu' or codes.size == 0 or np.any(codes < 0) or np.any(codes >= 2**bits):
        raise ValueError('Invalid codec indices')
    values = codes.reshape(-1).astype(np.uint16)
    binary = ((values[:, None] >> np.arange(bits-1, -1, -1)) & 1).astype(np.uint8)
    body = np.packbits(binary.reshape(-1), bitorder='big').tobytes()
    header = dict(family=family, fingerprint=model_fingerprint, sample_rate=24000,
                  samples=int(samples), shape=list(codes.shape), bits=bits, setting=setting,
                  body_sha256=hashlib.sha256(body).hexdigest())
    raw = json.dumps(header, separators=(',', ':'), allow_nan=False).encode()
    packet = MAGIC + struct.pack('>I', len(raw)) + raw + body
    # Check that sender emits packets the independent receiver accepts.
    unpack_codes(packet)
    return packet


def unpack_codes(packet):
    if not isinstance(packet, bytes) or not 8 <= len(packet) <= MAX_PACKET or packet[:4] != MAGIC:
        raise ValueError('Invalid TextUplink packet or packet too large')
    size = struct.unpack('>I', packet[4:8])[0]
    if not 0 < size <= 4096 or 8+size > len(packet):
        raise ValueError('Invalid packet header length')
    try:
        header = json.loads(packet[8:8+size])
        family = header['family']
        bits, max_books = FAMILIES[family]
        shape = header['shape']
        samples = header['samples']
        if type(samples) is not int or not 1 <= samples <= MAX_SAMPLES:
            raise ValueError('Unsupported duration')
        if header['sample_rate'] != 24000 or header['bits'] != bits:
            raise ValueError('Unsupported sample rate or code width')
        if not isinstance(shape, list) or len(shape) != (4 if family == 'encodec' else 3):
            raise ValueError('Invalid code shape')
        if any(type(x) is not int or x < 1 for x in shape) or any(x != 1 for x in shape[:-2]):
            raise ValueError('Only single-recording mono packets are supported')
        if shape[-2] > max_books or shape[-1] > 5000:
            raise ValueError('Code dimensions exceed limits')
        # EnCodec: 75 frames/s; Mimi: 12.5 frames/s. Allow codec end padding.
        frame_rate = {'encodec':75, 'mimi':12.5, 'focalcodec':50}[family]
        if abs(shape[-1] - math.ceil(samples/24000*frame_rate)) > 2:
            raise ValueError('Code length does not match declared duration')
        fingerprint = header['fingerprint']
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
            raise ValueError('Invalid model fingerprint')
        count = math.prod(shape)
        body = packet[8+size:]
        if len(body) != math.ceil(count*bits/8):
            raise ValueError('Payload length mismatch')
        if hashlib.sha256(body).hexdigest() != header['body_sha256']:
            raise ValueError('Corrupt codec payload')
        binary = np.unpackbits(np.frombuffer(body, dtype=np.uint8), bitorder='big')[:count*bits].reshape(count,bits)
        values = (binary.astype(np.uint16) * (1 << np.arange(bits-1,-1,-1))).sum(axis=1).astype(np.int64)
        return header, values.reshape(shape)
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError('Malformed codec packet header') from exc
