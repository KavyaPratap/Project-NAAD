# S3 — ADPCM Round-trip Test
# Run this BEFORE flashing to ESP32 to verify Python decoder works
# Usage: python tools/test_adpcm.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

import numpy as np
from adpcm_decoder import AdpcmDecoder

STEP_TABLE   = [7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
                50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230,
                253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
                1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327,
                3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487,
                12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767]
INDEX_TABLE  = [-1, -1, -1, -1, 2, 4, 6, 8]


def encode_py(pcm_list):
    """Python ADPCM encoder — mirrors adpcm.c exactly."""
    pred, idx, codes = 0, 0, []
    for s in pcm_list:
        step = STEP_TABLE[idx]
        diff = int(s) - pred
        code = 0
        if diff < 0: code = 8; diff = -diff
        if diff >= step:      code |= 4; diff -= step
        if diff >= step >> 1: code |= 2; diff -= step >> 1
        if diff >= step >> 2: code |= 1
        r = step >> 3
        if code & 4: r += step
        if code & 2: r += step >> 1
        if code & 1: r += step >> 2
        if code & 8: r = -r
        pred = max(-32768, min(32767, pred + r))
        idx  = max(0, min(88, idx + INDEX_TABLE[code & 7]))
        codes.append(code & 0xF)
    return codes


def pack_codes(codes):
    """Pack 4-bit codes into bytes: low nibble first, high nibble second."""
    return bytes([(codes[i]) | (codes[i+1] << 4) for i in range(0, len(codes)-1, 2)])


def run_test(name, pcm, min_snr=25.0):
    codes  = encode_py(pcm.tolist())
    packed = pack_codes(codes)
    dec    = AdpcmDecoder()
    decoded = np.array(dec.decode(packed), dtype=np.int16)
    min_len = min(len(pcm), len(decoded))
    
    max_diff = np.max(np.abs(pcm[:min_len].astype(np.int32) - decoded[:min_len].astype(np.int32)))
    
    sig_pow = np.mean(pcm[:min_len].astype(np.float64) ** 2)
    nse_pow = np.mean((pcm[:min_len] - decoded[:min_len]).astype(np.float64) ** 2)
    
    if sig_pow < 1e-6 and max_diff == 0:
        snr = float('inf')
        passed = True
        snr_str = "  inf (exact)"
    else:
        snr = 10 * np.log10((sig_pow + 1e-9) / (nse_pow + 1e-9))
        passed = snr >= min_snr
        snr_str = f"{snr:5.1f} dB"

    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name:30s}  SNR = {snr_str}")
    return passed


print("=" * 60)
print("  ADPCM Round-trip Self-Test")
print("=" * 60)
t = np.linspace(0, 1, 16000)
all_pass = True
all_pass &= run_test("440Hz sine @ 16kHz",      (np.sin(2*np.pi*440*t)*10000).astype(np.int16), min_snr=25.0)
all_pass &= run_test("1kHz sine @ 16kHz",       (np.sin(2*np.pi*1000*t)*10000).astype(np.int16), min_snr=25.0)
all_pass &= run_test("silence (zeros)",          np.zeros(16000, dtype=np.int16))
all_pass &= run_test("white noise",              (np.random.randn(16000)*3000).clip(-32768,32767).astype(np.int16), min_snr=12.0)
all_pass &= run_test("max amplitude signal",     (np.sin(2*np.pi*440*t)*32000).astype(np.int16), min_snr=25.0)
print("=" * 60)
print(f"  Overall: {'✅ ALL PASS' if all_pass else '❌ SOME TESTS FAILED'}")
print("=" * 60)
if all_pass:
    print("\n  → Safe to use adpcm.c + adpcm_decoder.py together on hardware.")
else:
    print("\n  → Fix mismatch between adpcm.c and adpcm_decoder.py before hardware test!")
