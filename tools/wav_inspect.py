# Owner: S2
# Tool: WAV Quality Inspector
# Usage: python tools/wav_inspect.py <path_to_wav>
# Checks: sample rate, bit depth, clipping, RMS level
# Last validated: 2026-09-03

import sys
import os
import numpy as np
import soundfile as sf

EXPECTED_SR   = 16000
EXPECTED_BITS = 16
RMS_LOW_DB    = -40   # below this = probably silent / too quiet
RMS_HIGH_DB   = -3    # above this = near clipping


def inspect_wav(path: str) -> bool:
    """
    Inspect a WAV file for quality issues.
    Returns True if file passes all checks, False otherwise.
    """
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return False

    info = sf.info(path)
    data, sr = sf.read(path, dtype='float32', always_2d=False)

    print(f"\n{'='*50}")
    print(f"File      : {path}")
    print(f"Sample rate: {sr} Hz  (expected {EXPECTED_SR})")
    print(f"Channels  : {info.channels}")
    print(f"Subtype   : {info.subtype}  (expected PCM_16)")
    print(f"Duration  : {len(data)/sr:.3f}s")
    print(f"Samples   : {len(data)}")

    # Compute RMS
    rms = np.sqrt(np.mean(data ** 2))
    rms_db = 20 * np.log10(rms + 1e-9)
    peak_db = 20 * np.log10(np.abs(data).max() + 1e-9)
    clipped  = int(np.sum(np.abs(data) >= 0.999))

    print(f"RMS       : {rms_db:.1f} dBFS")
    print(f"Peak      : {peak_db:.1f} dBFS")
    print(f"Clipped samples: {clipped}")
    print(f"{'='*50}")

    # Checks
    passed = True

    if sr != EXPECTED_SR:
        print(f"[FAIL] Sample rate {sr} != {EXPECTED_SR}")
        passed = False
    else:
        print(f"[PASS] Sample rate: {sr} Hz")

    if info.subtype != 'PCM_16':
        print(f"[FAIL] Subtype {info.subtype} != PCM_16")
        passed = False
    else:
        print(f"[PASS] Bit depth: PCM_16")

    if clipped > 0:
        print(f"[FAIL] Clipping detected: {clipped} samples")
        passed = False
    else:
        print(f"[PASS] No clipping")

    if rms_db < RMS_LOW_DB:
        print(f"[WARN] RMS {rms_db:.1f} dBFS is very low — check mic / distance")
    elif rms_db > RMS_HIGH_DB:
        print(f"[WARN] RMS {rms_db:.1f} dBFS is very high — risk of clipping")
    else:
        print(f"[PASS] RMS {rms_db:.1f} dBFS in acceptable range")

    print(f"\n{'PASS ✅' if passed else 'FAIL ❌'} — {os.path.basename(path)}")
    return passed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/wav_inspect.py <path_to_wav>")
        print("       python tools/wav_inspect.py ml/data/train/keyword/*.wav")
        sys.exit(1)

    files = sys.argv[1:]
    results = []
    for f in files:
        results.append(inspect_wav(f))

    if len(files) > 1:
        passed = sum(results)
        print(f"\n{'='*50}")
        print(f"SUMMARY: {passed}/{len(files)} files passed")
        print(f"{'='*50}")

    sys.exit(0 if all(results) else 1)
