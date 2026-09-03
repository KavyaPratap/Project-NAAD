# PARALLEL WORKING GUIDE + MOCK STUBS
## Project NAAD · SIH26172 · How All 3 Work Simultaneously

---

## THE PROBLEM

```
S2 needs S1's mfcc_reference.py  → can't start training
S3 needs S2's TFLM model         → can't build trigger pipeline
S3 needs real ESP32 hardware     → can't test server
```

This cascade would mean: S2 waits 4hrs for S1 → S3 waits 8hrs for S2 → you lose 12 hours.

## THE SOLUTION: MOCKS

Every dependency is replaced with a mock stub that has the **exact same interface**
as the real thing. When the real version is ready, you swap ONE import line.
Zero other code changes needed.

```
mocks/
  mfcc_mock.py          ← S2 uses while S1 is coding
  adpcm_mock_stream.py  ← S3 uses while hardware isn't ready
  kws_mock.py           ← S3 uses while S2 is training

Rule: mocks live in mocks/ folder, never in the main code path.
Swap: change `from mocks.mfcc_mock import extract_mfcc`
      to    `from ml.scripts.mfcc_reference import extract_mfcc`
```

---

## DAY 0 PARALLEL SPRINT — WHO DOES WHAT RIGHT NOW

```
Hour 0-1:  ALL → GitHub setup (clone, branch, folder structure)
Hour 1+:   SPLIT

S1:  Writes mfcc_reference.py (real)        NO dependencies → start immediately
S2:  Records dataset + uses mfcc_mock.py    NO dependencies → start immediately
S3:  Builds Python server + adpcm_mock      NO dependencies → start immediately
```

**Nobody waits for anyone.** Here's the exact breakdown:

| Time | S1 (DSP) | S2 (KWS/ML) | S3 (Server) |
|------|----------|-------------|-------------|
| Hr 0-2 | GitHub setup + mfcc_reference.py skeleton | GitHub setup + record.py + start recording 50 samples | GitHub setup + server/receiver.py + adpcm_decoder.py |
| Hr 2-6 | Implement MFCC step by step (pre-emphasis → FFT → Mel → DCT) | Augment with mfcc_mock.py, set up DS-CNN architecture | Build faster-whisper + endpoint, test with pre-recorded WAV |
| Hr 6-8 | **MERGE mfcc_reference.py to main** | **S2 swaps mock → real S1 MFCC, restart pipeline** | Wire adpcm_decoder to receiver, test full server loop |
| Hr 8-16 | Port VAD+MFCC to C (firmware) | Train DS-CNN, quantize to INT8, **MERGE kws_model files** | ESP32 I2S bring-up + ADPCM encoder, use kws_mock |
| Hr 16-24 | Integrate C DSP into FreeRTOS task | Measure FAR/recall on test set | **S3 swaps kws_mock → real S2 TFLM**, integrate end-to-end |

---

## MOCK FILES (Copy these into your `mocks/` folder NOW)

### `mocks/__init__.py`
```python
# This file is intentionally empty
```

---

### `mocks/mfcc_mock.py`
**WHO USES THIS:** S2 — while waiting for S1 to finish `mfcc_reference.py`

```python
# mocks/mfcc_mock.py
# Owner: ALL (shared mock)
# Purpose: Drop-in replacement for ml/scripts/mfcc_reference.py
#          Has IDENTICAL function signature — swap one import line when S1 is done
#
# S2 uses:   from mocks.mfcc_mock import extract_mfcc
# After S1:  from ml.scripts.mfcc_reference import extract_mfcc  ← ONLY change needed

import numpy as np
from typing import Union
import librosa

# Constants MUST match S1's real implementation exactly
SAMPLE_RATE   = 16000
N_FRAMES      = 49
N_MFCC        = 13

def extract_mfcc(
    source: Union[str, np.ndarray],
    target_frames: int = 49,
    quantize: bool = True
) -> np.ndarray:
    """
    MOCK implementation of mfcc_reference.extract_mfcc()
    
    Uses librosa's MFCC as approximation — good enough for:
      - Building and testing the training pipeline
      - Checking data loading works
      - Verifying DS-CNN input/output shapes
      - Running a first training pass

    NOT good enough for:
      - Final model training (must swap to S1's version for that)
      - Validating C vs Python consistency
    
    Interface is IDENTICAL to the real version:
      Input:  WAV path or float32 np.ndarray at 16kHz
      Output: np.ndarray shape (49, 13), dtype int8 (if quantize=True)
    """
    if isinstance(source, str):
        audio, _ = librosa.load(source, sr=SAMPLE_RATE, mono=True)
    else:
        audio = source.astype(np.float32)

    # librosa MFCC — approximate, not the canonical version
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=26,
        fmin=80,
        fmax=8000
    ).T  # librosa returns (n_mfcc, frames) → transpose to (frames, n_mfcc)

    # Pad or trim to target_frames
    if mfcc.shape[0] < target_frames:
        pad = np.zeros((target_frames - mfcc.shape[0], N_MFCC), dtype=np.float32)
        mfcc = np.vstack([mfcc, pad])
    else:
        mfcc = mfcc[:target_frames]

    if not quantize:
        return mfcc.astype(np.float32)

    # Normalize and cast to int8
    mean = mfcc.mean(axis=0, keepdims=True)
    std  = mfcc.std(axis=0, keepdims=True) + 1e-8
    normalized = (mfcc - mean) / std
    scaled = np.clip(normalized * 32.0, -128, 127)
    return scaled.astype(np.int8)
```

**S2 — how to use it:**
```python
# At the TOP of augment.py and train.py, write this:

# ============================================================
# DEPENDENCY STATUS:
# [ ] Using MOCK MFCC (S1 not done yet)  ← check this box once swapped
# [x] Using REAL MFCC from S1
# To swap: change the import below, nothing else
# ============================================================

# While S1 is working:
from mocks.mfcc_mock import extract_mfcc

# Once S1 merges mfcc_reference.py to main (S1 will tell you on WhatsApp):
# from ml.scripts.mfcc_reference import extract_mfcc   ← uncomment this
# from mocks.mfcc_mock import extract_mfcc             ← comment out above
```

---

### `mocks/adpcm_mock_stream.py`
**WHO USES THIS:** S3 — while waiting for ESP32 hardware to be set up

```python
# mocks/adpcm_mock_stream.py
# Owner: S3
# Purpose: Simulates the exact TCP packet stream that the ESP32 firmware sends.
#          S3 runs this on their laptop to test server/receiver.py without hardware.
#
# Run:  python mocks/adpcm_mock_stream.py --wav path/to/keyword.wav
# This sends fake-but-correct ADPCM packets to your local server on port 5555

import socket
import struct
import numpy as np
import librosa
import argparse
import time

# Must match firmware/adpcm/adpcm.c EXACTLY
STEP_TABLE = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34,
    37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143,
    157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494,
    544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552,
    1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428,
    4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635,
    13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
]
INDEX_TABLE = [-1, -1, -1, -1, 2, 4, 6, 8]

def adpcm_encode(pcm_samples, pred=0, idx=0):
    """Encode int16 PCM to ADPCM bytes. Returns (bytes, final_pred, final_idx)."""
    codes = []
    for s in pcm_samples:
        step = STEP_TABLE[idx]
        diff = int(s) - pred
        code = 0
        if diff < 0: code = 8; diff = -diff
        if diff >= step:        code |= 4; diff -= step
        if diff >= step >> 1:   code |= 2; diff -= (step >> 1)
        if diff >= step >> 2:   code |= 1
        r = step >> 3
        if code & 4: r += step
        if code & 2: r += step >> 1
        if code & 1: r += step >> 2
        if code & 8: r = -r
        pred = max(-32768, min(32767, pred + r))
        idx  = max(0, min(88, idx + INDEX_TABLE[code & 7]))
        codes.append(code & 0xF)
    # Pack 2 codes per byte: low nibble first
    packed = bytes([(codes[i] | (codes[i+1] << 4)) for i in range(0, len(codes)-1, 2)])
    return packed, pred, idx

def build_packet(adpcm_payload: bytes, seq: int, timestamp_ms: int) -> bytes:
    """Build a NAAD TCP packet. Matches firmware/transport/tcp_client.c format."""
    MAGIC   = bytes([0xAD, 0x9A])
    VERSION = 0x01
    CODEC   = 0x01  # IMA-ADPCM
    RATE    = 16000
    CH      = 1

    header = struct.pack(
        '>BIIBBHBH',        # format: version(B) seq(I) ts(I) codec(B) ... len(H)
        VERSION,
        seq & 0xFFFFFFFF,
        timestamp_ms & 0xFFFFFFFF,
        CODEC,
        RATE >> 8,          # high byte of 16000
        RATE & 0xFF,        # low byte of 16000
        CH,
        len(adpcm_payload)
    )
    # Actually let's do it properly with fixed struct:
    header = struct.pack('>BIIBHBH',
        VERSION,            # 1 byte
        seq,                # 4 bytes
        timestamp_ms,       # 4 bytes
        CODEC,              # 1 byte
        RATE,               # 2 bytes
        CH,                 # 1 byte
        len(adpcm_payload)  # 2 bytes  → total 15 bytes
    )
    return MAGIC + header + adpcm_payload

def stream_wav_to_server(wav_path: str, host: str = '127.0.0.1', port: int = 5555):
    """
    Load a WAV file, encode as ADPCM, and stream it as TCP packets
    exactly as the ESP32 firmware would. Perfect for testing the server
    without any hardware.
    """
    print(f"[mock_stream] Loading: {wav_path}")
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    pcm = (audio * 32767).astype(np.int16)
    total_samples = len(pcm)
    print(f"[mock_stream] Duration: {total_samples/16000:.2f}s  ({total_samples} samples)")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print(f"[mock_stream] Connected to {host}:{port}")

    CHUNK = 320   # 20ms @ 16kHz — same as DMA frame size on ESP32
    pred, idx = 0, 0
    seq = 0
    t_start = time.time()

    for i in range(0, total_samples - CHUNK, CHUNK):
        chunk_pcm = pcm[i : i + CHUNK]
        adpcm_bytes, pred, idx = adpcm_encode(chunk_pcm.tolist(), pred, idx)

        timestamp_ms = int((time.time() - t_start) * 1000)
        packet = build_packet(adpcm_bytes, seq, timestamp_ms)
        sock.sendall(packet)
        seq += 1

        # Simulate real-time pacing (ESP32 produces 20ms frames)
        time.sleep(0.018)  # slightly under 20ms to account for processing time

    # Send 500ms of silence to trigger endpointing
    silence = np.zeros(CHUNK, dtype=np.int16)
    for _ in range(25):  # 25 * 20ms = 500ms silence
        adpcm_bytes, pred, idx = adpcm_encode(silence.tolist(), pred, idx)
        packet = build_packet(adpcm_bytes, seq, int((time.time() - t_start) * 1000))
        sock.sendall(packet)
        seq += 1
        time.sleep(0.018)

    sock.close()
    print(f"[mock_stream] Done. Sent {seq} packets.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock ESP32 ADPCM stream")
    parser.add_argument("--wav",  default=None, help="WAV file to stream")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    if args.wav:
        stream_wav_to_server(args.wav, args.host, args.port)
    else:
        # Stream 3 seconds of random noise if no WAV provided
        print("[mock_stream] No WAV given — streaming 3s of random noise")
        noise = (np.random.randn(48000) * 3000).astype(np.int16)

        import soundfile as sf, tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp = f.name
        sf.write(tmp, noise, 16000, subtype='PCM_16')
        stream_wav_to_server(tmp, args.host, args.port)
        os.unlink(tmp)
```

**S3 — how to test your server without any hardware:**
```bash
# Terminal 1: start your server
python server/receiver.py

# Terminal 2: run the mock stream with a real keyword WAV
python mocks/adpcm_mock_stream.py --wav ml/data/train/keyword/spk01_keyword_room_0.5m_001.wav

# You should see in Terminal 1:
# [receiver] Client connected: 127.0.0.1
# [receiver] Endpoint detected at 2340ms
# [ASR] Transcript: 'naad activate'  (or whatever your keyword is)
```

---

### `mocks/kws_mock.py`
**WHO USES THIS:** S3 — while waiting for S2's TFLM model to be ready

```python
# mocks/kws_mock.py
# Owner: S3
# Purpose: Simulates keyword trigger events so S3 can test the
#          pre-roll + streaming + server pipeline end-to-end.
#
# In firmware: replace the real KWS task with this Python trigger
#              when testing the server independently.

import time
import threading
import random

class MockKWSTrigger:
    """
    Simulates keyword trigger events from the firmware side.
    S3 uses this to test: trigger → pre-roll flush → ADPCM stream → server
    
    Replace with: real TFLM inference once S2 merges kws_infer.h
    """

    def __init__(self, trigger_interval_s: float = 10.0, jitter_s: float = 2.0):
        """
        trigger_interval_s: fire a fake trigger every N seconds
        jitter_s: add random jitter to make it realistic
        """
        self._interval = trigger_interval_s
        self._jitter   = jitter_s
        self._callback = None
        self._running  = False
        self._thread   = None

    def set_trigger_callback(self, callback):
        """callback(score: float) is called on trigger."""
        self._callback = callback

    def start(self):
        """Start firing fake triggers in background."""
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[kws_mock] Mock KWS trigger started — firing every ~10s")

    def stop(self):
        self._running = False

    def fire_now(self, score: float = 0.92):
        """Manually fire a trigger (useful for testing)."""
        if self._callback:
            self._callback(score)

    def _loop(self):
        while self._running:
            wait = self._interval + random.uniform(-self._jitter, self._jitter)
            time.sleep(max(1.0, wait))
            if self._running and self._callback:
                fake_score = random.uniform(0.87, 0.99)
                print(f"[kws_mock] TRIGGER! Score={fake_score:.3f}")
                self._callback(fake_score)


# ============================================================
# SWAP GUIDE for S3:
# ============================================================
# While S2 is training (server-side simulation of triggers):
#   kws = MockKWSTrigger()
#   kws.set_trigger_callback(on_trigger)
#   kws.start()
#
# Once S2 merges firmware/kws_tflm/ files:
#   Remove the mock. Trigger comes from the real ESP32 TCP stream header.
#   Server detects trigger from a "trigger=true" field in packet header.
# ============================================================
```

---

## INTEGRATION LADDER (When to swap mocks)

```
                HOUR 0          HOUR 6          HOUR 16         HOUR 24+
S1 output:      [skeleton]  →   [mfcc DONE]  →  [C MFCC DONE]  →  [FreeRTOS task]
                                    ↓
S2 swaps:    mfcc_mock.py  →  mfcc_reference.py
S2 output:      [dataset]   →   [training]   →  [INT8 DONE]    →  [TFLM on board]
                                                      ↓
S3 swaps:    kws_mock.py   →   real trigger from ESP32

S3 output:   [server done] →  [mock test OK] →  [mock+ADPCM OK] → [real E2E]
```

**The golden rule:** Each person's server/Python side is always ahead of the hardware.
If you're blocked on hardware, your software should already be 100% done using mocks.

---

## S3 SPECIFIC — SERVER WORKS ON DAY 0

S3 has the least dependencies of anyone. Your full server can be tested end-to-end
on your laptop alone, without S1 or S2 doing anything.

```bash
# Terminal 1: start server
python server/receiver.py

# Terminal 2: use mock stream with any WAV you have (even a random file)
python mocks/adpcm_mock_stream.py --wav any_recording.wav

# Expected output in Terminal 1:
# [receiver] Client connected: 127.0.0.1:12345
# [receiver] Frame 0: seq=0 ts=0ms payload=160B
# [receiver] Frame 1: seq=1 ts=18ms payload=160B
# ...
# [receiver] Endpoint detected at 2240ms
# [ASR] Transcript: '...'
```

**S3 milestone check (no hardware, no S1, no S2 needed):**
- [ ] receiver.py accepts connection from adpcm_mock_stream.py
- [ ] adpcm_decoder.py decodes frames back to recognizable audio
- [ ] endpoint.py fires after 300ms of silence
- [ ] asr.py returns a transcript
- [ ] dashboard.py shows live metrics

Once all 5 are checked off → **S3's software is done.** Wait for S2's TFLM → flash firmware → test real E2E.

---

## COMMUNICATION PROTOCOL (WhatsApp / Signal)

Use these exact message templates to unblock each other:

**S1 → ALL when MFCC is ready:**
```
[S1 → ALL] mfcc_reference.py merged to main.
   git pull origin main
   S2: swap your import from mocks.mfcc_mock to ml.scripts.mfcc_reference
   Branch: s1/dsp, commit: [paste hash]
```

**S2 → S3 when TFLM files are ready:**
```
[S2 → S3] TFLM files merged to main.
   firmware/kws_tflm/kws_model_data.cc + kws_infer.h ready.
   git pull origin main
   Model size: [X]KB, arena needed: ~80KB
   Branch: s2/kws, commit: [paste hash]
```

**S3 → ALL when server is end-to-end tested:**
```
[S3 → ALL] Server pipeline fully tested with mock stream.
   receiver.py + adpcm_decoder.py + endpoint.py + asr.py all working.
   Waiting for: S2 TFLM + ESP32 hardware.
   Branch: s3/server, commit: [paste hash]
```

---

*PROJECT NAAD · SIH26172 · Parallel Working Guide v1.0*
*"Use mocks so nobody waits. Swap mocks when real code lands. Prove the server works before the hardware arrives."*
