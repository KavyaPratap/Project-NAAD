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

    # Fixed struct: version(B) seq(I) ts(I) codec(B) rate(H) channels(B) payload_len(H)
    # Total header = 1+4+4+1+2+1+2 = 15 bytes
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
