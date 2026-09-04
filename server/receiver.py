# Owner: S3
# Module: TCP Frame Receiver + ASR Pipeline
# Run: python receiver.py
# This is the main server process. Keep it running while ESP32 is on.
# Last validated: [date, commit hash]

import socket
import struct
import threading
import time
import numpy as np

from adpcm_decoder import AdpcmDecoder
from endpoint import Endpointer
from asr import transcribe

# ── Server config ──────────────────────────────────────────────
SERVER_PORT = 5555
MAGIC       = bytes([0xAD, 0x9A])

# Header layout (bytes after magic):
#   version(1) + seq(4) + timestamp_ms(4) + codec(1) + rate(2) + ch(1) + payload_len(2) = 15 bytes
HEADER_FMT  = ">BIIBHBH"   # big-endian: B=uint8, I=uint32, H=uint16
HEADER_SIZE = 15            # bytes after magic


# ── Helper: exact recv ─────────────────────────────────────────
def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes, blocking. Raises ConnectionError on disconnect."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed")
        data += chunk
    return data


def find_magic(sock: socket.socket) -> None:
    """Scan byte-by-byte until magic 0xAD9A is found (resync on partial packet)."""
    buf = b"\x00\x00"
    while buf != MAGIC:
        b = sock.recv(1)
        if not b:
            raise ConnectionError("Socket closed during magic search")
        buf = buf[1:] + b


# ── Per-client handler ─────────────────────────────────────────
def handle_client(conn: socket.socket, addr: tuple) -> None:
    print(f"\n[server] ✅ Client connected: {addr}")

    decoder    = AdpcmDecoder()
    endpointer = Endpointer()
    pcm_buffer = []
    seq_expected = 0
    utterance_count = 0

    t_connect = time.time()

    try:
        while True:
            # 1. Find frame magic bytes (handles resync if bytes are lost)
            find_magic(conn)

            # 2. Read fixed header (15 bytes after magic)
            header_raw = recv_exact(conn, HEADER_SIZE)
            version, seq, timestamp_ms, codec, sample_rate, channels, payload_len = \
                struct.unpack(HEADER_FMT, header_raw)

            # 3. Validate header
            if version != 1 or codec != 1:
                print(f"[server] ⚠️  Bad header: version={version} codec={codec} — skipping")
                continue

            if seq != seq_expected and seq_expected != 0:
                lost = seq - seq_expected
                print(f"[server] ⚠️  Sequence gap: expected={seq_expected} got={seq} (lost ~{lost} packets)")
            seq_expected = seq + 1

            # 4. Read ADPCM payload
            payload = recv_exact(conn, payload_len)

            # 5. Decode ADPCM → PCM
            pcm_samples = decoder.decode(payload)
            pcm_buffer.extend(pcm_samples)

            if seq == 0:
                print(f"[server] 🎙️  Audio stream started from ESP32 mic...", flush=True)
            elif seq % 20 == 0:
                print(f"[server] 📡 Receiving live audio... {seq * 20}ms ({seq} packets)", flush=True)

            # 6. Check for utterance endpoint
            if endpointer.update(pcm_samples):
                utterance_count += 1
                duration_ms = endpointer.total_frames * 20
                print(f"\n[server] 🎯 Utterance #{utterance_count} complete "
                      f"({duration_ms}ms, {len(pcm_buffer)} samples)")

                # Convert to float32 for ASR
                audio = np.array(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe
                t_asr_start = time.time()
                transcript = transcribe(audio)
                t_asr_ms = (time.time() - t_asr_start) * 1000

                # E2E latency: time from first frame to transcript
                e2e_ms = (time.time() - t_connect) * 1000

                print(f"[server] 📝 TRANSCRIPT: '{transcript}'")
                print(f"[server] ⏱️  ASR latency: {t_asr_ms:.0f}ms")
                print(f"[server] ⏱️  E2E latency: {e2e_ms:.0f}ms (since client connected)")

                # Save to results file
                _log_result(utterance_count, transcript, len(pcm_buffer), t_asr_ms)

                # Reset for next utterance
                pcm_buffer.clear()
                decoder.reset()
                endpointer.reset()
                t_connect = time.time()  # reset E2E timer

    except (ConnectionError, OSError) as e:
        print(f"[server] Client {addr} stream ended / disconnected: {e}", flush=True)
        # Process any remaining audio in buffer if endpointer didn't trigger
        if len(pcm_buffer) >= 1600:  # at least 100ms
            utterance_count += 1
            duration_ms = len(pcm_buffer) // 16
            print(f"\n[server] 🎯 Flushing remaining stream #{utterance_count} ({duration_ms}ms, {len(pcm_buffer)} samples)", flush=True)
            audio = np.array(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            t_asr_start = time.time()
            transcript = transcribe(audio)
            t_asr_ms = (time.time() - t_asr_start) * 1000
            e2e_ms = (time.time() - t_connect) * 1000
            print(f"[server] 📝 TRANSCRIPT: '{transcript}'", flush=True)
            print(f"[server] ⏱️  ASR latency: {t_asr_ms:.0f}ms", flush=True)
            print(f"[server] ⏱️  E2E latency: {e2e_ms:.0f}ms", flush=True)
            _log_result(utterance_count, transcript, len(pcm_buffer), t_asr_ms)
            pcm_buffer.clear()
    except struct.error as e:
        print(f"[server] ❌ Struct unpack error: {e}", flush=True)
    except Exception as e:
        import traceback
        print(f"[server] ❌ UNEXPECTED ERROR in handler: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    finally:
        conn.close()
        print(f"[server] Connection closed. Total utterances: {utterance_count}", flush=True)
        print(f"[server] Waiting for next ESP32 connection...", flush=True)


def _log_result(count: int, transcript: str, samples: int, asr_ms: float) -> None:
    """Append result to docs/test_results/ for demo evidence."""
    import os
    os.makedirs("../docs/test_results", exist_ok=True)
    with open("../docs/test_results/asr_wer.md", "a") as f:
        if count == 1:
            f.write("\n## Live Test Results\n")
            f.write("| # | Transcript | Samples | ASR latency |\n")
            f.write("|---|-----------|---------|-------------|\n")
        f.write(f"| {count} | {transcript or '(empty)'} | {samples} | {asr_ms:.0f}ms |\n")


# ── Main server loop ───────────────────────────────────────────
def run_server() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", SERVER_PORT))
    sock.listen(5)

    print(f"[server] 🚀 Project NAAD ASR Server")
    print(f"[server] Listening on port {SERVER_PORT}...")
    print(f"[server] Waiting for ESP32 to connect...")
    print(f"[server] Press Ctrl+C to stop\n")

    while True:
        try:
            conn, addr = sock.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
        except KeyboardInterrupt:
            print("\n[server] Shutting down...")
            break
    sock.close()


if __name__ == "__main__":
    run_server()
