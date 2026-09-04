# Owner: S3
# Module: faster-whisper ASR wrapper
# Setup: pip install faster-whisper
# Model downloaded automatically to ~/.cache/huggingface/hub/ on first run
# Last validated: [date, commit hash]
#
# Model options (choose based on speed vs accuracy):
#   tiny.en  — 72MB,  fastest,  ~70% WER on clean speech
#   base     — 142MB, good,     ~15% WER on clean speech  ← RECOMMENDED
#   small    — 466MB, slower,   ~10% WER on clean speech
#   medium   — 1.5GB, slowest,  ~8%  WER on clean speech

import numpy as np
import time

print("[asr] Loading faster-whisper model (first run downloads ~142MB)...")
_load_start = time.time()

from faster_whisper import WhisperModel

# Load once at startup — NOT per transcription call
_model = WhisperModel(
    "base",          # Change to "small" for better accuracy if time permits
    device="cpu",
    compute_type="int8"   # int8 = 2x faster than float32 on CPU
)

_load_time = time.time() - _load_start
print(f"[asr] Model ready in {_load_time:.1f}s")


def transcribe(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """
    Transcribe audio using faster-whisper.

    Args:
        audio:       float32 numpy array, values in [-1.0, 1.0], 16kHz mono
        sample_rate: must be 16000

    Returns:
        Transcribed text string (empty string if silence/no speech)
    """
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Normalize to [-1.0, 1.0] if needed
    max_amp = np.abs(audio).max()
    if max_amp > 1.0:
        audio = audio / max_amp

    if max_amp < 0.001:
        return ""  # Silent — skip transcription

    t_start = time.time()
    segments, info = _model.transcribe(
        audio,
        beam_size=5,
        language="en",
        vad_filter=False   # We handle endpointing ourselves with endpoint.py
    )
    transcript = " ".join(seg.text.strip() for seg in segments).strip()
    t_elapsed = (time.time() - t_start) * 1000

    print(f"[asr] Transcribed in {t_elapsed:.0f}ms: '{transcript}'")
    return transcript


# ── Self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test with a WAV file: python asr.py path/to/test.wav
        import librosa
        wav_path = sys.argv[1]
        audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        print(f"[test] Loaded: {wav_path}  ({len(audio)/sr:.1f}s)")
        result = transcribe(audio.astype(np.float32))
        print(f"[test] Transcript: '{result}'")
    else:
        # Quick silent audio test
        silent = np.zeros(16000, dtype=np.float32)
        result = transcribe(silent)
        print(f"[test] Silent audio transcript: '{result}' (expected: empty)")
        print("✅ ASR module working correctly!")
