# Owner: S1
# Module: Micro-VAD — Python Reference
# Interface OUTPUT: bool per 20ms frame (True=speech, False=silence)
# Interface INPUT:  int16 numpy array (320 samples = 20ms @ 16kHz)
# Last validated: 2026-09-03 (pending commit — see docs/test_results/mfcc_validation.md)

import numpy as np

# ============================================================
# CONSTANTS — These mirror the C implementation exactly
# VAD_FRAME_SAMPLES = 320 (20ms @ 16kHz — different from MFCC frame!)
# ============================================================
VAD_FRAME_SAMPLES = 320          # 20ms @ 16kHz
STE_THRESHOLD     = 120_000_000  # Tune from your room! Start here.
ZCR_LOW           = 10
ZCR_HIGH          = 200
HANGOVER_FRAMES   = 8             # 8 * 20ms = 160ms


def compute_ste(frame: np.ndarray) -> int:
    """
    Short-Time Energy: sum of squared int16 samples.
    Input:  (320,) int16
    Output: int (always >= 0)
    """
    return int(np.sum(frame.astype(np.int64) ** 2))


def compute_zcr(frame: np.ndarray) -> int:
    """
    Zero-Crossing Rate: count of sign changes.
    Input:  (320,) int16
    Output: int
    """
    signs = np.sign(frame.astype(np.int32))
    signs[signs == 0] = 1   # Treat zero as positive
    return int(np.sum(np.abs(np.diff(signs)) // 2))


class MicroVAD:
    """
    Stateful VAD — same logic as the C implementation.
    Use one instance per audio stream.
    """

    def __init__(self):
        self._hangover = 0

    def update(self, frame: np.ndarray) -> bool:
        """
        Process one 20ms frame.
        Input:  (320,) int16 numpy array
        Output: bool (True = speech active)

        Raises:
            TypeError:  frame is not a numpy ndarray, or not dtype int16
            ValueError: frame is not exactly VAD_FRAME_SAMPLES (320) samples
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must be a numpy ndarray, got {type(frame).__name__}")
        if frame.dtype != np.int16:
            raise TypeError(f"frame must be dtype int16, got {frame.dtype}")
        if frame.ndim != 1:
            raise ValueError(f"frame must be 1-D, got shape {frame.shape}")
        if len(frame) != VAD_FRAME_SAMPLES:
            raise ValueError(f"frame must be {VAD_FRAME_SAMPLES} samples, got {len(frame)}")

        energy_ok = compute_ste(frame) > STE_THRESHOLD
        zcr       = compute_zcr(frame)
        zcr_ok    = ZCR_LOW < zcr < ZCR_HIGH

        if energy_ok and zcr_ok:
            self._hangover = HANGOVER_FRAMES
            return True
        if self._hangover > 0:
            self._hangover -= 1
            return True
        return False

    def reset(self):
        self._hangover = 0


def run_vad_on_file(wav_path: str) -> list:
    """
    Run VAD on a WAV file.
    Returns: list of (frame_index, is_speech) tuples
    """
    import librosa
    audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    pcm = (audio * 32767).astype(np.int16)  # float32 -> int16

    vad = MicroVAD()
    results = []
    for i in range(0, len(pcm) - VAD_FRAME_SAMPLES, VAD_FRAME_SAMPLES):
        frame = pcm[i : i + VAD_FRAME_SAMPLES]
        speech = vad.update(frame)
        results.append((i // VAD_FRAME_SAMPLES, speech))
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vad_reference.py path/to/audio.wav")
        sys.exit(1)
    results = run_vad_on_file(sys.argv[1])
    speech_frames  = sum(1 for _, s in results if s)
    silence_frames = sum(1 for _, s in results if not s)
    print(f"Total frames:   {len(results)}")
    print(f"Speech frames:  {speech_frames} ({100*speech_frames/len(results):.1f}%)")
    print(f"Silence frames: {silence_frames}")
