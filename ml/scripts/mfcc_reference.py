# Owner: S1
# Module: MFCC Feature Extraction — Python Reference
# Interface OUTPUT: np.ndarray, shape (49, 13), dtype=np.int8
# Interface INPUT:  WAV file path OR np.ndarray float32 at 16kHz
# Last validated: 2026-09-03 (pending commit — see docs/test_results/mfcc_validation.md)

import os
import numpy as np
import librosa
from typing import Union

# ============================================================
# CONSTANTS — These are frozen. Do not change without
# notifying S2 and S3 (they depend on these values).
# ============================================================
SAMPLE_RATE    = 16000
FRAME_SAMPLES  = 400       # 25ms @ 16kHz
HOP_SAMPLES    = 160       # 10ms @ 16kHz
FFT_SIZE       = 512       # Next power-of-2 >= 400
N_MEL          = 26        # Mel filterbank banks
N_MFCC         = 13        # DCT coefficients to keep
PREEMPH_ALPHA  = 0.97
MEL_FMIN       = 80.0      # Hz
MEL_FMAX       = 8000.0    # Hz
LOG_EPSILON    = 1e-6      # Avoid log(0)
MFCC_INT8_SCALE = 32.0     # Scaling factor before int8 clamp


def load_audio(path: str) -> np.ndarray:
    """Load and normalize audio to float32 in [-1, 1] at 16kHz mono."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"WAV file not found: {path}")

    try:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        # librosa/soundfile/audioread raise a variety of exception types
        # (RuntimeError, LibsndfileError, NoBackendError, etc.) for
        # corrupt/unsupported files — normalize to a single clear error.
        raise ValueError(f"Could not load audio file '{path}': {exc}") from exc

    if audio.size == 0:
        raise ValueError(f"Audio file '{path}' contains no samples")

    return audio.astype(np.float32)


def apply_preemphasis(audio: np.ndarray, alpha: float = PREEMPH_ALPHA) -> np.ndarray:
    """
    High-pass filter to boost high frequencies.
    y[n] = x[n] - alpha * x[n-1]
    Input:  (N,) float32
    Output: (N,) float32
    """
    return np.append(audio[0], audio[1:] - alpha * audio[:-1]).astype(np.float32)


def compute_frames(audio: np.ndarray) -> np.ndarray:
    """
    Split audio into overlapping frames.
    Input:  (N,) float32
    Output: (num_frames, FRAME_SAMPLES) float32
    """
    if len(audio) < FRAME_SAMPLES:
        raise ValueError(
            f"Audio too short: {len(audio)} samples, need at least "
            f"{FRAME_SAMPLES} samples ({FRAME_SAMPLES / SAMPLE_RATE * 1000:.0f}ms @ {SAMPLE_RATE}Hz)"
        )
    num_frames = 1 + (len(audio) - FRAME_SAMPLES) // HOP_SAMPLES
    frames = np.zeros((num_frames, FRAME_SAMPLES), dtype=np.float32)
    for i in range(num_frames):
        start = i * HOP_SAMPLES
        frames[i] = audio[start : start + FRAME_SAMPLES]
    return frames


def apply_hamming(frames: np.ndarray) -> np.ndarray:
    """
    Apply Hamming window to each frame.
    Input:  (num_frames, FRAME_SAMPLES) float32
    Output: (num_frames, FRAME_SAMPLES) float32
    """
    window = np.hamming(FRAME_SAMPLES).astype(np.float32)
    return frames * window


def compute_power_spectrum(frames: np.ndarray) -> np.ndarray:
    """
    Zero-pad to FFT_SIZE, compute FFT, return power spectrum.
    Input:  (num_frames, FRAME_SAMPLES) float32
    Output: (num_frames, FFT_SIZE//2 + 1) float32
    """
    padded = np.zeros((frames.shape[0], FFT_SIZE), dtype=np.float32)
    padded[:, :FRAME_SAMPLES] = frames
    spectrum = np.fft.rfft(padded, n=FFT_SIZE)
    power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
    return power


def build_mel_filterbank() -> np.ndarray:
    """
    Build 26 triangular Mel filterbank weights.
    Output: (N_MEL, FFT_SIZE//2 + 1) float32
    This is computed once — in C, this is a table in flash.
    """
    freq_bins = np.linspace(0, SAMPLE_RATE / 2, FFT_SIZE // 2 + 1)

    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mel_min = hz_to_mel(MEL_FMIN)
    mel_max = hz_to_mel(MEL_FMAX)
    mel_points = np.linspace(mel_min, mel_max, N_MEL + 2)
    hz_points = mel_to_hz(mel_points)

    filterbank = np.zeros((N_MEL, FFT_SIZE // 2 + 1), dtype=np.float32)
    for m in range(N_MEL):
        left = hz_points[m]
        center = hz_points[m + 1]
        right = hz_points[m + 2]
        for k, f in enumerate(freq_bins):
            if left <= f <= center:
                filterbank[m, k] = (f - left) / (center - left)
            elif center < f <= right:
                filterbank[m, k] = (right - f) / (right - center)
    return filterbank


# Precompute once at module load
_MEL_FILTERBANK = build_mel_filterbank()


def apply_mel_filterbank(power: np.ndarray) -> np.ndarray:
    """
    Apply Mel filterbank to power spectrum.
    Input:  (num_frames, FFT_SIZE//2+1) float32
    Output: (num_frames, N_MEL) float32
    """
    return np.dot(power, _MEL_FILTERBANK.T).astype(np.float32)


def apply_log(mel_energy: np.ndarray) -> np.ndarray:
    """
    Log compression. Adds epsilon to avoid log(0).
    Input:  (num_frames, N_MEL) float32
    Output: (num_frames, N_MEL) float32
    """
    return np.log(mel_energy + LOG_EPSILON).astype(np.float32)


def apply_dct(log_mel: np.ndarray) -> np.ndarray:
    """
    DCT-II to decorrelate features. Keep first N_MFCC coefficients.
    Input:  (num_frames, N_MEL) float32
    Output: (num_frames, N_MFCC) float32
    """
    from scipy.fft import dct
    return dct(log_mel, type=2, axis=1, norm='ortho')[:, :N_MFCC].astype(np.float32)


def normalize_and_quantize(mfcc_float: np.ndarray) -> np.ndarray:
    """
    Normalize and cast to int8 for TFLM compatibility.
    Input:  (num_frames, N_MFCC) float32
    Output: (num_frames, N_MFCC) int8
    """
    # Normalize to approx [-1, 1] then scale to [-128, 127]
    # Simple mean-std normalization per feature
    mean = mfcc_float.mean(axis=0, keepdims=True)
    std = mfcc_float.std(axis=0, keepdims=True) + 1e-8
    normalized = (mfcc_float - mean) / std
    scaled = np.clip(normalized * MFCC_INT8_SCALE, -128, 127)
    return scaled.astype(np.int8)


def extract_mfcc(
    source: Union[str, np.ndarray],
    target_frames: int = 49,
    quantize: bool = True
) -> np.ndarray:
    """
    Full MFCC pipeline. THE canonical function — S2 calls this.

    Args:
        source:        WAV file path or numpy float32 array at 16kHz
        target_frames: Number of time frames to output (pad/trim to this)
        quantize:      If True, output int8. If False, output float32.

    Returns:
        np.ndarray shape (target_frames, N_MFCC) = (49, 13)
        dtype: int8 if quantize=True, float32 if False

    Raises:
        TypeError:  source is not a str path or np.ndarray
        ValueError: WAV file is missing/corrupt/empty, or audio is
                    shorter than one MFCC frame (400 samples @ 16kHz)
    """
    if isinstance(source, str):
        audio = load_audio(source)
    elif isinstance(source, np.ndarray):
        if source.ndim != 1:
            raise ValueError(f"Audio array must be 1-D (mono), got shape {source.shape}")
        if source.size == 0:
            raise ValueError("Audio array is empty")
        audio = source.astype(np.float32)
    else:
        raise TypeError(
            f"source must be a WAV file path (str) or np.ndarray, got {type(source).__name__}"
        )

    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")

    audio = apply_preemphasis(audio)
    frames = compute_frames(audio)
    frames = apply_hamming(frames)
    power = compute_power_spectrum(frames)
    mel = apply_mel_filterbank(power)
    log_mel = apply_log(mel)
    mfcc = apply_dct(log_mel)

    # Pad or trim to target_frames
    if mfcc.shape[0] < target_frames:
        pad = np.zeros((target_frames - mfcc.shape[0], N_MFCC), dtype=mfcc.dtype)
        mfcc = np.vstack([mfcc, pad])
    else:
        mfcc = mfcc[:target_frames]

    if quantize:
        return normalize_and_quantize(mfcc)
    return mfcc


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python mfcc_reference.py path/to/audio.wav")
        sys.exit(1)
    result = extract_mfcc(sys.argv[1], quantize=True)
    print(f"Output shape: {result.shape}")   # Should be (49, 13)
    print(f"Output dtype: {result.dtype}")   # Should be int8
    print(f"Min: {result.min()}, Max: {result.max()}")
    print(f"First frame: {result[0]}")
    # Save for validation
    np.save("mfcc_test_output.npy", result)
    print("Saved to mfcc_test_output.npy")
