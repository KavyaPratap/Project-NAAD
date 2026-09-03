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
