# Owner: S2
# Module: Data Augmentation
# Input: raw WAV file path
# Output: list of augmented np.ndarray (float32, 16kHz)
# Last validated: [date, commit]

import os
import sys
import glob
import numpy as np
import librosa

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

SAMPLE_RATE = 16000

def add_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise at target SNR in dB."""
    rms_signal = np.sqrt(np.mean(audio ** 2)) + 1e-9
    rms_noise  = np.sqrt(np.mean(noise ** 2)) + 1e-9
    target_rms = rms_signal / (10 ** (snr_db / 20))
    scaled_noise = noise * (target_rms / rms_noise)
    mixed = audio + scaled_noise[:len(audio)]
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)

def add_reverb(audio: np.ndarray, rt60: float = 0.3) -> np.ndarray:
    """Simulate room reverb using pyroomacoustics."""
    try:
        import pyroomacoustics as pra
        room_dim = [5, 4, 3]
        room = pra.ShoeBox(room_dim, fs=SAMPLE_RATE, materials=pra.Material(0.2), max_order=15)
        room.add_source([1.5, 1.5, 1.5], signal=audio)
        room.add_microphone([2.5, 2.0, 1.5])
        room.simulate()
        reverbed = room.mic_array.signals[0, :len(audio)]
        return np.clip(reverbed / (np.abs(reverbed).max() + 1e-9), -1.0, 1.0).astype(np.float32)
    except ImportError:
        print("pyroomacoustics not installed — skipping reverb")
        return audio

def random_gain(audio: np.ndarray, low_db: float = -6, high_db: float = 6) -> np.ndarray:
    """Apply random gain."""
    db = np.random.uniform(low_db, high_db)
    return np.clip(audio * (10 ** (db / 20)), -1.0, 1.0).astype(np.float32)

def time_shift(audio: np.ndarray, max_ms: float = 20) -> np.ndarray:
    """Randomly shift audio by up to max_ms milliseconds."""
    max_samples = int(max_ms * SAMPLE_RATE / 1000)
    shift = np.random.randint(-max_samples, max_samples)
    return np.roll(audio, shift).astype(np.float32)

def augment_sample(audio: np.ndarray, noise_files: list, n_augments: int = 3) -> list:
    """
    Generate n_augments augmented versions of one audio sample.
    Returns: list of np.ndarray float32
    """
    results = []
    for _ in range(n_augments):
        aug = audio.copy()

        # Random gain always
        aug = random_gain(aug)

        # 50% chance of noise
        if noise_files and np.random.rand() > 0.5:
            noise_path = np.random.choice(noise_files)
            noise, _ = librosa.load(noise_path, sr=SAMPLE_RATE, mono=True)
            if len(noise) >= len(aug):
                start = np.random.randint(0, len(noise) - len(aug))
                noise = noise[start:start + len(aug)]
            else:
                noise = np.tile(noise, int(np.ceil(len(aug) / len(noise))))[:len(aug)]
            snr = np.random.uniform(5, 20)
            aug = add_noise(aug, noise, snr)

        # 40% chance of reverb
        if np.random.rand() > 0.6:
            rt60 = np.random.uniform(0.1, 0.5)
            aug = add_reverb(aug, rt60)

        # Time shift
        aug = time_shift(aug)

        results.append(aug)
    return results


def build_augmented_dataset(
    data_dir: str = "ml/data",
    noise_dir: str = "ml/data/noise",
    n_augments: int = 3
) -> dict:
    """
    Load all WAVs, augment, extract MFCC features using mfcc_reference.py.

    ============================================================
    DEPENDENCY STATUS:
    [ ] Using MOCK MFCC (S1 not done yet)  ← check this box once swapped
    [x] Using REAL MFCC from S1
    To swap: change the import below, nothing else
    ============================================================

    Returns: dict with keys 'train', 'val', 'test', each (X, y) numpy arrays
    """
    # While S1 is working:
    from mocks.mfcc_mock import extract_mfcc

    # Once S1 merges mfcc_reference.py to main (S1 will tell you on WhatsApp):
    # from ml.scripts.mfcc_reference import extract_mfcc   ← uncomment this
    # from mocks.mfcc_mock import extract_mfcc             ← comment out above

    noise_files = glob.glob(os.path.join(noise_dir, "*.wav")) if os.path.exists(noise_dir) else []
    if not noise_files and os.path.exists("naad_kws/noise"):
        noise_files = glob.glob("naad_kws/noise/*.wav")
    print(f"Loaded {len(noise_files)} background noise files for augmentation.")

    # Per-class augmentation counts for train only.
    # keyword: 64 files x (1+15) = 1024  |  negative: 1036 files x (1+3) = 4144
    # Then down-sample negatives to match keyword count -> balanced 50/50.
    AUGMENTS = {
        "keyword":  15,   # heavy augmentation to fix class imbalance
        "negative": 3,
    }

    def is_hard_confuser(filename: str) -> bool:
        base = os.path.basename(filename).lower()
        confusers = ['naa', 'naak', 'naath', 'naam', 'naal', 'naav', 'naan', 'naag', 'daad', 'baad', 'chaad', 'yaad']
        for c in confusers:
            if base.startswith(c) or f"_{c}_" in base or f"_{c}." in base:
                return True
        return False

    all_splits = {}
    for split in ["train", "val", "test"]:
        X_list, y_list, is_hard_list = [], [], []
        for label, class_name in [(1, "keyword"), (0, "negative")]:
            split_dir = os.path.join(data_dir, split, class_name)
            if not os.path.exists(split_dir):
                continue
            wav_files = glob.glob(os.path.join(split_dir, "*.wav"))
            print(f"[{split}] {class_name}: {len(wav_files)} files")
            for wav_path in wav_files:
                audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
                hard = (class_name == "negative" and is_hard_confuser(wav_path))

                # Original sample
                mfcc = extract_mfcc(audio, target_frames=49, quantize=True)
                X_list.append(mfcc)
                y_list.append(label)
                is_hard_list.append(hard)

                # Augmented samples (train only)
                if split == "train":
                    if class_name == "keyword":
                        n_aug = 15
                    elif hard:
                        n_aug = 12   # Heavy augmentation for hard phonetic confusers (naak, naath, etc.)
                    else:
                        n_aug = 2    # Light augmentation for general ambient/negatives
                        
                    if n_aug > 0:
                        augmented = augment_sample(audio, noise_files, n_aug)
                        for aug_audio in augmented:
                            aug_mfcc = extract_mfcc(aug_audio, target_frames=49, quantize=True)
                            X_list.append(aug_mfcc)
                            y_list.append(label)
                            is_hard_list.append(hard)

        if X_list:
            X = np.array(X_list, dtype=np.float32)[:, :, :, np.newaxis]  # Add channel dim
            y = np.array(y_list, dtype=np.int32)
            is_hard = np.array(is_hard_list, dtype=bool)

            # Balance train set with Hard Negative Mining:
            # 100% of hard confusers are kept; remaining negative quota is filled from general negatives.
            if split == "train":
                kw_idx  = np.where(y == 1)[0]
                neg_hard_idx = np.where((y == 0) & (is_hard == True))[0]
                neg_soft_idx = np.where((y == 0) & (is_hard == False))[0]

                target_neg_count = len(kw_idx)
                print(f"[{split}] Hard confusers available: {len(neg_hard_idx)} | Target per class: {target_neg_count}")

                np.random.seed(42)
                if len(neg_hard_idx) >= target_neg_count:
                    neg_sel = np.random.choice(neg_hard_idx, target_neg_count, replace=False)
                else:
                    needed_soft = target_neg_count - len(neg_hard_idx)
                    soft_sel = np.random.choice(neg_soft_idx, min(needed_soft, len(neg_soft_idx)), replace=False)
                    neg_sel = np.concatenate([neg_hard_idx, soft_sel])

                idx = np.concatenate([kw_idx, neg_sel])
                np.random.shuffle(idx)
                X, y = X[idx], y[idx]
                print(f"[{split}] Balanced with {len(neg_hard_idx)} hard confusers preserved: {len(X)} total samples")

            all_splits[split] = (X, y)
            print(f"[{split}] Final: {len(X)} samples (keyword={int(sum(y))}, negative={int(len(y)-sum(y))})")

    return all_splits


if __name__ == "__main__":
    splits = build_augmented_dataset()
    # Save for use in training
    for split, (X, y) in splits.items():
        np.save(f"ml/data/{split}_X.npy", X)
        np.save(f"ml/data/{split}_y.npy", y)
        print(f"Saved {split}: X={X.shape}, y={y.shape}")
