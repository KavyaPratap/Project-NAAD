# Owner: S2
# Tool: Dataset Organizer
# Purpose: Split raw_wav/ and real_data_wav/ into ml/data/{train,val,test}/{keyword,negative}/
#
# Sources:
#   raw_wav/positive/  → keyword WAVs (TTS, 50 files)
#   raw_wav/negative/  → negative WAVs (TTS, 1296 files)
#   real_data_wav/     → real human keyword WAVs (31 files)
#
# Strategy:
#   keyword (50 TTS + 31 real = 81 total):
#     train → 65 (80%)
#     val   → 8  (10%)
#     test  → 8  (10%)
#
#   negative (1296 TTS):
#     train → 1036 (80%)
#     val   → 130  (10%)
#     test  → 130  (10%)
#
# Usage: python tools/organize_dataset.py [--dry-run]

import os
import glob
import shutil
import random
import argparse

SEED = 42

def copy_files(files, dest_dir, prefix='', dry_run=False):
    os.makedirs(dest_dir, exist_ok=True)
    for i, src in enumerate(files):
        basename = os.path.basename(src)
        if prefix:
            name, ext = os.path.splitext(basename)
            dst_name = f"{prefix}_{name}{ext}"
        else:
            dst_name = basename
        dst = os.path.join(dest_dir, dst_name)
        if dry_run:
            print(f"  [DRY] {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
    return len(files)

def split_list(items, train_ratio=0.8, val_ratio=0.1):
    """Split list into train/val/test."""
    n = len(items)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]

def main():
    parser = argparse.ArgumentParser(description='Organize dataset into ml/data splits')
    parser.add_argument('--dry-run', action='store_true', help='Preview without copying')
    args = parser.parse_args()

    random.seed(SEED)

    # ── KEYWORD FILES ──────────────────────────────────────────────
    tts_keyword   = sorted(glob.glob('raw_wav/positive/*.wav'))
    real_keyword  = sorted(glob.glob('real_data_wav/*.wav'))
    all_keyword   = tts_keyword + real_keyword
    random.shuffle(all_keyword)

    kw_train, kw_val, kw_test = split_list(all_keyword)

    print(f"\n=== KEYWORD ({len(all_keyword)} total) ===")
    print(f"  train: {len(kw_train)}, val: {len(kw_val)}, test: {len(kw_test)}")

    copy_files(kw_train, 'ml/data/train/keyword', dry_run=args.dry_run)
    copy_files(kw_val,   'ml/data/val/keyword',   dry_run=args.dry_run)
    copy_files(kw_test,  'ml/data/test/keyword',  dry_run=args.dry_run)

    # ── NEGATIVE FILES ─────────────────────────────────────────────
    all_negative = sorted(glob.glob('raw_wav/negative/*.wav'))
    random.shuffle(all_negative)

    neg_train, neg_val, neg_test = split_list(all_negative)

    print(f"\n=== NEGATIVE ({len(all_negative)} total) ===")
    print(f"  train: {len(neg_train)}, val: {len(neg_val)}, test: {len(neg_test)}")

    copy_files(neg_train, 'ml/data/train/negative', dry_run=args.dry_run)
    copy_files(neg_val,   'ml/data/val/negative',   dry_run=args.dry_run)
    copy_files(neg_test,  'ml/data/test/negative',  dry_run=args.dry_run)

    # ── SUMMARY ────────────────────────────────────────────────────
    mode = '[DRY RUN]' if args.dry_run else '[DONE]'
    print(f"\n{mode} Dataset organized into ml/data/")
    print(f"\n  Train: {len(kw_train)} keyword + {len(neg_train)} negative = {len(kw_train)+len(neg_train)}")
    print(f"  Val  : {len(kw_val)} keyword + {len(neg_val)} negative = {len(kw_val)+len(neg_val)}")
    print(f"  Test : {len(kw_test)} keyword + {len(neg_test)} negative = {len(kw_test)+len(neg_test)}")

    if not args.dry_run:
        print("\nNext step: python ml/scripts/augment.py")

if __name__ == '__main__':
    main()
