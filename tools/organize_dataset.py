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
    tts_keyword   = sorted(glob.glob('naad_kws/raw_wav/positive/*.wav') + glob.glob('raw_wav/positive/*.wav'))
    real_keyword  = sorted(glob.glob('naad_kws/real/positive/*.wav') + glob.glob('real_data_wav/*.wav'))
    
    # Split separately to guarantee speaker/source diversity across train/val/test
    random.shuffle(tts_keyword)
    random.shuffle(real_keyword)
    
    tts_kw_tr, tts_kw_va, tts_kw_te = split_list(tts_keyword)
    real_kw_tr, real_kw_va, real_kw_te = split_list(real_keyword)
    
    total_kw = len(tts_keyword) + len(real_keyword)
    print(f"\n=== KEYWORD ({total_kw} total: {len(tts_keyword)} TTS + {len(real_keyword)} Real) ===")
    print(f"  train: {len(tts_kw_tr) + len(real_kw_tr)} ({len(tts_kw_tr)} TTS, {len(real_kw_tr)} Real)")
    print(f"  val  : {len(tts_kw_va) + len(real_kw_va)} ({len(tts_kw_va)} TTS, {len(real_kw_va)} Real)")
    print(f"  test : {len(tts_kw_te) + len(real_kw_te)} ({len(tts_kw_te)} TTS, {len(real_kw_te)} Real)")

    copy_files(tts_kw_tr, 'ml/data/train/keyword', prefix='tts', dry_run=args.dry_run)
    copy_files(real_kw_tr, 'ml/data/train/keyword', prefix='real', dry_run=args.dry_run)
    
    copy_files(tts_kw_va, 'ml/data/val/keyword', prefix='tts', dry_run=args.dry_run)
    copy_files(real_kw_va, 'ml/data/val/keyword', prefix='real', dry_run=args.dry_run)
    
    copy_files(tts_kw_te, 'ml/data/test/keyword', prefix='tts', dry_run=args.dry_run)
    copy_files(real_kw_te, 'ml/data/test/keyword', prefix='real', dry_run=args.dry_run)

    # ── NEGATIVE FILES ─────────────────────────────────────────────
    tts_negative  = sorted(glob.glob('naad_kws/raw_wav/negative/*.wav') + glob.glob('raw_wav/negative/*.wav'))
    real_negative = sorted(glob.glob('naad_kws/real/negative/*.wav'))

    random.shuffle(tts_negative)
    random.shuffle(real_negative)

    tts_neg_tr, tts_neg_va, tts_neg_te = split_list(tts_negative)
    real_neg_tr, real_neg_va, real_neg_te = split_list(real_negative)

    total_neg = len(tts_negative) + len(real_negative)
    print(f"\n=== NEGATIVE ({total_neg} total: {len(tts_negative)} TTS + {len(real_negative)} Real) ===")
    print(f"  train: {len(tts_neg_tr) + len(real_neg_tr)} ({len(tts_neg_tr)} TTS, {len(real_neg_tr)} Real)")
    print(f"  val  : {len(tts_neg_va) + len(real_neg_va)} ({len(tts_neg_va)} TTS, {len(real_neg_va)} Real)")
    print(f"  test : {len(tts_neg_te) + len(real_neg_te)} ({len(tts_neg_te)} TTS, {len(real_neg_te)} Real)")

    copy_files(tts_neg_tr, 'ml/data/train/negative', prefix='tts', dry_run=args.dry_run)
    copy_files(real_neg_tr, 'ml/data/train/negative', prefix='real', dry_run=args.dry_run)

    copy_files(tts_neg_va, 'ml/data/val/negative', prefix='tts', dry_run=args.dry_run)
    copy_files(real_neg_va, 'ml/data/val/negative', prefix='real', dry_run=args.dry_run)

    copy_files(tts_neg_te, 'ml/data/test/negative', prefix='tts', dry_run=args.dry_run)
    copy_files(real_neg_te, 'ml/data/test/negative', prefix='real', dry_run=args.dry_run)

    # ── SUMMARY ────────────────────────────────────────────────────
    kw_train_total = len(tts_kw_tr) + len(real_kw_tr)
    kw_val_total   = len(tts_kw_va) + len(real_kw_va)
    kw_test_total  = len(tts_kw_te) + len(real_kw_te)

    neg_train_total = len(tts_neg_tr) + len(real_neg_tr)
    neg_val_total   = len(tts_neg_va) + len(real_neg_va)
    neg_test_total  = len(tts_neg_te) + len(real_neg_te)

    mode = '[DRY RUN]' if args.dry_run else '[DONE]'
    print(f"\n{mode} Dataset organized into ml/data/")
    print(f"\n  Train: {kw_train_total} keyword + {neg_train_total} negative = {kw_train_total + neg_train_total}")
    print(f"  Val  : {kw_val_total} keyword + {neg_val_total} negative = {kw_val_total + neg_val_total}")
    print(f"  Test : {kw_test_total} keyword + {neg_test_total} negative = {kw_test_total + neg_test_total}")

    if not args.dry_run:
        print("\nNext step: python ml/scripts/augment.py")

if __name__ == '__main__':
    main()
