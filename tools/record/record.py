# Owner: S2
# Module: Data Collection CLI
# Usage: python record.py --speaker spk01 --type keyword --env room --dist 0.5m --count 50
# Last validated: [date, commit]

import argparse
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time

SAMPLE_RATE = 16000
DURATION    = 1.5      # seconds per recording (keyword fits in 1.5s)
CHANNELS    = 1

def record_sample(output_path: str, duration: float = DURATION):
    """Record a single sample and save as WAV."""
    print(f"  Recording in 3... ", end="", flush=True)
    time.sleep(1)
    print("2... ", end="", flush=True); time.sleep(1)
    print("1... ", end="", flush=True); time.sleep(1)
    print("GO!", flush=True)

    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype='int16')
    sd.wait()
    sf.write(output_path, audio, SAMPLE_RATE, subtype='PCM_16')
    print(f"  Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="NAAD Data Collection Tool")
    parser.add_argument("--speaker",  required=True, help="Speaker ID, e.g. spk01")
    parser.add_argument("--type",     required=True, choices=["keyword", "negative"], help="Sample type")
    parser.add_argument("--env",      required=True, help="Environment: room, fan, hallway, outdoor")
    parser.add_argument("--dist",     required=True, help="Distance: 0.5m, 1m, 2m")
    parser.add_argument("--count",    type=int, default=20, help="Number of samples to record")
    parser.add_argument("--split",    default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    out_dir = os.path.join("ml", "data", args.split, args.type)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== NAAD Data Collection ===")
    print(f"Speaker: {args.speaker} | Type: {args.type} | Env: {args.env} | Dist: {args.dist}")
    print(f"Recording {args.count} samples to: {out_dir}")
    print(f"Say the keyword each time you hear 'GO!'")
    print("=" * 40)

    for i in range(1, args.count + 1):
        print(f"\n[{i}/{args.count}]")
        fname = f"{args.speaker}_{args.type}_{args.env}_{args.dist}_{i:03d}.wav"
        fpath = os.path.join(out_dir, fname)
        record_sample(fpath)
        input("Press Enter for next sample (or Ctrl+C to stop)...")

    print(f"\nDone! Recorded {args.count} samples.")

if __name__ == "__main__":
    main()
