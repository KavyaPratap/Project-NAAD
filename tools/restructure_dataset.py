import os
import shutil

base_dir = r"c:\Users\priya\OneDrive\Desktop\Project-NAAD"
kws_dir = os.path.join(base_dir, "naad_kws")

os.makedirs(kws_dir, exist_ok=True)
print(f"Created/Verified: {kws_dir}")

folders_to_move = [
    "raw",
    "raw_wav",
    "voice_tests",
    "noise",
    "augmented",
    "final"
]

for folder in folders_to_move:
    src = os.path.join(base_dir, folder)
    dst = os.path.join(kws_dir, folder)
    if os.path.exists(src):
        if not os.path.exists(dst):
            shutil.move(src, dst)
            print(f"Moved: {folder} -> naad_kws/{folder}")
        else:
            print(f"Destination already exists: naad_kws/{folder}")

# Handle real dataset:
real_pos = os.path.join(kws_dir, "real", "positive")
real_neg = os.path.join(kws_dir, "real", "negative")
os.makedirs(real_pos, exist_ok=True)
os.makedirs(real_neg, exist_ok=True)
print("Created: naad_kws/real/positive and naad_kws/real/negative")

real_data_src = os.path.join(base_dir, "real_data_wav")
if os.path.exists(real_data_src):
    for fname in os.listdir(real_data_src):
        fpath = os.path.join(real_data_src, fname)
        if os.path.isfile(fpath):
            shutil.move(fpath, os.path.join(real_pos, fname))
    try:
        os.rmdir(real_data_src)
        print("Moved all files from real_data_wav -> naad_kws/real/positive/ and removed real_data_wav")
    except Exception as e:
        print(f"Note: real_data_wav directory cleanup: {e}")
