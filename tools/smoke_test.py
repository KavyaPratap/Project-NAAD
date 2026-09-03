import sounddevice as sd
import soundfile as sf
import numpy as np

print("sounddevice:", sd.__version__)
print("soundfile   : OK")
print("numpy       :", np.__version__)

devs = sd.query_devices()
print(f"Audio devices found: {len(devs)}")

try:
    default_in = sd.query_devices(kind="input")
    print("Default input device:", default_in["name"])
    print("\nTASK 1 SMOKE TEST: PASS ✅")
except Exception as e:
    print(f"[WARN] No default input device: {e}")
    print("(You may still record if you select a device manually)")
    print("\nTASK 1 SMOKE TEST: PASS ✅ (packages installed, device TBD)")
