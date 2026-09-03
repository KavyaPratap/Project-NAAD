# MFCC and VAD Validation

Owner: S1 — DSP Lead
Status: pending commit (not yet committed/pushed)

All numbers in this document were produced by actually executing the
implementations and comparing outputs element-by-element. No result below
was estimated, rounded away, or assumed.

---

## 1. Scope

This document validates the S1 DSP deliverables against the frozen
behavioral contract in `s1.md`:

- `ml/scripts/mfcc_reference.py` — Python MFCC reference (canonical)
- `firmware/mfcc/mfcc.c` / `mfcc.h` — C MFCC port
- `ml/scripts/vad_reference.py` — Python Micro-VAD reference (canonical)
- `firmware/micro_vad/micro_vad.c` / `micro_vad.h` — C Micro-VAD port

It does not cover `dsp_task.c` (not yet implemented) or S2/S3 integration.

---

## 2. Test Environment

- OS: Windows 11 (MINGW64_NT-10.0-26200, MSYS2/Git Bash shell)
- Python: 3.13.15 (project-local `.venv`, created for this validation work
  since no numpy/scipy/librosa were available in any system Python)
- Python packages: numpy 2.5.2, scipy 1.18.1, librosa 1.0.0, soundfile 0.14.0
- C compiler: `gcc.exe (Rev2, Built by MSYS2 project) 14.2.0`
- Sample rate: 16000 Hz throughout
- Input format: 16-bit signed mono PCM
- Deterministic seeds: `numpy.random.default_rng(42)` (MFCC tone+noise test,
  VAD noise tests), `default_rng(7)` (VAD synthetic speech-like jitter),
  `default_rng(123)` (MFCC low-amplitude noise test)
- No ESP-IDF / PlatformIO project exists anywhere in this repository at the
  time of this validation (confirmed by directory search — no
  `CMakeLists.txt`, `platformio.ini`, or `sdkconfig`).

---

## 3. Frozen DSP Constants

Confirmed present and unmodified in both the Python and C source files at
the time of this validation.

**MFCC** (`ml/scripts/mfcc_reference.py`, `firmware/mfcc/mfcc.h`):

| Constant | Value |
|---|---|
| Sample rate | 16000 Hz |
| Input format | 16-bit mono PCM |
| Frame | 400 samples (25ms) |
| Hop | 160 samples (10ms) |
| FFT size | 512 |
| Mel banks | 26 |
| Mel range | 80–8000 Hz |
| DCT | DCT-II, norm='ortho' |
| MFCC coefficients | 13 |
| Output frames | 49 |
| Pre-emphasis alpha | 0.97 |
| Log epsilon | 1e-6 |
| Quantization scale | 32.0 |
| int8 clamp | [-128, 127] |

**VAD** (`ml/scripts/vad_reference.py`, `firmware/micro_vad/micro_vad.h`):

| Constant | Value |
|---|---|
| Sample rate | 16000 Hz |
| Frame | 320 samples (20ms) |
| STE threshold | 120,000,000 |
| ZCR low | 10 |
| ZCR high | 200 |
| Hangover | 8 frames (160ms) |

No constants were changed to produce any result in this document.

---

## 4. Python MFCC Reference

`extract_mfcc(source, target_frames=49, quantize=True)` is the canonical
function. Pipeline: load audio (int16 PCM ÷ 32768.0 → float32 via
`librosa.load`) → pre-emphasis → framing → Hamming window → 512-pt
zero-padded FFT → power spectrum → 26-band Mel filterbank → log(+1e-6) →
DCT-II (ortho, 13 coeffs) → pad/trim to 49 frames → per-feature mean/std
normalize → ×32.0 → clamp → int8. This was implemented and tested in Task 1;
this document validates the C port against it, not the Python code itself.

---

## 5. C MFCC Validation

### 5.1 Test methodology

1. Generate deterministic float32 test signals in Python (fixed seeds where
   randomness is used).
2. Convert to int16 PCM, write to (a) a `.wav` file and (b) a raw `.bin` of
   the identical int16 samples.
3. Run the actual `extract_mfcc(wav_path)` → save int8 output to a `.bin`.
4. Compile `firmware/mfcc/mfcc.c` with a temporary host-side test harness
   (`tools/test_mfcc.c`, deleted after use) that reads the same raw int16
   `.bin` and calls the real `mfcc_compute_matrix()` → save int8 output to
   a `.bin`.
5. Load both int8 outputs in Python and compare every one of the 637
   elements (49×13).

This guarantees C and Python process byte-identical PCM input — not just
similar-sounding audio.

### 5.2 Test vectors

| Name | Description | Samples |
|---|---|---|
| `task1_style_synth` | 220Hz+880Hz tones + Gaussian noise (seed=42) | 16000 |
| `sine_440hz` | Pure 440Hz sine, no noise | 16000 |
| `mixed_freq` | Sum of 220/440/880/1500Hz sines, no noise | 16000 |
| `low_amp_noise` | Gaussian noise, std=0.02 (seed=123) | 16000 |

### 5.3 Results

Measured by loading both `.bin` outputs as `int8`, reshaping to (49,13),
and computing `abs(python_int32 - c_int32)` over all 637 elements.

**Test: task1_style_synth**
- shape: Python = (49,13), C = (49,13)
- dtype: Python = int8, C = int8
- max_abs_diff = 1
- mean_abs_diff = 0.00157
- different_elements = 1 / 637
- elements_with_diff_1 = 1
- diff_greater_than_1 = 0
- Result: PASS (max_abs_diff 1 <= 1)

**Test: sine_440hz**
- shape: Python = (49,13), C = (49,13)
- dtype: Python = int8, C = int8
- max_abs_diff = 0
- mean_abs_diff = 0.0
- different_elements = 0 / 637
- elements_with_diff_1 = 0
- diff_greater_than_1 = 0
- Result: PASS (exact match)

**Test: mixed_freq**
- shape: Python = (49,13), C = (49,13)
- dtype: Python = int8, C = int8
- max_abs_diff = 0
- mean_abs_diff = 0.0
- different_elements = 0 / 637
- elements_with_diff_1 = 0
- diff_greater_than_1 = 0
- Result: PASS (exact match)

**Test: low_amp_noise**
- shape: Python = (49,13), C = (49,13)
- dtype: Python = int8, C = int8
- max_abs_diff = 0
- mean_abs_diff = 0.0
- different_elements = 0 / 637
- elements_with_diff_1 = 0
- diff_greater_than_1 = 0
- Result: PASS (exact match)

### 5.4 Acceptance criterion

**Rule: PASS if max_abs_diff <= 1.**

Measured max_abs_diff was 1 for `task1_style_synth` and 0 for the other
three tests. All four tests satisfy max_abs_diff <= 1. **Therefore: PASS**
for all 4 tests actually run. This does not constitute proof that every
possible input satisfies the criterion — only these 4 deterministic
vectors were tested.

---

## 6. Python vs C VAD Validation

### 6.1 Test methodology

Five 320-sample int16 test frames were generated in Python with fixed
seeds, dumped as raw `.bin` files, and fed identically to:
- Python `compute_ste()` / `compute_zcr()` / `MicroVAD.update()`
  (`ml/scripts/vad_reference.py`)
- C `compute_ste()` / `compute_zcr()` / `micro_vad_update()`, compiled via a
  temporary harness (`tools/test_micro_vad.c`, deleted after use) against
  the real `firmware/micro_vad/micro_vad.c`.

Frame generation: `silence` = all-zero. `noise_low/medium/high` = Gaussian
noise with std 150/600/4000 (`default_rng(42)`, generated in that order
from one RNG stream). `speech_like` = a **synthetic** (not real recorded
speech) 150Hz+harmonics tone with added broadband jitter noise
(std=1000, `default_rng(7)`) — modeling the breathiness/glottal-pulse
sharpness real voiced speech has; a pure clean harmonic tone without jitter
was tested earlier (Task 2) and measured ZCR=5, below `ZCR_LOW`, and did
**not** trigger the VAD.

### 6.2 Frame-level results

| Frame | Py STE | C STE | Py ZCR | C ZCR | Py VAD | C VAD | Match |
|---|---|---|---|---|---|---|---|
| silence | 0 | 0 | 0 | 0 | False | false | Yes |
| noise_low (std=150) | 6,266,845 | 6266845 | 137 | 137 | False | false | Yes |
| noise_medium (std=600) | 118,416,156 | 118416156 | 162 | 162 | False | false | Yes |
| noise_high (std=4000) | 5,443,212,413 | 5443212413 | 151 | 151 | True | true | Yes |
| speech_like (synthetic) | 12,763,315,120 | 12763315120 | 25 | 25 | True | true | Yes |

**Known limitation, not hidden:** the `noise_high` row shows loud broadband
Gaussian noise (std=4000, well within int16 range) crossing **both**
`energy_ok` and `zcr_ok` and being classified as speech by both the Python
and C implementation. This is an inherent property of the frozen STE+ZCR
algorithm at these frozen thresholds — white noise naturally produces a ZCR
in the 10–200 range, so sufficiently loud broadband noise is
indistinguishable from speech under this algorithm. This is not a
Python/C mismatch (both agree exactly) — it is a characteristic of the
algorithm itself.

### 6.3 Hangover validation

Sequence: `speech_like` trigger frame, followed by 11 `silence` frames.

- Python: `[True, True, True, True, True, True, True, True, True, False, False, False]`
- C: `[True, True, True, True, True, True, True, True, True, False, False, False]`

Both sequences are identical: element 0 is the trigger frame result
(True), elements 1–8 are hangover-held True (8 frames = `HANGOVER_FRAMES`),
elements 9–11 are False after the hangover counter reaches zero. **Match: Yes.**

### 6.4 Reset validation

- Python: trigger frame → `True`; after `reset()`, next silence frame → `False`
- C: trigger frame → `true`; after `micro_vad_reset()`, next silence frame → `false`

**Match: Yes.** `reset()` correctly clears the pending hangover rather than
letting it count down.

### 6.5 NULL / invalid-input boundary test

- C: `micro_vad_update(NULL)` → returns `false`, does not crash.
- Python: `MicroVAD.update(None)` → raises `TypeError: frame must be a
  numpy ndarray, got NoneType` (Python's type system rejects this at a
  different layer than C's pointer check — there is no behavioral
  equivalence claim here beyond "neither crashes uncontrolled").

---

## 7. Error Handling (C MFCC)

Measured via a temporary harness calling `mfcc_compute_matrix()` directly
(deleted after use). Exact status codes observed:

| Input | Observed status | Enum value |
|---|---|---|
| `audio_in = NULL`, `audio_len = 16000` | -1 | `MFCC_ERR_NULL_INPUT` |
| `out_features = NULL` | -2 | `MFCC_ERR_NULL_OUTPUT` |
| `audio_len = 100` (< 400) | -3 | `MFCC_ERR_INSUFFICIENT_LENGTH` |
| `audio_len = 400` (exact minimum) | 0 | `MFCC_OK` |
| `audio_len = 0` | -3 | `MFCC_ERR_INSUFFICIENT_LENGTH` |
| `audio_len = -5` | -3 | `MFCC_ERR_INSUFFICIENT_LENGTH` |

For the `NULL`/insufficient-length cases, `out_features` was pre-poisoned
with `0x7F` bytes before the call and observed to be zero-filled (`0`)
afterward, confirming the function does not leave the caller's buffer in an
undefined state on error.

---

## 8. Compilation Results

Commands actually executed (host-side; no ESP-IDF/PlatformIO project
exists in this repository, so **ESP32-S3 cross-compilation was not
performed and ESP32-S3 hardware execution was not performed** — only
host-side GCC validation, stated explicitly per this task's requirement):

```
gcc -std=c99 -Wall -Wextra -Wpedantic -Wconversion -c firmware/mfcc/mfcc.c -o mfcc.o -Ifirmware/mfcc -lm
gcc -std=c99 -Wall -Wextra -Wpedantic firmware/mfcc/mfcc.c tools/test_mfcc.c -o test_mfcc.exe -lm
gcc -std=c99 -Wall -Wextra -Wpedantic -Wconversion firmware/micro_vad/micro_vad.c tools/test_micro_vad.c -o test_micro_vad.exe
```

All three commands returned exit code 0 with no warnings and no errors
printed to stdout/stderr.

---

## 9. Limitations

- **No ESP32-S3 hardware or ESP-IDF toolchain validation.** Everything in
  this document is host-machine (Windows/MSYS2 GCC) validation only.
- **FFT implementation.** `firmware/mfcc/mfcc.c` uses a portable radix-2
  Cooley-Tukey FFT, not esp-dsp's `dsps_fft2r_fc32()` (s1.md's suggestion),
  because no ESP-IDF project exists in this repo to provide esp-dsp.
- **VAD thresholds are untuned defaults.** `STE_THRESHOLD=120,000,000` and
  `ZCR_LOW/HIGH=10/200` are the frozen starting values from s1.md, not
  calibrated against real room/microphone noise.
- **All test audio is synthetic.** No real recorded speech or room-noise
  WAV files exist anywhere in this repository at the time of this
  validation; every signal used above is deterministically generated.
- **Limited test-vector count.** 4 MFCC signals and 5 VAD frames were
  tested. This is not exhaustive coverage of the input space.
- **`mfcc_compute_matrix()` is not reentrant** (uses internal static
  working buffers) — documented in `mfcc.h`, relevant for future
  `dsp_task.c` integration on a single Core-1 task.

---

## 10. Final S1 Validation Status

| Deliverable | Validated against | Result |
|---|---|---|
| C MFCC (`firmware/mfcc/mfcc.c`) vs Python reference | 4 deterministic signals, all 637 output elements each | max_abs_diff = 1, 0, 0, 0 → **PASS** (criterion: <=1) |
| C VAD (`firmware/micro_vad/micro_vad.c`) vs Python reference | 5 deterministic frames + hangover + reset sequences | STE, ZCR, and VAD decision identical in every case tested → **PASS** |
| C MFCC error handling | 6 boundary conditions | All returned the documented status code, no crash → **PASS** |
| ESP32-S3 firmware/hardware validation | — | **Not performed** — no ESP-IDF/PlatformIO project exists in this repo |

This status reflects only the specific inputs measured above. It is not a
claim of general correctness across all possible audio inputs.
