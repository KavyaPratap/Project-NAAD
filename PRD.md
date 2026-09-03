# PROJECT NAAD — SIH26172 · ISRO Hardware Track
## Product Requirements Document (PRD) · Team Engineering Bible

> **Purpose:** This document is the single source of truth for all 3 team members. If you are using a different AI assistant, paste this file into your context window FIRST before asking anything. Every decision here is intentional — do not override without team consensus.

---

## 0. GOLDEN RULES (Read Before Anything Else)

| # | Rule |
|---|------|
| 1 | **No orphan code.** Every feature = code + test + measured result + owner. If it's not measured, it doesn't count. |
| 2 | **Offline first.** Prove every algorithm on your PC in Python before porting to ESP32. |
| 3 | **One interface owner.** When two modules touch a boundary, the downstream owner defines the contract, the upstream owner implements to it. |
| 4 | **No redundant implementations.** `mfcc.py` is the reference. `mfcc.c` ports it. Nobody writes a third version. |
| 5 | **Freeze at Hour 24.** No architecture changes after Hour 24 of the hackathon unless a measured failure forces it. |
| 6 | **Targets are not Results.** Replace every "estimated" or "expected" with an actual measured number before the demo. |

---

## 1. PROJECT OVERVIEW

**Project Name:** NAAD (SIH26172)
**Track:** ISRO Hardware Track
**Problem Statement:** Build a custom keyword-spotting system on ESP32-S3 with minimal compute, custom wake word, and local offline ASR streaming.

### One-Slide Architecture
```
INMP441 Mic
   |
   v (I2S / DMA, 16 kHz, 16-bit mono)
PCM Ring Buffer
   |
   v
[STE + ZCR Micro-VAD]  <- Always running, gates KWS
   | speech frame
   v
[MFCC: 49x13 int8]     <- Pre-emphasis -> Hamming -> 512-FFT -> 26 Mel -> log -> DCT
   | feature matrix
   v
[INT8 DS-CNN / TFLite Micro]  <- Keyword score
   | trigger (score + hysteresis)
   v
[Pre-roll + Live PCM]  <- 500ms circular buffer
   |
   v
[IMA-ADPCM Encoder]    <- 16-bit PCM -> 4-bit / 64 kbps payload
   |
   v
[TCP Framed Stream]    <- length-prefix + sequence number
   |
   v (LAN / offline hotspot)
[Server: ADPCM Decode -> Server VAD endpoint -> faster-whisper ASR]
   |
   v
TRANSCRIPT
```

---

## 2. TEAM ROLES & OWNERSHIP

> **We are a 3-person team.** Each person owns a vertical slice end-to-end.

| Member | Role Label | Primary Ownership | Secondary |
|--------|-----------|-------------------|-----------|
| **Person 1 (S1)** | DSP Lead | VAD, MFCC (Python + C), embedded DSP | Assist S2 with feature export |
| **Person 2 (S2)** | KWS / ML Lead | DS-CNN training, INT8, TFLM deployment | Dataset pipeline |
| **Person 3 (S3)** | Server / Integration Lead | ADPCM, TCP server, endpointing, faster-whisper, end-to-end integration | Hardware measurement bridge |

> **IMPORTANT:** We are 3 not 6. The original NAAD guide is for 6. Merge roles as follows:
> - H1 (I2S capture) -> S3 handles hardware bring-up
> - H2 (data collection) -> S2 owns dataset + collection workflow
> - H3 (measurement) -> All three own their module metrics

---

## 3. REPOSITORY STRUCTURE

```
naad/
+-- PRD.md                      <- THIS FILE. Do not move or rename.
+-- learningresources.md        <- Algorithm deep-dives. Read before implementing.
+-- INTERFACES.md               <- All cross-module contracts
+-- LICENSES.md                 <- License audit for all dependencies
|
+-- firmware/                   <- ESP32-S3 ESP-IDF C code
|   +-- audio_i2s/              <- I2S + DMA capture (Owner: S3)
|   +-- micro_vad/              <- STE + ZCR + hangover (Owner: S1 port)
|   +-- mfcc/                   <- Fixed-point DSP (Owner: S1 port)
|   +-- kws_tflm/               <- TFLite Micro inference (Owner: S2)
|   +-- adpcm/                  <- IMA-ADPCM encoder (Owner: S3)
|   +-- transport/              <- TCP framing + Wi-Fi (Owner: S3)
|   +-- telemetry/              <- CPU/RAM/latency logging (Owner: S3)
|
+-- ml/                         <- Training pipeline (Owner: S2)
|   +-- data/
|   |   +-- train/
|   |   +-- val/
|   |   +-- test/
|   +-- notebooks/
|   |   +-- 01_eda.ipynb
|   |   +-- 02_mfcc_reference.ipynb    <- S1 owns this
|   |   +-- 03_training.ipynb          <- S2 owns this
|   |   +-- 04_quantization.ipynb      <- S2 owns this
|   +-- models/
|   |   +-- naad_kws_float.h5
|   |   +-- naad_kws_int8.tflite
|   +-- scripts/
|       +-- mfcc_reference.py          <- THE canonical MFCC (S1 writes, S2 uses)
|       +-- augment.py                 <- Noise/reverb augmentation (S2)
|       +-- train.py                   <- DS-CNN training (S2)
|       +-- evaluate.py                <- Confusion matrix + FAR (S2)
|
+-- server/                     <- Python server (Owner: S3)
|   +-- receiver.py             <- TCP frame parser
|   +-- adpcm_decoder.py        <- IMA-ADPCM decode
|   +-- endpoint.py             <- Server VAD / silence endpointing
|   +-- asr.py                  <- faster-whisper wrapper
|   +-- dashboard.py            <- Rich telemetry display
|
+-- tools/                      <- Shared utilities
|   +-- record/                 <- Data collection CLI (S2)
|   +-- wav_inspect.py          <- Waveform/RMS/clipping checker (S1)
|   +-- metrics.py              <- FAR / latency / WER calculator (S3)
|
+-- docs/
|   +-- architecture/           <- Block diagrams
|   +-- test_results/           <- ALL measured results go here (mandatory)
|   +-- demo/
|
+-- README.md
```

---

## 4. INTERFACE CONTRACTS

> These are the non-negotiable boundaries between modules. If you need to change an interface, ALL owners of connected modules must agree.

### Interface A: MFCC Output Format
```
Shape:     (49, 13) int8 tensor
Frame:     25 ms = 400 samples @ 16 kHz
Hop:       10 ms = 160 samples
FFT:       512-point (zero-padded from 400)
Mel banks: 26 triangular filters
DCT coefs: 13 (DCT-II)
Range:     int8 [-128, 127]
Owner:     S1 defines, S2 consumes
```

### Interface B: KWS Trigger Signal
```
Type:      bool trigger + float32 score
Condition: score > THRESHOLD for N consecutive frames (hysteresis)
Threshold: tunable, start at 0.85
Owner:     S2 defines output, S3 listens via FreeRTOS queue
```

### Interface C: ADPCM Stream Packet Format
```
Field         Bytes   Notes
magic         2       0xAD 0x9A
version       1       0x01
seq_num       4       uint32, wraps
timestamp_ms  4       uint32, ms since boot
codec         1       0x01 = IMA-ADPCM
sample_rate   2       uint16, 16000
channels      1       0x01 = mono
payload_len   2       uint16
payload       N       IMA-ADPCM bytes (predictor+index in first 4 bytes of each frame)
crc8          1       optional debug check
Owner:        S3 defines, firmware implements
```

### Interface D: Dataset File Naming Convention
```
Format: {speakerID}_{keyword|negative}_{env}_{dist}_{take}.wav
Example: spk01_NAAD_room_0.5m_001.wav
         spk02_negative_fan_1m_012.wav
Splits:  train/ val/ test/ (speaker-disjoint - no speaker appears in 2 splits)
Owner:   S2 defines, all members collecting data follow
```

---

## 5. PHASE-BY-PHASE EXECUTION PLAN

### PHASE 0 — Setup (All members, first 2 hours)
- [ ] S1: Clone repo, set up Python env (Python 3.10+, librosa, numpy, scipy, tensorflow)
- [ ] S2: Set up Kaggle/Colab for training (GPU), verify TF 2.x install
- [ ] S3: Flash ESP32-S3 with known-good ESP-IDF blink example, verify toolchain
- [ ] All: Read `learningresources.md` — especially your module section
- [ ] All: Confirm keyword choice (freeze ONE keyword for the sprint)
- [ ] S1: Create `ml/scripts/mfcc_reference.py` — this is the canonical reference

### PHASE 1 — Hardware Bring-up + Offline DSP (Hours 0–6)

**S1 Tasks:**
- [ ] Implement `mfcc_reference.py`:
  - Pre-emphasis alpha=0.97
  - Framing: 400 samples, hop 160
  - Hamming window (precomputed)
  - 512-point FFT (zero-pad)
  - 26 Mel filterbank
  - Log (add epsilon=1e-6)
  - DCT-II -> 13 coefficients
  - Output: (49, 13) float32, then int8-quantized
- [ ] Run on 3 WAV files, save output tensors to `docs/test_results/mfcc_reference_check.npy`
- [ ] Implement Python VAD: STE threshold + ZCR (10-200) + hangover (8 frames)
- [ ] Verify: VAD activates on speech WAV, stays silent on noise WAV

**S2 Tasks:**
- [ ] Set up `tools/record/` CLI for data collection
- [ ] Collect >= 50 keyword samples per speaker (target: >= 500 total)
- [ ] Collect hard negatives (similar-sounding, normal speech, fan noise)
- [ ] Verify speaker-disjoint split: no speaker ID appears in both train and test
- [ ] Run `wav_inspect.py` on all collected audio

**S3 Tasks:**
- [ ] Wire INMP441: 3.3V, GND, BCLK, WS/LRCLK, SD/DOUT
- [ ] Configure I2S: 16 kHz, 16-bit mono, DMA mode
- [ ] Record 5s WAV, copy to PC, inspect waveform
- [ ] Run 10-minute continuous capture, count dropped DMA blocks
- [ ] Document exact working pin config in `firmware/audio_i2s/README.md`

**Phase 1 Done Criteria:**
- MFCC Python reference produces (49, 13) tensor from any 1-second WAV
- VAD correctly gates on speech, off on silence
- ESP32 captures clean 16 kHz audio without drops
- >= 300 labeled WAV files collected

---

### PHASE 2 — KWS Training + Embedded DSP (Hours 6–18)

**S1 Tasks:**
- [ ] Port VAD to C: `firmware/micro_vad/micro_vad.c` + `.h`
  - `int64_t compute_ste(const int16_t *frame, int len)`
  - `int compute_zcr(const int16_t *frame, int len)`
  - `bool micro_vad_update(const int16_t *frame)`
- [ ] Port MFCC to C using esp-dsp FFT: `firmware/mfcc/mfcc.c`
  - Pre-emphasis in Q15 fixed-point
  - Hamming table in flash
  - esp_dsp_fft2r_fc32 or fixed-point variant
  - Mel weights table in flash (26 x 257 or sparse)
  - Integer log2 approximation
  - DCT-II table in flash
- [ ] Cross-validate: same WAV -> Python MFCC vs C MFCC -> difference < 1 int8 unit per cell

**S2 Tasks:**
- [ ] Complete dataset (>= 500 positives, >= 500 negatives)
- [ ] Run augmentation (`augment.py`):
  - Additive noise at SNR in [0, 20] dB
  - Random gain in [-6, +6] dB
  - Room reverb (pyroomacoustics, RT60 in [0.1, 0.8])
  - Time shift +-20ms
- [ ] Extract MFCC from ALL samples using `mfcc_reference.py` (NOT librosa MFCC)
- [ ] Train DS-CNN baseline (see `learningresources.md` for exact architecture)
- [ ] Evaluate: confusion matrix on speaker-disjoint test set
- [ ] Target: recall >= 90%, FAR < 2/hour on 1-hour negative test
- [ ] INT8 quantization with representative calibration tensors (use real MFCC tensors from train set)
- [ ] Save `naad_kws_int8.tflite`, log size

**S3 Tasks:**
- [ ] Build TCP server skeleton: receive bytes, parse length-prefixed frames
- [ ] Implement `adpcm_decoder.py` (IMA-ADPCM state machine)
- [ ] Test round-trip: encode PCM -> ADPCM -> decode -> compare to original
- [ ] Set up faster-whisper: `pip install faster-whisper`, download `tiny.en` or `base`
- [ ] Test: saved WAV file -> faster-whisper -> transcript

**Phase 2 Done Criteria:**
- C MFCC output matches Python reference within +-1 int8 unit
- DS-CNN baseline trained, recall/precision/FAR logged
- INT8 model fits in firmware
- ADPCM round-trip produces intelligible audio
- faster-whisper transcribes offline WAV

---

### PHASE 3 — Embedded KWS + Streaming (Hours 18–30)

**S1 Tasks:**
- [ ] Integrate VAD + MFCC into FreeRTOS task on Core 1
- [ ] Measure: VAD CPU%, MFCC CPU%, latency per frame
- [ ] Log to UART: frame count, VAD decision, MFCC feature shape
- [ ] Fix any MFCC mismatch against Python reference

**S2 Tasks:**
- [ ] Deploy `.tflite` to ESP32-S3: `firmware/kws_tflm/`
  - Register only required ops (Conv2D, DepthwiseConv2D, Mean, FullyConnected, Softmax, Reshape)
  - Start with 96 KB arena, measure actual usage with `arena_used_bytes()`
  - Measure inference time (us) per invocation
- [ ] Implement score smoothing + hysteresis trigger
- [ ] Test on device with real keyword + real negatives
- [ ] Log: trigger events, score, false activations

**S3 Tasks:**
- [ ] Implement IMA-ADPCM encoder in C: `firmware/adpcm/adpcm.c`
- [ ] Implement 500ms pre-roll circular buffer: `firmware/transport/preroll.c`
- [ ] Implement TCP client with length-prefix framing: `firmware/transport/tcp_client.c`
- [ ] On trigger: flush pre-roll + start streaming live PCM -> encode -> send
- [ ] Test: ESP32 trigger -> packets arrive at server -> decode -> intelligible

**Phase 3 Done Criteria:**
- TFLM inference runs on-device, arena size measured
- Keyword detected in real-time on ESP32
- ADPCM stream arrives at server and decodes correctly
- Pre-roll + live audio together cover full keyword utterance

---

### PHASE 4 — End-to-End Integration + Measurement (Hours 30–42)

**ALL HANDS — Integration Steps (in this order):**
1. S3 starts server, confirms TCP listening
2. S1+S3 confirm I2S -> VAD pipeline on device logs frames
3. S2+S1 confirm VAD -> MFCC -> KWS produces scores on device
4. S3 confirms trigger -> ADPCM encoder -> TCP send works
5. Server: receive -> ADPCM decode -> endpoint -> faster-whisper
6. Full end-to-end: speak keyword -> see transcript

**Measurement (each person measures their module, mandatory):**

| Metric | Who | How | Where to log |
|--------|-----|-----|--------------|
| RAM (free heap, min free, largest block) | S3 | `esp_get_free_heap_size()` | `docs/test_results/ram_measurement.md` |
| TFLM tensor arena | S2 | `interpreter.arena_used_bytes()` | same |
| Task stack HWM | S3 | `uxTaskGetStackHighWaterMark()` | same |
| VAD CPU% | S1 | idle task monitor | `docs/test_results/cpu_measurement.md` |
| MFCC latency | S1 | timestamp per frame | same |
| KWS inference time | S2 | timestamp around Invoke() | same |
| FAR | S2 | run 60+ min negative audio | `docs/test_results/far_measurement.md` |
| Keyword recall (TPR) | S2 | 50+ test utterances | same |
| E2E latency | S3 | keyword_end to ASR_complete | `docs/test_results/latency_measurement.md` |
| ASR WER | S3 | fixed test set | `docs/test_results/asr_wer.md` |

---

### PHASE 5 — Freeze + Demo Prep (Hours 42–48)

- [ ] **FREEZE:** No new features. No architecture changes. Bug fixes only.
- [ ] All: update `docs/test_results/` with final measured numbers
- [ ] S3: prepare backup demo (known-good recorded clip + known-good firmware binary)
- [ ] All: 10 full rehearsals of the demo sequence:
  1. Show hardware (ESP32-S3 + INMP441)
  2. Start telemetry dashboard
  3. Speak keyword -> show local trigger + score
  4. Show pre-roll + live stream flowing
  5. Show server decode + endpoint
  6. Show faster-whisper transcript
  7. Repeat under noise
  8. Show measured metrics slide
- [ ] All: Every team member can explain every algorithm (use `learningresources.md`)

---

## 6. CODING STANDARDS

### Python (ml/, server/, tools/)
```python
# Owner: [S1|S2|S3]
# Module: [module name]
# Interface: [what this module inputs/outputs - reference INTERFACES section]
# Last validated: [date, commit hash]
```
- Python 3.10+
- Type hints on all function signatures
- numpy arrays: always specify dtype explicitly (np.float32, np.int8)
- No librosa for MFCC - use mfcc_reference.py only (consistency!)
- Every function has a docstring with input shape, output shape, and units

### C / ESP-IDF (firmware/)
```c
/* Owner: [S1|S2|S3]
 * Module: [module name]
 * Interface: [inputs/outputs]
 * Last tested: [date, firmware commit]
 */
```
- ESP-IDF 5.x, C17
- All audio buffers: int16_t for PCM, int8_t for MFCC
- Fixed-point arithmetic: Q15 (multiply then shift >>15)
- Never use float on Core 0 (I/O path) - keep DSP on Core 1
- ESP_LOGI() for all state transitions (VAD on/off, trigger, packet sent)
- No dynamic allocation after boot: all buffers statically declared

### Git Workflow
```
Branches:
  main       <- only tested, measured, merged code
  s1/dsp     <- S1 working branch
  s2/kws     <- S2 working branch
  s3/server  <- S3 working branch

Commit message format:
  [S1|S2|S3] [module]: short description
  Test result: [pass/fail + metric]

Example:
  [S1] mfcc: fix Mel weight normalization
  Test result: C vs Python max diff = 0 int8 units on test WAV

Rules:
  - Never commit untested code to main
  - Every merge to main must include a test result in the commit
  - Never store raw datasets in the repo
  - PR review: at least one other member reviews before merging to main
```

---

## 7. DEPENDENCY MANIFEST

| Dependency | Version | Purpose | License | Owner |
|------------|---------|---------|---------|-------|
| ESP-IDF | 5.1+ | Firmware framework | Apache 2.0 | S3 |
| esp-dsp | latest | FFT/DSP on ESP32 | Apache 2.0 | S1 |
| TensorFlow Lite Micro | latest | MCU inference | Apache 2.0 | S2 |
| tensorflow / keras | 2.x | Training | Apache 2.0 | S2 |
| faster-whisper | latest | Offline ASR | MIT | S3 |
| numpy | 1.24+ | Numerics | BSD | All |
| librosa | 0.10 | Audio I/O ONLY (not MFCC!) | ISC | S1 |
| audiomentations | latest | Augmentation | MIT | S2 |
| pyroomacoustics | latest | Reverb simulation | MIT | S2 |
| rich | latest | Telemetry dashboard | MIT | S3 |
| pyserial | latest | UART log reading | BSD | S3 |

> librosa is used ONLY for loading WAV files (librosa.load()). MFCC computation always uses mfcc_reference.py.

---

## 8. ENVIRONMENT SETUP

### S1 + S2 Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy librosa tensorflow audiomentations pyroomacoustics jupyter
```

### S3 Server Environment
```bash
python3 -m venv .venv_server
source .venv_server/bin/activate
pip install faster-whisper rich pyserial numpy
```

### S3 Firmware Environment
```bash
# Install ESP-IDF 5.x
. $IDF_PATH/export.sh
idf.py create-project naad_firmware
# Clone esp-dsp as component
cd components && git clone https://github.com/espressif/esp-dsp.git
```

---

## 9. AI ASSISTANT CONSISTENCY PROTOCOL

> If you or your teammate is using a different AI (ChatGPT, Gemini, Copilot etc.), paste this block at the start of every session:

```
PROJECT CONTEXT (paste this before every AI session):

Project: NAAD - SIH26172 ISRO Hardware Track keyword spotting system
Stack: ESP32-S3 + INMP441 mic, ESP-IDF 5.x, TFLite Micro, faster-whisper
Pipeline: I2S/DMA -> STE+ZCR VAD -> MFCC (49x13 int8) -> INT8 DS-CNN -> IMA-ADPCM -> TCP -> ASR

My role: [S1: DSP | S2: KWS/ML | S3: Server+Integration]

Canonical MFCC reference: ml/scripts/mfcc_reference.py (do NOT use librosa MFCC)
Interface contract file: INTERFACES.md
All measured results: docs/test_results/

Current task: [describe your specific task]
```

---

## 10. DEMO SCRIPT (Memorize)

**When judge asks "How does it work?"**
1. Mic -> I2S/DMA -> PCM (16 kHz, 16-bit)
2. STE+ZCR Micro-VAD quickly rejects non-speech
3. MFCC converts speech to 49x13 compact time-frequency features
4. INT8 DS-CNN classifies keyword locally on ESP32-S3
5. Score smoothing + hysteresis confirm trigger
6. Ring buffer pre-roll preserves 500ms before activation
7. IMA-ADPCM compresses 256 kbps -> 64 kbps payload
8. TCP framed stream sent to offline local server
9. Server decodes -> endpoints -> faster-whisper transcript

**When judge asks "Prove it":**
- RAM: measured minimum free heap = [X] bytes under stress
- CPU: measured idle = [X]% during continuous KWS
- FAR: [X] false activations over [Y] hours = [Z]/hour
- Latency: keyword_end -> transcript = [X] ms (each stage timestamped)
- Recall: [X]% on speaker-disjoint test set

---

## 11. 48-HOUR MASTER CHECKLIST

### Phase 0
- [ ] Keyword frozen and agreed by all
- [ ] Repo cloned and branches created
- [ ] All environments set up
- [ ] learningresources.md read by all members

### Phase 1 — Foundation
- [ ] ESP32 boots and flashes
- [ ] INMP441 wired + 16 kHz capture verified
- [ ] mfcc_reference.py outputs (49,13) tensor
- [ ] Python VAD gates correctly on speech/silence
- [ ] >= 300 labeled WAV files collected

### Phase 2 — Algorithms
- [ ] C MFCC matches Python within +-1 int8
- [ ] DS-CNN trained, confusion matrix logged
- [ ] INT8 model converted + size logged
- [ ] ADPCM round-trip verified
- [ ] faster-whisper runs offline

### Phase 3 — Embedded
- [ ] TFLM inference on-device, arena measured
- [ ] Real-time keyword detection working
- [ ] ADPCM stream flows from ESP32 to server
- [ ] Pre-roll preserves keyword start

### Phase 4 — Integration
- [ ] Full end-to-end: speak -> transcript
- [ ] All metrics measured and logged
- [ ] Noisy-room test done

### Phase 5 — Demo
- [ ] Firmware/model/server FROZEN
- [ ] 10 full demo rehearsals done
- [ ] Backup demo ready
- [ ] Measured metrics slide ready
- [ ] Every member can explain every algorithm

---

*PROJECT NAAD . SIH26172 . PRD v1.0 . Team Edition*
*"A feature is done only when it has working code + a test + a logged result + an owner + a known failure mode."*
