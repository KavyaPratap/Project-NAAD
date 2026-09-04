# S3 E2E Latency Measurement Results
# Owner: S3
# Date: [fill in after measurement]
# Commit: [fill in after measurement]

## What to Measure

**E2E latency = time from keyword utterance end → transcript appears on server**

## How to Measure

### Method 1: Firmware-side (precise)
1. In `telemetry.c`: `telemetry_record_trigger()` logs timestamp at keyword trigger
2. Server: log timestamp when transcript is printed
3. E2E = server_transcript_time - firmware_trigger_time

### Method 2: Simple (good enough for demo)
1. Clap hands near mic + start stopwatch
2. Stop when transcript appears on server terminal
3. Repeat 10 times, take average

## Results (fill in after measurement)

| Stage | Latency | Notes |
|-------|---------|-------|
| KWS trigger → first TCP packet | ??? ms | From `telemetry.c` logs |
| TCP transmission (20ms frame) | ~20 ms | Network bound |
| ADPCM decode (per frame) | < 1 ms | Python, fast |
| Endpointing | ~300 ms | 15 frames silence |
| faster-whisper transcribe | ??? ms | From `asr.py` logs |
| **Total E2E** | **??? ms** | **Target < 2000ms** |

## Test runs

| Run | Keyword | Transcript | E2E ms | Correct? |
|-----|---------|-----------|--------|---------|
| 1 | naad | [fill] | [fill] | [Y/N] |
| 2 | naad | [fill] | [fill] | [Y/N] |
| 3 | naad | [fill] | [fill] | [Y/N] |
| 4 | naad | [fill] | [fill] | [Y/N] |
| 5 | naad | [fill] | [fill] | [Y/N] |
| **Average** | | | **??? ms** | |
