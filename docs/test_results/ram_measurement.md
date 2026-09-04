# S3 RAM Measurement Results
# Owner: S3
# Date: [fill in after measurement]
# Commit: [fill in after measurement]

## How to Measure

Run firmware with all tasks active (I2S + DSP + KWS + transport + telemetry).
Wait 60 seconds, then read UART output for TELEMETRY REPORT lines.

```bash
# On your laptop, read ESP32 UART (replace /dev/ttyUSB0 with your port):
idf.py -p /dev/ttyUSB0 monitor
# OR:
python -m serial.tools.miniterm /dev/ttyUSB0 115200
```

Look for lines like:
```
[TELEMETRY] RAM free:         XXXXX bytes  (XX KB)
[TELEMETRY] RAM min ever:     XXXXX bytes  (XX KB)
[TELEMETRY] Largest block:    XXXXX bytes  (XX KB)
```

## Results (fill in after measurement)

| Metric | Value | Target |
|--------|-------|--------|
| RAM free (steady state) | ??? KB | > 50 KB |
| RAM minimum ever | ??? KB | > 30 KB |
| Largest contiguous block | ??? KB | > 20 KB |
| TFLM arena used (S2) | ??? KB | < 96 KB |
| AudioCapture stack HWM | ??? words | > 256 |
| DSP task stack HWM | ??? words | > 1024 |
| Transport task stack HWM | ??? words | > 512 |

## Notes
- [ ] Measured under load: streaming + inference running simultaneously
- [ ] Measured after 60s uptime (initial allocations settled)
- [ ] No heap corruption detected (esp_heap_check passes)
