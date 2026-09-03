/* Owner: S1
 * Module: Micro-VAD (Voice Activity Detection)
 * Interface INPUT:  int16_t frame[VAD_FRAME_SAMPLES] (20ms @ 16kHz)
 * Interface OUTPUT: bool (true=speech, false=silence)
 * Behavioral reference: ml/scripts/vad_reference.py (MicroVAD class) — this
 * C port must match that Python reference's STE/ZCR/hangover logic exactly.
 * Last tested: 2026-09-03 (pending commit — see docs/test_results/mfcc_validation.md)
 *
 * Caller contract: frame must point to exactly VAD_FRAME_SAMPLES valid
 * int16_t samples. micro_vad_update() only defends against a NULL pointer
 * (returns false without touching hangover state) — it cannot validate
 * buffer length at runtime in C, unlike the Python reference.
 */
#ifndef MICRO_VAD_H
#define MICRO_VAD_H

#include <stdbool.h>
#include <stdint.h>

/* ============================================================
 * CONSTANTS — Frozen. Mirror ml/scripts/vad_reference.py exactly.
 * Do not change without notifying S2 and S3.
 * ============================================================ */
#define VAD_FRAME_SAMPLES  320          /* 20ms @ 16kHz */
#define STE_THRESHOLD      120000000LL
#define ZCR_LOW            10
#define ZCR_HIGH           200
#define HANGOVER_FRAMES    8            /* 8 * 20ms = 160ms */

#ifdef __cplusplus
extern "C" {
#endif

/* Reset VAD state (hangover counter) to idle. Call once at stream start. */
void micro_vad_reset(void);

/* Process one 20ms frame of VAD_FRAME_SAMPLES int16 samples.
 * Returns true if speech is active (directly detected or within hangover). */
bool micro_vad_update(const int16_t *frame);

/* Short-Time Energy: sum of squared samples, int64_t accumulator (no overflow). */
int64_t compute_ste(const int16_t *frame, int len);

/* Zero-Crossing Rate: count of sign changes (zero sample treated as positive). */
int compute_zcr(const int16_t *frame, int len);

#ifdef __cplusplus
}
#endif

#endif /* MICRO_VAD_H */
