/* Owner: S1
 * Module: MFCC Feature Extraction (Float Reference Port)
 * Behavioral reference: ml/scripts/mfcc_reference.py — this C port must
 * reproduce that pipeline's numerical output within +/-1 int8 unit.
 *
 * Interface INPUT:  int16_t audio_in[audio_len] — raw 16-bit PCM @ 16kHz,
 *                    mono, NOT pre-emphasized (this module applies
 *                    pre-emphasis internally, exactly like load_audio() +
 *                    apply_preemphasis() do in the Python reference).
 * Interface OUTPUT: int8_t out_features[49][13] (stacked frames)
 * Last tested: 2026-09-03 (pending commit — see docs/test_results/mfcc_validation.md)
 *
 * Deviation from the literal s1.md snippet: mfcc_compute_matrix() returns
 * mfcc_status_t instead of void. This task's explicit error-handling
 * requirement ("NULL input / NULL output / insufficient length must return
 * a clear error status rather than crashing") cannot be satisfied by a void
 * return — a status enum is the minimal change that satisfies it while
 * keeping the same 3 parameters and the same output semantics for valid
 * input.
 */
#ifndef MFCC_H
#define MFCC_H

#include <stdint.h>

/* ============================================================
 * CONSTANTS — Frozen. Mirror ml/scripts/mfcc_reference.py exactly.
 * Do not change without notifying S2 and S3.
 * ============================================================ */
#define MFCC_SAMPLE_RATE     16000
#define MFCC_FRAME_SAMPLES   400        /* 25ms @ 16kHz */
#define MFCC_HOP_SAMPLES     160        /* 10ms @ 16kHz */
#define MFCC_FFT_SIZE        512
#define MFCC_N_MEL           26
#define MFCC_N_COEF          13
#define MFCC_N_FRAMES        49
#define MFCC_PREEMPH_ALPHA   0.97f
#define MFCC_MEL_FMIN        80.0
#define MFCC_MEL_FMAX        8000.0
#define MFCC_LOG_EPSILON     1e-6f
#define MFCC_INT8_SCALE      32.0f

/* Largest audio_len (in int16 samples) that can influence the output.
 * The Python reference computes ALL frames then truncates to the first
 * MFCC_N_FRAMES (mfcc[:target_frames]) — so only the first
 * (MFCC_N_FRAMES-1)*MFCC_HOP_SAMPLES + MFCC_FRAME_SAMPLES samples of the
 * input ever reach the output. Anything beyond this index is provably
 * irrelevant and is never read. */
#define MFCC_MAX_INPUT_SAMPLES ((MFCC_N_FRAMES - 1) * MFCC_HOP_SAMPLES + MFCC_FRAME_SAMPLES)

typedef enum {
    MFCC_OK                       = 0,
    MFCC_ERR_NULL_INPUT           = -1,
    MFCC_ERR_NULL_OUTPUT          = -2,
    MFCC_ERR_INSUFFICIENT_LENGTH  = -3,  /* audio_len < MFCC_FRAME_SAMPLES: cannot form even one frame */
} mfcc_status_t;

#ifdef __cplusplus
extern "C" {
#endif

/* Full MFCC pipeline: pre-emphasis -> framing -> Hamming -> 512-pt FFT ->
 * power spectrum -> 26-band Mel filterbank -> log -> DCT-II (13 coeffs) ->
 * pad/trim to 49 frames -> per-feature mean/std normalize -> scale x32.0
 * -> clamp [-128,127] -> int8.
 *
 * On error, out_features (if non-NULL) is zero-filled and a non-OK status
 * is returned; the caller must check the return value rather than assuming
 * a zero matrix means silence.
 *
 * NOTE: uses internal static working buffers — NOT reentrant / not safe to
 * call concurrently from multiple tasks. Call from a single DSP task only
 * (matches the single Core-1 dsp_task design in s1.md).
 */
mfcc_status_t mfcc_compute_matrix(
    const int16_t *audio_in,   /* full audio buffer */
    int            audio_len,  /* number of int16 samples in audio_in */
    int8_t         out_features[MFCC_N_FRAMES][MFCC_N_COEF]
);

#ifdef __cplusplus
}
#endif

#endif /* MFCC_H */
