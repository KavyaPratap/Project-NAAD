/* Owner: S2
 * Module: KWS Inference on TFLM
 * Interface INPUT:  int8_t mfcc[49][13] — from S1's C MFCC
 * Interface OUTPUT: float keyword_score [0.0, 1.0]
 * Last tested: [date, firmware commit]
 */
#pragma once
#include <stdint.h>
#include <stdbool.h>

#define KWS_ARENA_BYTES    (96 * 1024)
#define KWS_THRESHOLD      0.85f
#define KWS_SMOOTH_FRAMES  3

/**
 * Initialize TFLM interpreter (call once at boot).
 * Allocates tensor arena, loads model, registers ops.
 */
void kws_init(void);

/**
 * Run inference on one MFCC matrix.
 * @param mfcc  Pointer to int8_t[49][13] feature matrix from S1's MFCC pipeline
 * @return keyword probability in [0.0, 1.0]
 */
float kws_infer(const int8_t mfcc[49][13]);

/**
 * Returns true if smoothed score exceeds threshold (hysteresis applied).
 * Call after kws_infer(). Applies KWS_SMOOTH_FRAMES averaging.
 * @param score  Raw score from kws_infer()
 */
bool kws_is_triggered(float score);

/**
 * Reset trigger state (call after trigger is handled by S3's transport layer).
 */
void kws_reset_trigger(void);
