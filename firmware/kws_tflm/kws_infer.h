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
#define KWS_THRESHOLD      0.70f   // 70% confidence threshold
#define KWS_SMOOTH_FRAMES  1

#ifdef __cplusplus
extern "C" {
#endif

void kws_init(void);
float kws_infer(const int8_t mfcc[49][13]);
bool kws_is_triggered(float score);
void kws_reset_trigger(void);

#ifdef __cplusplus
}
#endif
