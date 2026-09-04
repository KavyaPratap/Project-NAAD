/* Owner: S1 + Integration
 * Module: DSP Task — FreeRTOS Core-1 integration of Micro-VAD + MFCC + KWS Inference
 */
#include "micro_vad.h"
#include "../mfcc/mfcc.h"
#include "../kws_tflm/kws_infer.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <stdio.h>
#include <string.h>
#include <stddef.h>

extern void audio_i2s_read_blocking(int16_t *out, size_t num_samples);
extern void notify_kws_trigger(float score);

#define KWS_DSP_TASK_NAME        "kws_dsp_task"
#define KWS_DSP_TASK_STACK_WORDS 12288
#define KWS_DSP_TASK_PRIORITY    5
#define KWS_DSP_TASK_CORE_ID     1

static int16_t s_vad_frame[VAD_FRAME_SAMPLES];
static int16_t s_mfcc_accum[MFCC_MAX_INPUT_SAMPLES];
static size_t  s_mfcc_fill_count = 0;
static int8_t  s_mfcc_out[MFCC_N_FRAMES][MFCC_N_COEF];

static void kws_dsp_task(void *pvParameters) {
    (void)pvParameters;

    micro_vad_reset();
    s_mfcc_fill_count = 0;

    for (;;) {
        /* 1. Acquire one 20ms VAD frame */
        audio_i2s_read_blocking(s_vad_frame, VAD_FRAME_SAMPLES);

        /* 2. Run VAD */
        bool is_speech = micro_vad_update(s_vad_frame);
        if (is_speech) {
            printf("[DSP][VAD] speech=1\n");
        }

        /* 3. Accumulate audio into MFCC window */
        size_t space_remaining = MFCC_MAX_INPUT_SAMPLES - s_mfcc_fill_count;
        size_t copy_count = (VAD_FRAME_SAMPLES < space_remaining) ? VAD_FRAME_SAMPLES : space_remaining;
        memcpy(&s_mfcc_accum[s_mfcc_fill_count], s_vad_frame, copy_count * sizeof(int16_t));
        s_mfcc_fill_count += copy_count;

        /* 4. Once window full, run MFCC + KWS inference */
        if (s_mfcc_fill_count >= MFCC_MAX_INPUT_SAMPLES) {
            mfcc_status_t status = mfcc_compute_matrix(s_mfcc_accum, (int)MFCC_MAX_INPUT_SAMPLES, s_mfcc_out);

            if (status == MFCC_OK) {
                /* Run KWS model inference on device */
                float score = kws_infer((const int8_t (*)[13])s_mfcc_out);
                printf("[DSP][KWS] score=%.3f\n", score);

                if (kws_is_triggered(score)) {
                    printf("[DSP][KWS] TRIGGER! Notifying transport task...\n");
                    notify_kws_trigger(score);
                }
            } else {
                printf("[DSP][MFCC] status=ERROR code=%d\n", (int)status);
            }

            s_mfcc_fill_count = 0;
        }
    }
}

void dsp_task_start(void) {
    xTaskCreatePinnedToCore(
        kws_dsp_task,
        KWS_DSP_TASK_NAME,
        KWS_DSP_TASK_STACK_WORDS,
        NULL,
        KWS_DSP_TASK_PRIORITY,
        NULL,
        KWS_DSP_TASK_CORE_ID
    );
}
