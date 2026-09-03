/* Owner: S1
 * Module: DSP Task — FreeRTOS Core-1 integration of Micro-VAD + MFCC
 * Deliverable per s1.md: "FreeRTOS DSP task | firmware/micro_vad/dsp_task.c
 * | Runs on Core 1, logs VAD+MFCC to UART"
 *
 * ============================================================
 * BUILD STATUS — READ BEFORE ATTEMPTING TO COMPILE
 * ============================================================
 * This repository has NO ESP-IDF or PlatformIO project at the time this
 * file was written (no CMakeLists.txt, platformio.ini, or sdkconfig
 * anywhere in the tree). This file therefore CANNOT be built as-is:
 *   - "freertos/FreeRTOS.h" and "freertos/task.h" are genuine ESP-IDF
 *     headers that do not exist on this host and are not vendored here.
 *     They are NOT stubbed out in this repository, by design (stubbing
 *     them would fake a successful build rather than reflect reality).
 *   - audio_i2s_read_blocking() (declared below) is a documented contract
 *     this file depends on, owned by S3's firmware/audio_i2s/ module.
 *     Only firmware/audio_i2s/README.md exists there right now — no header,
 *     no implementation — so this symbol is left undefined here. A real
 *     build will fail to LINK until S3 implements it, in addition to
 *     needing an actual ESP-IDF build environment to fail to COMPILE.
 * This file is written as genuine, integration-ready FreeRTOS/C source —
 * not a mockup — so that dropping it into a real ESP-IDF component
 * directory (once one exists) requires no rewrite, only wiring.
 * ============================================================
 */
#include "micro_vad.h"
#include "../mfcc/mfcc.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <stdio.h>
#include <string.h>
#include <stddef.h>

/* ------------------------------------------------------------
 * Audio input contract (owned by S3, not yet implemented)
 * ------------------------------------------------------------
 * dsp_task.c needs exactly one thing from the I2S capture layer: a
 * blocking read of `num_samples` int16 PCM samples @ 16kHz mono. This
 * prototype documents that single contract without inventing an I2S/DMA
 * driver body — see firmware/audio_i2s/README.md for the capture module
 * this is expected to come from.
 *
 * Blocking (rather than a non-blocking poll) is deliberate: the DSP task
 * loop below has no artificial vTaskDelay of its own — the real-time
 * pacing for both VAD (20ms/frame) and MFCC accumulation comes entirely
 * from this call blocking until fresh samples actually arrive from the
 * microphone, which is the correct source of truth for audio timing.
 * ------------------------------------------------------------ */
extern void audio_i2s_read_blocking(int16_t *out, size_t num_samples);

/* ------------------------------------------------------------
 * Task configuration
 * ------------------------------------------------------------ */
#define KWS_DSP_TASK_NAME       "kws_dsp_task"
/* Exact value from s1.md's failure-triage table: "ESP32 crash in MFCC |
 * Stack too small — increase kws_dsp_task stack to 12288" — using that
 * value directly rather than inventing a different one. */
#define KWS_DSP_TASK_STACK_WORDS 12288
/* s1.md does not specify a task priority anywhere. Using a moderate
 * placeholder; this should be tuned by whoever integrates this against
 * S3's I2S/networking tasks once those exist. */
#define KWS_DSP_TASK_PRIORITY   5
#define KWS_DSP_TASK_CORE_ID    1   /* Core 1, per s1.md: "Runs on Core 1" */

/* ------------------------------------------------------------
 * Fixed-size static working buffers — no dynamic allocation.
 * ------------------------------------------------------------ */
static int16_t s_vad_frame[VAD_FRAME_SAMPLES];
static int16_t s_mfcc_accum[MFCC_MAX_INPUT_SAMPLES];
static size_t  s_mfcc_fill_count = 0;
static int8_t  s_mfcc_out[MFCC_N_FRAMES][MFCC_N_COEF];

static void kws_dsp_task(void *pvParameters) {
    (void)pvParameters;

    micro_vad_reset();
    s_mfcc_fill_count = 0;

    for (;;) {
        /* 1. Acquire one 20ms VAD frame. This single capture also feeds
         *    the MFCC accumulation buffer below, so audio is captured
         *    exactly once per frame — no duplicate reads. */
        audio_i2s_read_blocking(s_vad_frame, VAD_FRAME_SAMPLES);

        /* 2. VAD: run every 20ms, log the decision to UART. */
        bool is_speech = micro_vad_update(s_vad_frame);
        printf("[DSP][VAD] speech=%d\n", (int)is_speech);

        /* 3. Feed the same samples into the MFCC accumulation window.
         *    MFCC_MAX_INPUT_SAMPLES (8080) is not an exact multiple of
         *    VAD_FRAME_SAMPLES (320), so the final chunk of a window is
         *    partial — copy only as many samples as still fit. */
        size_t space_remaining = MFCC_MAX_INPUT_SAMPLES - s_mfcc_fill_count;
        size_t copy_count = (VAD_FRAME_SAMPLES < space_remaining) ? VAD_FRAME_SAMPLES : space_remaining;
        memcpy(&s_mfcc_accum[s_mfcc_fill_count], s_vad_frame, copy_count * sizeof(int16_t));
        s_mfcc_fill_count += copy_count;

        /* 4. Once the window is full, run the real MFCC pipeline once and
         *    log a compact summary to UART (not the full 637-value matrix
         *    every cycle — that would flood a UART link). */
        if (s_mfcc_fill_count >= MFCC_MAX_INPUT_SAMPLES) {
            mfcc_status_t status = mfcc_compute_matrix(s_mfcc_accum, (int)MFCC_MAX_INPUT_SAMPLES, s_mfcc_out);

            if (status == MFCC_OK) {
                printf("[DSP][MFCC] status=OK frame0=[");
                for (int c = 0; c < MFCC_N_COEF; ++c) {
                    printf("%d%s", s_mfcc_out[0][c], (c < MFCC_N_COEF - 1) ? "," : "");
                }
                printf("]\n");
            } else {
                printf("[DSP][MFCC] status=ERROR code=%d\n", (int)status);
            }

            /* Non-overlapping windows: start accumulating the next window
             * from scratch. (A production version could instead slide the
             * window by less than a full reset for lower latency — kept
             * simple here since s1.md does not specify overlap behavior.) */
            s_mfcc_fill_count = 0;
        }

        /* No vTaskDelay here: audio_i2s_read_blocking() above is the
         * pacing source (see its doc comment). */
    }
}

/* Public entry point. s1.md's deliverable table lists only dsp_task.c (no
 * paired dsp_task.h) for this task, so this prototype is declared here
 * rather than in a new header — a real ESP-IDF component would normally
 * expose this via firmware/micro_vad/dsp_task.h alongside the component's
 * CMakeLists.txt, once that project structure exists. */
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
