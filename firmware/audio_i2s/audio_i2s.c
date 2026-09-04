/* Owner: S3
 * Module: I2S DMA Audio Capture
 * Hardware: INMP441 MEMS mic on ESP32-S3
 * Last tested: [date, firmware commit]
 *
 * HOW IT WORKS:
 *   - I2S DMA runs on Core 0 task "AudioCapture" (priority 5)
 *   - Every 20ms: 320 int16 samples arrive from INMP441 via DMA
 *   - Samples go into s_internal_queue (used by audio_i2s_read_blocking)
 *   - Optionally ALSO go into s_pcm_queue (user-supplied, for other consumers)
 *   - S1's dsp_task.c calls audio_i2s_read_blocking() — it blocks here
 *     until exactly num_samples are available from s_internal_queue
 */
#include "audio_i2s.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include <string.h>
#include "transport/preroll.h"   /* Feed preroll ring on every captured frame */

static const char *TAG = "AUDIO_I2S";

/* ── Internal state ─────────────────────────────────────────── */
static i2s_chan_handle_t  s_rx_handle      = NULL;
static QueueHandle_t      s_pcm_queue      = NULL;   // optional external queue
static QueueHandle_t      s_internal_queue = NULL;   // for audio_i2s_read_blocking
static uint32_t           s_drop_count     = 0;

/* Internal queue depth: 40 frames = 800ms headroom.
 * DSP accumulates 25 frames before MFCC+KWS inference.
 * During inference the capture task keeps producing — needs >= 25 spare slots. */
#define INTERNAL_QUEUE_DEPTH  25

/* ── Audio capture task (Core 0) ────────────────────────────── */
static void audio_capture_task(void *arg) {
    /* INMP441 outputs 32-bit I2S slots (18-bit data, left-justified).
     * We read int32 and shift right by 14 to get signed int16. */
    static int32_t dma_buf32[AUDIO_DMA_SAMPLES];   // 32-bit raw from I2S
    static int16_t dma_buf16[AUDIO_DMA_SAMPLES];   // 16-bit converted

    while (1) {
        size_t bytes_read = 0;
        esp_err_t ret = i2s_channel_read(
            s_rx_handle, dma_buf32, AUDIO_DMA_BYTES_32, &bytes_read, pdMS_TO_TICKS(200)
        );

        if (ret != ESP_OK || bytes_read != AUDIO_DMA_BYTES_32) {
            /* Timeout or partial read — not an error for INMP441, just retry */
            if (ret != ESP_ERR_TIMEOUT) {
                ESP_LOGW(TAG, "I2S read issue: ret=%d bytes=%u/%u",
                         ret, (unsigned)bytes_read, (unsigned)AUDIO_DMA_BYTES_32);
            }
            continue;
        }

        /* Convert 32-bit INMP441 samples to int16 */
        for (int i = 0; i < AUDIO_DMA_SAMPLES; i++) {
            /* INMP441: 18-bit data in MSB of 32-bit word, right-shift to int16 */
            dma_buf16[i] = (int16_t)(dma_buf32[i] >> 14);
        }

        /* Push int16 frames to internal queue (for audio_i2s_read_blocking) */
        if (xQueueSend(s_internal_queue, dma_buf16, 0) != pdTRUE) {
            s_drop_count++;
            if (s_drop_count % 50 == 1) {
                ESP_LOGW(TAG, "Internal queue full! Drop count: %lu", s_drop_count);
            }
        }

        /* Feed pre-roll circular buffer on EVERY frame — always running.
         * This ensures preroll_flush() has real audio when KWS triggers.
         * (Previously the preroll was never fed, causing empty transcripts.) */
        preroll_push(dma_buf16, AUDIO_DMA_SAMPLES);

        /* Also push to optional external queue (for transport task consumer) */
        if (s_pcm_queue != NULL) {
            xQueueSend(s_pcm_queue, dma_buf16, 0);  // non-blocking, drop on full
        }
    }
}

/* ── Public API ─────────────────────────────────────────────── */

void audio_i2s_start(QueueHandle_t pcm_queue) {
    s_pcm_queue = pcm_queue;

    /* Create internal blocking-read queue */
    s_internal_queue = xQueueCreate(INTERNAL_QUEUE_DEPTH, AUDIO_DMA_BYTES);
    if (s_internal_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create internal queue!");
        return;
    }

    /* Configure I2S channel */
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_PORT, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &s_rx_handle));

    /* Configure I2S standard mode (INMP441 = Philips/I2S justified, 32-bit slots) */
    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(AUDIO_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO
        ),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = PIN_BCLK,
            .ws   = PIN_LRCLK,
            .dout = I2S_GPIO_UNUSED,
            .din  = PIN_DIN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false
            }
        }
    };

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx_handle));

    ESP_LOGI(TAG, "I2S started: %d Hz, 16-bit mono, DMA frame=20ms",
             AUDIO_SAMPLE_RATE);
    ESP_LOGI(TAG, "Pins: BCLK=GPIO%d  LRCLK=GPIO%d  DIN=GPIO%d",
             PIN_BCLK, PIN_LRCLK, PIN_DIN);

    /* Start capture task on Core 0 (DSP runs on Core 1) */
    xTaskCreatePinnedToCore(
        audio_capture_task,
        "AudioCapture",
        4096,
        NULL,
        5,    // priority
        NULL,
        0     // Core 0
    );
}

void audio_i2s_stop(void) {
    if (s_rx_handle) {
        i2s_channel_disable(s_rx_handle);
        i2s_del_channel(s_rx_handle);
        s_rx_handle = NULL;
    }
    ESP_LOGI(TAG, "I2S stopped. Total drops: %lu", s_drop_count);
}

void audio_i2s_read_blocking(int16_t *out, size_t num_samples) {
    /* Called by S1's dsp_task.c — blocks until num_samples are ready.
     * We read one 320-sample frame from the internal queue at a time.
     * If num_samples > 320, loop until all samples are filled. */
    size_t filled = 0;
    static int16_t frame_buf[AUDIO_DMA_SAMPLES];

    while (filled < num_samples) {
        /* Block indefinitely until a frame arrives */
        if (xQueueReceive(s_internal_queue, frame_buf, portMAX_DELAY) == pdTRUE) {
            size_t to_copy = num_samples - filled;
            if (to_copy > AUDIO_DMA_SAMPLES) to_copy = AUDIO_DMA_SAMPLES;
            memcpy(out + filled, frame_buf, to_copy * sizeof(int16_t));
            filled += to_copy;
        }
    }
}

uint32_t audio_i2s_get_drop_count(void) {
    return s_drop_count;
}
