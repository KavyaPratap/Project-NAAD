/* Owner: S3
 * Module: I2S DMA Audio Capture
 * Hardware: ESP32-S3 + INMP441 MEMS mic
 * Interface OUTPUT: int16_t audio frames via FreeRTOS queue
 *                   AND blocking read for S1's dsp_task.c
 * Last tested: [date, firmware commit]
 *
 * PIN WIRING (INMP441 -> ESP32-S3):
 *   VDD  -> 3.3V
 *   GND  -> GND
 *   SD   -> GPIO5   (data out from mic = DIN on ESP32)
 *   WS   -> GPIO6   (LRCLK)
 *   SCK  -> GPIO7   (BCLK)
 *   L/R  -> GND     (selects left channel = mono)
 */
#pragma once
#include <stdint.h>
#include <stddef.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

/* ── Audio format constants ─────────────────────────────────── */
#define AUDIO_SAMPLE_RATE    16000
#define AUDIO_BIT_DEPTH      16
#define AUDIO_CHANNELS       1
#define AUDIO_DMA_FRAME_MS   20               // 20ms per DMA buffer
#define AUDIO_DMA_SAMPLES    320              // 20ms @ 16kHz
#define AUDIO_DMA_BYTES      (AUDIO_DMA_SAMPLES * 2)  // int16 output = 2 bytes
#define AUDIO_DMA_BYTES_32   (AUDIO_DMA_SAMPLES * 4)  // 32-bit I2S slot = 4 bytes (INMP441)

/* ── Pin config (change to match YOUR board wiring) ─────────── */
#define I2S_PORT        I2S_NUM_0
#define PIN_BCLK        GPIO_NUM_7
#define PIN_LRCLK       GPIO_NUM_6
#define PIN_DIN         GPIO_NUM_5   // DOUT from INMP441 -> DIN on ESP32

/* ── Public API ─────────────────────────────────────────────── */

/**
 * Start I2S DMA capture. Spawns AudioCapture task on Core 0.
 * @param pcm_queue  FreeRTOS queue that will receive 320-sample int16 blocks.
 *                   Pass NULL if only using audio_i2s_read_blocking().
 */
void audio_i2s_start(QueueHandle_t pcm_queue);

/**
 * Stop I2S capture and release hardware.
 */
void audio_i2s_stop(void);

/**
 * BLOCKING read — S1's dsp_task.c calls this.
 * Blocks until exactly num_samples int16 PCM samples are available.
 * Internally reads from the internal DMA queue (not pcm_queue above).
 * @param out          Buffer to write samples into (must be >= num_samples * 2 bytes)
 * @param num_samples  Number of int16 samples to read (typically 320 = 20ms)
 */
void audio_i2s_read_blocking(int16_t *out, size_t num_samples);

/**
 * Returns total number of DMA frames dropped due to full queue.
 * Should be 0 in normal operation.
 */
uint32_t audio_i2s_get_drop_count(void);
