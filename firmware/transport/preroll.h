/* Owner: S3
 * Module: Pre-roll Circular Buffer
 * Purpose: Store 500ms of PCM BEFORE keyword trigger,
 *          so the server gets the full keyword utterance
 * Last tested: [date, firmware commit]
 *
 * HOW IT WORKS:
 *   - Always running: every 20ms audio frame is pushed in
 *   - Buffer holds 500ms = 8000 samples in a ring
 *   - On KWS trigger: preroll_flush() dumps all buffered audio first,
 *     then live audio follows → server gets complete keyword from start
 */
#pragma once
#include <stdint.h>
#include <stdbool.h>

/* 500ms pre-roll @ 16kHz = 8000 samples */
#define PREROLL_MS        500
#define PREROLL_SAMPLES   8000
#define PREROLL_BYTES     (PREROLL_SAMPLES * 2)

/** Initialize the pre-roll buffer. Call once at boot. */
void preroll_init(void);

/**
 * Push a frame of PCM into the circular buffer.
 * Call every 20ms with each DMA audio frame (320 samples).
 * Automatically overwrites oldest data when full.
 * @param frame      Pointer to int16 PCM samples
 * @param n_samples  Number of samples (typically 320)
 */
void preroll_push(const int16_t *frame, int n_samples);

/**
 * Flush all buffered pre-roll audio into output buffer.
 * Call ONCE on KWS trigger, before starting live stream.
 * @param out          Output buffer (must be >= PREROLL_BYTES bytes)
 * @param out_samples  Filled with number of valid samples in out
 */
void preroll_flush(int16_t *out, int *out_samples);

/** Reset buffer (call after trigger handling completes) */
void preroll_reset(void);

/** Returns how many samples are currently buffered */
int preroll_samples_buffered(void);
