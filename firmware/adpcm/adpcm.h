/* Owner: S3
 * Module: IMA-ADPCM Encoder/Decoder
 * Interface INPUT:  int16_t PCM samples
 * Interface OUTPUT: uint8_t ADPCM bytes (2 samples per byte = 4-bit codes)
 * State: predictor + step_index — reset at stream start
 * Last tested: [date, firmware commit]
 *
 * IMPORTANT: adpcm_decode_sample() here MUST match server/adpcm_decoder.py exactly.
 *            Any mismatch = corrupted audio on server.
 */
#pragma once
#include <stdint.h>

typedef struct {
    int16_t predictor;
    int     step_index;
} adpcm_state_t;

/** Reset state — call before streaming starts */
void adpcm_state_reset(adpcm_state_t *state);

/** Encode a single int16 PCM sample -> 4-bit ADPCM code */
uint8_t adpcm_encode_sample(adpcm_state_t *state, int16_t sample);

/** Decode a 4-bit ADPCM code -> int16 PCM sample */
int16_t adpcm_decode_sample(adpcm_state_t *state, uint8_t code);

/**
 * Encode a block of PCM samples.
 * @param state     ADPCM encoder state (persistent across calls)
 * @param pcm_in    Input PCM samples (int16, n_samples count)
 * @param n_samples Number of input samples — MUST be even
 * @param adpcm_out Output buffer (must be n_samples/2 bytes)
 *                  Packing: low nibble = first sample, high nibble = second
 */
void adpcm_encode_block(
    adpcm_state_t *state,
    const int16_t *pcm_in,
    int            n_samples,
    uint8_t       *adpcm_out
);
