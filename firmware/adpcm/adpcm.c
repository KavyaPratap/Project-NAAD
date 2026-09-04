/* Owner: S3
 * Module: IMA-ADPCM Encoder/Decoder (C)
 * Last tested: [date, firmware commit]
 *
 * IMA-ADPCM standard:
 *   - 89 step sizes (7 to 32767)
 *   - 4-bit output codes (sign bit + 3 magnitude bits)
 *   - 2 samples packed per byte: low nibble first, high nibble second
 *   - Compression ratio: 4:1 (16-bit PCM -> 4-bit ADPCM)
 *   - Bitrate: 16kHz * 4 bits = 64 kbps (vs 256 kbps raw PCM)
 */
#include "adpcm.h"
#include <stdint.h>

/* IMA-ADPCM standard step table (89 entries) */
static const int step_table[89] = {
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34,
    37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143,
    157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494,
    544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552,
    1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428,
    4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635,
    13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
};

/* Index adjustment table for step_index update */
static const int index_table[8] = { -1, -1, -1, -1, 2, 4, 6, 8 };

static inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

void adpcm_state_reset(adpcm_state_t *s) {
    s->predictor  = 0;
    s->step_index = 0;
}

uint8_t adpcm_encode_sample(adpcm_state_t *s, int16_t sample) {
    int step = step_table[s->step_index];
    int diff = (int)sample - (int)s->predictor;

    /* Build 4-bit code: bit3=sign, bits2-0=magnitude */
    uint8_t code = 0;
    if (diff < 0) { code = 8; diff = -diff; }
    if (diff >= step)        { code |= 4; diff -= step; }
    if (diff >= (step >> 1)) { code |= 2; diff -= (step >> 1); }
    if (diff >= (step >> 2)) { code |= 1; }

    /* Reconstruct — encoder mirrors decoder exactly so state stays in sync */
    int recon_diff = (step >> 3);
    if (code & 4) recon_diff += step;
    if (code & 2) recon_diff += (step >> 1);
    if (code & 1) recon_diff += (step >> 2);
    if (code & 8) recon_diff = -recon_diff;

    s->predictor  = (int16_t)clamp_int((int)s->predictor + recon_diff, -32768, 32767);
    s->step_index = clamp_int(s->step_index + index_table[code & 7], 0, 88);

    return code & 0x0F;  // 4-bit code
}

int16_t adpcm_decode_sample(adpcm_state_t *s, uint8_t code) {
    int step = step_table[s->step_index];
    int diff = (step >> 3);
    if (code & 4) diff += step;
    if (code & 2) diff += (step >> 1);
    if (code & 1) diff += (step >> 2);
    if (code & 8) diff = -diff;

    s->predictor  = (int16_t)clamp_int((int)s->predictor + diff, -32768, 32767);
    s->step_index = clamp_int(s->step_index + index_table[code & 7], 0, 88);
    return s->predictor;
}

void adpcm_encode_block(adpcm_state_t *s, const int16_t *pcm, int n, uint8_t *out) {
    /* Pack two 4-bit codes per byte: low nibble first, high nibble second */
    for (int i = 0; i < n; i += 2) {
        uint8_t lo = adpcm_encode_sample(s, pcm[i]);
        uint8_t hi = adpcm_encode_sample(s, pcm[i + 1]);
        out[i / 2] = (hi << 4) | lo;
    }
}
