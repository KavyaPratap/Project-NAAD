/* Owner: S1
 * Module: Micro-VAD (Voice Activity Detection) — C port
 * Mirrors ml/scripts/vad_reference.py (MicroVAD class) exactly:
 * STE (int64 sum of squares) + ZCR (sign-change count, zero=positive)
 * + hangover state machine.
 */
#include "micro_vad.h"
#include <stddef.h>

static int s_hangover = 0;

int64_t compute_ste(const int16_t *frame, int len) {
    int64_t energy = 0;
    for (int i = 0; i < len; ++i) {
        int32_t s = frame[i];
        energy += (int64_t)s * s;
    }
    return energy;
}

int compute_zcr(const int16_t *frame, int len) {
    int zcr = 0;
    for (int i = 1; i < len; ++i) {
        /* Zero treated as positive, matching the Python reference's
         * np.sign(...) with signs[signs == 0] = 1. */
        bool prev_nonneg = frame[i - 1] >= 0;
        bool curr_nonneg = frame[i] >= 0;
        zcr += (curr_nonneg != prev_nonneg);
    }
    return zcr;
}

void micro_vad_reset(void) {
    s_hangover = 0;
}

bool micro_vad_update(const int16_t *frame) {
    if (frame == NULL) {
        return false;
    }

    bool energy_ok = compute_ste(frame, VAD_FRAME_SAMPLES) > STE_THRESHOLD;
    int  zcr        = compute_zcr(frame, VAD_FRAME_SAMPLES);
    bool zcr_ok      = (zcr > ZCR_LOW && zcr < ZCR_HIGH);

    if (energy_ok && zcr_ok) {
        s_hangover = HANGOVER_FRAMES;
        return true;
    }
    if (s_hangover > 0) {
        --s_hangover;
        return true;
    }
    return false;
}
