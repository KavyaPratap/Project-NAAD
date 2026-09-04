/* Owner: S3
 * Module: Pre-roll Circular Buffer
 * Last tested: [date, firmware commit]
 */
#include "preroll.h"
#include <string.h>

/* ── Circular buffer storage ─────────────────────────────────── */
static int16_t s_buf[PREROLL_SAMPLES];
static int     s_write_pos = 0;   // next write index
static int     s_count     = 0;   // total samples buffered (max PREROLL_SAMPLES)

void preroll_init(void) {
    memset(s_buf, 0, sizeof(s_buf));
    s_write_pos = 0;
    s_count     = 0;
}

void preroll_push(const int16_t *frame, int n_samples) {
    for (int i = 0; i < n_samples; i++) {
        s_buf[s_write_pos] = frame[i];
        s_write_pos = (s_write_pos + 1) % PREROLL_SAMPLES;
        if (s_count < PREROLL_SAMPLES) s_count++;
        // When s_count == PREROLL_SAMPLES, we overwrite oldest — that's the ring behavior
    }
}

void preroll_flush(int16_t *out, int *out_samples) {
    /* Read out s_count samples starting from oldest entry.
     * Oldest entry is at: (s_write_pos - s_count + PREROLL_SAMPLES) % PREROLL_SAMPLES */
    int read_pos = (s_write_pos - s_count + PREROLL_SAMPLES) % PREROLL_SAMPLES;
    for (int i = 0; i < s_count; i++) {
        out[i] = s_buf[read_pos];
        read_pos = (read_pos + 1) % PREROLL_SAMPLES;
    }
    *out_samples = s_count;
}

void preroll_reset(void) {
    s_write_pos = 0;
    s_count     = 0;
    memset(s_buf, 0, sizeof(s_buf));
}

int preroll_samples_buffered(void) {
    return s_count;
}
