/* Owner: S1
 * Module: MFCC Feature Extraction (Float Reference Port) — C implementation
 * Mirrors ml/scripts/mfcc_reference.py stage-for-stage. See mfcc.h for the
 * interface contract and the documented deviation (status return).
 *
 * No ESP-IDF/esp-dsp project exists in this repository yet, so the FFT here
 * is a portable radix-2 Cooley-Tukey implementation (no external deps).
 * When an ESP-IDF build is set up, this should be swapped for esp-dsp's
 * dsps_fft2r_fc32() as noted in s1.md — the surrounding pipeline (framing,
 * windowing, Mel filterbank, DCT, quantization) would not need to change.
 */
#include "mfcc.h"
#include <math.h>
#include <string.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================
 * Static working buffers (no dynamic allocation; not reentrant —
 * see the "NOTE" in mfcc.h).
 * ============================================================ */
static float s_preemph[MFCC_MAX_INPUT_SAMPLES];
static float s_mfcc_float[MFCC_N_FRAMES][MFCC_N_COEF];
static float s_fft_re[MFCC_FFT_SIZE];
static float s_fft_im[MFCC_FFT_SIZE];

/* Lazily-initialized deterministic tables (computed once, mirrors the
 * Python module precomputing np.hamming(400) and _MEL_FILTERBANK once
 * at import time). Not random / not runtime-input-dependent. */
static float s_hamming_window[MFCC_FRAME_SAMPLES];
static float s_mel_filterbank[MFCC_N_MEL][MFCC_FFT_SIZE / 2 + 1];
static int   s_tables_ready = 0;

static double hz_to_mel(double f) {
    return 2595.0 * log10(1.0 + f / 700.0);
}

static double mel_to_hz(double m) {
    return 700.0 * (pow(10.0, m / 2595.0) - 1.0);
}

static void build_hamming_window(void) {
    /* Matches np.hamming(N): w[n] = 0.54 - 0.46*cos(2*pi*n/(N-1)) */
    const int N = MFCC_FRAME_SAMPLES;
    for (int n = 0; n < N; ++n) {
        double w = 0.54 - 0.46 * cos(2.0 * M_PI * (double)n / (double)(N - 1));
        s_hamming_window[n] = (float)w;
    }
}

static void build_mel_filterbank(void) {
    /* Mirrors build_mel_filterbank() in mfcc_reference.py exactly:
     * - freq_bins = linspace(0, SR/2, FFT_SIZE/2+1)
     * - mel_points = linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MEL+2)
     * - hz_points = mel_to_hz(mel_points)
     * - triangular filters between hz_points[m], [m+1], [m+2]
     */
    const int n_bins = MFCC_FFT_SIZE / 2 + 1; /* 257 */
    double freq_bins[MFCC_FFT_SIZE / 2 + 1];
    for (int k = 0; k < n_bins; ++k) {
        /* linspace(0, SR/2, n_bins): step = (SR/2)/(n_bins-1) */
        freq_bins[k] = (double)k * (MFCC_SAMPLE_RATE / 2.0) / (double)(n_bins - 1);
    }

    double mel_min = hz_to_mel(MFCC_MEL_FMIN);
    double mel_max = hz_to_mel(MFCC_MEL_FMAX);

    double hz_points[MFCC_N_MEL + 2];
    for (int i = 0; i < MFCC_N_MEL + 2; ++i) {
        double mel = mel_min + (double)i * (mel_max - mel_min) / (double)(MFCC_N_MEL + 2 - 1);
        hz_points[i] = mel_to_hz(mel);
    }

    memset(s_mel_filterbank, 0, sizeof(s_mel_filterbank));
    for (int m = 0; m < MFCC_N_MEL; ++m) {
        double left   = hz_points[m];
        double center = hz_points[m + 1];
        double right  = hz_points[m + 2];
        for (int k = 0; k < n_bins; ++k) {
            double f = freq_bins[k];
            double w = 0.0;
            if (f >= left && f <= center) {
                w = (f - left) / (center - left);
            } else if (f > center && f <= right) {
                w = (right - f) / (right - center);
            }
            s_mel_filterbank[m][k] = (float)w;
        }
    }
}

static void ensure_tables_ready(void) {
    if (!s_tables_ready) {
        build_hamming_window();
        build_mel_filterbank();
        s_tables_ready = 1;
    }
}

/* In-place iterative radix-2 Cooley-Tukey FFT, size MFCC_FFT_SIZE (512).
 * Unscaled forward transform (no 1/N factor) — matches numpy.fft.rfft's
 * convention exactly. re[]/im[] must each have MFCC_FFT_SIZE elements. */
static void fft_radix2(float *re, float *im) {
    const int n = MFCC_FFT_SIZE;

    /* Bit-reversal permutation */
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }

    /* Butterfly stages. Twiddles computed directly per-index (not
     * accumulated recursively) to avoid drift over the 9 stages. */
    for (int len = 2; len <= n; len <<= 1) {
        int half = len / 2;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < half; ++k) {
                double angle = -2.0 * M_PI * (double)k / (double)len;
                float wr = (float)cos(angle);
                float wi = (float)sin(angle);

                float ur = re[i + k];
                float ui = im[i + k];
                float vr = re[i + k + half] * wr - im[i + k + half] * wi;
                float vi = re[i + k + half] * wi + im[i + k + half] * wr;

                re[i + k]        = ur + vr;
                im[i + k]        = ui + vi;
                re[i + k + half] = ur - vr;
                im[i + k + half] = ui - vi;
            }
        }
    }
}

static void zero_fill_output(int8_t out_features[MFCC_N_FRAMES][MFCC_N_COEF]) {
    memset(out_features, 0, sizeof(int8_t) * MFCC_N_FRAMES * MFCC_N_COEF);
}

mfcc_status_t mfcc_compute_matrix(
    const int16_t *audio_in,
    int            audio_len,
    int8_t         out_features[MFCC_N_FRAMES][MFCC_N_COEF]
) {
    if (out_features == NULL) {
        return MFCC_ERR_NULL_OUTPUT;
    }
    if (audio_in == NULL) {
        zero_fill_output(out_features);
        return MFCC_ERR_NULL_INPUT;
    }
    if (audio_len < MFCC_FRAME_SAMPLES) {
        zero_fill_output(out_features);
        return MFCC_ERR_INSUFFICIENT_LENGTH;
    }

    ensure_tables_ready();

    /* Only the first MFCC_MAX_INPUT_SAMPLES samples can ever affect the
     * output (see mfcc.h) — everything beyond that is provably unused. */
    int proc_len = (audio_len < MFCC_MAX_INPUT_SAMPLES) ? audio_len : MFCC_MAX_INPUT_SAMPLES;

    /* Stage 1: int16 -> float32 in [-1,1] (divide by 32768.0, matching
     * librosa/soundfile's PCM16 convention exactly) + pre-emphasis, applied
     * over the CONTINUOUS signal (not per-frame) so overlapping frames see
     * the correct cross-frame-boundary sample, exactly like
     * apply_preemphasis(load_audio(path)) in the Python reference. */
    float prev_raw = 0.0f;
    for (int i = 0; i < proc_len; ++i) {
        float raw = (float)audio_in[i] / 32768.0f;
        if (i == 0) {
            s_preemph[i] = raw; /* y[0] = x[0], no subtraction (matches np.append(audio[0], ...)) */
        } else {
            s_preemph[i] = raw - MFCC_PREEMPH_ALPHA * prev_raw;
        }
        prev_raw = raw;
    }

    int num_frames_available = 1 + (proc_len - MFCC_FRAME_SAMPLES) / MFCC_HOP_SAMPLES;
    int num_frames = (num_frames_available < MFCC_N_FRAMES) ? num_frames_available : MFCC_N_FRAMES;

    /* Zero the float MFCC matrix first — frames beyond num_frames stay
     * zero, exactly matching the Python reference's np.zeros() pad. */
    memset(s_mfcc_float, 0, sizeof(s_mfcc_float));

    for (int f = 0; f < num_frames; ++f) {
        int start = f * MFCC_HOP_SAMPLES;

        /* Stage 2+3: framing + Hamming window */
        for (int i = 0; i < MFCC_FRAME_SAMPLES; ++i) {
            s_fft_re[i] = s_preemph[start + i] * s_hamming_window[i];
        }
        /* Stage 4a: zero-pad 400 -> 512 */
        for (int i = MFCC_FRAME_SAMPLES; i < MFCC_FFT_SIZE; ++i) {
            s_fft_re[i] = 0.0f;
        }
        memset(s_fft_im, 0, sizeof(s_fft_im));

        /* Stage 4b: 512-point FFT + power spectrum (real^2 + imag^2) */
        fft_radix2(s_fft_re, s_fft_im);

        float power[MFCC_FFT_SIZE / 2 + 1];
        for (int k = 0; k < MFCC_FFT_SIZE / 2 + 1; ++k) {
            power[k] = s_fft_re[k] * s_fft_re[k] + s_fft_im[k] * s_fft_im[k];
        }

        /* Stage 5: apply 26-band Mel filterbank */
        float mel_energy[MFCC_N_MEL];
        for (int m = 0; m < MFCC_N_MEL; ++m) {
            float sum = 0.0f;
            for (int k = 0; k < MFCC_FFT_SIZE / 2 + 1; ++k) {
                sum += power[k] * s_mel_filterbank[m][k];
            }
            mel_energy[m] = sum;
        }

        /* Stage 6: log compression */
        float log_mel[MFCC_N_MEL];
        for (int m = 0; m < MFCC_N_MEL; ++m) {
            log_mel[m] = logf(mel_energy[m] + MFCC_LOG_EPSILON);
        }

        /* Stage 7: DCT-II, norm='ortho', keep first MFCC_N_COEF coefficients.
         * Y_0 = sum(x)/sqrt(N); Y_k = sqrt(2/N) * sum(x_n * cos(pi*(2n+1)*k/(2N))) */
        const double N = (double)MFCC_N_MEL;
        for (int k = 0; k < MFCC_N_COEF; ++k) {
            double sum = 0.0;
            if (k == 0) {
                for (int n = 0; n < MFCC_N_MEL; ++n) {
                    sum += (double)log_mel[n];
                }
                s_mfcc_float[f][k] = (float)(sum / sqrt(N));
            } else {
                for (int n = 0; n < MFCC_N_MEL; ++n) {
                    double angle = M_PI * (double)(2 * n + 1) * (double)k / (2.0 * N);
                    sum += (double)log_mel[n] * cos(angle);
                }
                s_mfcc_float[f][k] = (float)(sqrt(2.0 / N) * sum);
            }
        }
    }

    /* Stage 8: per-feature (per-coefficient, across all 49 frames including
     * zero-padded ones) mean/std normalization, scale x32.0, clamp, int8.
     * Matches normalize_and_quantize() exactly, including numpy's default
     * population std (ddof=0) and truncating (not rounding) float->int8 cast. */
    for (int c = 0; c < MFCC_N_COEF; ++c) {
        double mean = 0.0;
        for (int f = 0; f < MFCC_N_FRAMES; ++f) {
            mean += (double)s_mfcc_float[f][c];
        }
        mean /= (double)MFCC_N_FRAMES;

        double var = 0.0;
        for (int f = 0; f < MFCC_N_FRAMES; ++f) {
            double d = (double)s_mfcc_float[f][c] - mean;
            var += d * d;
        }
        var /= (double)MFCC_N_FRAMES;
        double std = sqrt(var) + 1e-8;

        for (int f = 0; f < MFCC_N_FRAMES; ++f) {
            double normalized = ((double)s_mfcc_float[f][c] - mean) / std;
            double scaled = normalized * (double)MFCC_INT8_SCALE;
            if (scaled < -128.0) scaled = -128.0;
            if (scaled > 127.0)  scaled = 127.0;
            out_features[f][c] = (int8_t)scaled; /* truncates toward zero, matching .astype(np.int8) */
        }
    }

    return MFCC_OK;
}
