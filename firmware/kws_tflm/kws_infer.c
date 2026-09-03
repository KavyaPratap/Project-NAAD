/* Owner: S2
 * Module: KWS Inference Implementation (TFLM — fully wired)
 * Interface INPUT:  int8_t mfcc[49][13]   — from S1's C MFCC pipeline
 * Interface OUTPUT: float keyword_score [0.0, 1.0]
 * Model:  DS-CNN INT8 (naad_kws_int8.tflite, 43.7 KB)
 * Last validated: 2026-09-03
 *
 * Ops registered (MUST match the model — do NOT use AllOpsResolver):
 *   Conv2D, DepthwiseConv2D, Mean (GAP), FullyConnected, Softmax, Reshape
 *
 * Build requires: idf.py add-component tensorflow-lite-micro
 */

#include "kws_infer.h"
#include "kws_model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "kws_infer";

/* ── Static TFLM objects (no heap after boot) ────────────────────────────── */
static uint8_t s_arena[KWS_ARENA_BYTES] __attribute__((aligned(16)));

static tflite::MicroMutableOpResolver<6> s_resolver;
static tflite::MicroInterpreter         *s_interpreter = nullptr;

/* ── Score history for smoothing ─────────────────────────────────────────── */
static float s_score_history[KWS_SMOOTH_FRAMES] = {0};
static int   s_history_idx = 0;


/* ── kws_init ────────────────────────────────────────────────────────────── */
void kws_init(void)
{
    ESP_LOGI(TAG, "KWS init: model_len=%d, arena=%dKB",
             kws_model_data_len, KWS_ARENA_BYTES / 1024);

    /* 1. Validate flatbuffer */
    const tflite::Model *model = tflite::GetModel(kws_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch: got %lu, expected %d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    /* 2. Register ONLY the ops this DS-CNN uses.
     *    Do NOT use AllOpsResolver — adds ~200 KB to binary. */
    s_resolver.AddConv2D();
    s_resolver.AddDepthwiseConv2D();
    s_resolver.AddMean();           /* GlobalAveragePooling2D lowers to Mean */
    s_resolver.AddFullyConnected(); /* Dense classifier head */
    s_resolver.AddSoftmax();
    s_resolver.AddReshape();

    /* 3. Create interpreter (static storage, no heap) */
    static tflite::MicroInterpreter interpreter(
        model, s_resolver, s_arena, KWS_ARENA_BYTES);
    s_interpreter = &interpreter;

    /* 4. Allocate tensors */
    TfLiteStatus status = s_interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed — arena too small? "
                      "Try increasing KWS_ARENA_BYTES by 16 KB");
        s_interpreter = nullptr;
        return;
    }

    /* 5. Log arena usage so S3 can tune KWS_ARENA_BYTES */
    ESP_LOGI(TAG, "Arena used: %zuB / %dB",
             s_interpreter->arena_used_bytes(), KWS_ARENA_BYTES);

    /* 6. Sanity-check tensor shapes */
    TfLiteTensor *in  = s_interpreter->input(0);
    TfLiteTensor *out = s_interpreter->output(0);
    ESP_LOGI(TAG, "Input  tensor: type=%d, dims=[%d,%d,%d,%d]",
             in->type,
             in->dims->data[0], in->dims->data[1],
             in->dims->data[2], in->dims->data[3]);
    ESP_LOGI(TAG, "Output tensor: type=%d, dims=[%d,%d]",
             out->type, out->dims->data[0], out->dims->data[1]);

    memset(s_score_history, 0, sizeof(s_score_history));
    ESP_LOGI(TAG, "KWS init complete");
}


/* ── kws_infer ───────────────────────────────────────────────────────────── */
float kws_infer(const int8_t mfcc[49][13])
{
    if (s_interpreter == nullptr) {
        ESP_LOGE(TAG, "kws_infer() called before successful kws_init()");
        return 0.0f;
    }

    /* 1. Copy MFCC int8 features into model input tensor
     *    Input tensor shape: (1, 49, 13, 1) — flat copy matches row-major layout */
    TfLiteTensor *input = s_interpreter->input(0);
    memcpy(input->data.int8, mfcc, 49 * 13 * sizeof(int8_t));

    /* 2. Run inference and measure latency */
    int64_t t0 = esp_timer_get_time();
    TfLiteStatus status = s_interpreter->Invoke();
    int64_t t1 = esp_timer_get_time();

    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed");
        return 0.0f;
    }

    ESP_LOGD(TAG, "Inference time: %lldus", (long long)(t1 - t0));

    /* 3. Read output tensor — index 1 = keyword class probability
     *    Output tensor type: int8 with quantization params (scale, zero_point) */
    TfLiteTensor *output = s_interpreter->output(0);
    int8_t  raw_score = output->data.int8[1];          /* class 1 = keyword */
    float   scale     = output->params.scale;
    int32_t zero_pt   = output->params.zero_point;

    /* 4. Dequantize: float_val = (int8_val - zero_point) * scale */
    float score = (float)(raw_score - zero_pt) * scale;

    /* Clamp to [0, 1] for safety */
    if (score < 0.0f) score = 0.0f;
    if (score > 1.0f) score = 1.0f;

    ESP_LOGD(TAG, "raw=%d scale=%.6f zp=%ld -> score=%.4f",
             raw_score, scale, (long)zero_pt, score);

    return score;
}


/* ── kws_is_triggered ────────────────────────────────────────────────────── */
bool kws_is_triggered(float score)
{
    /* Sliding window average — smooths out single-frame glitches */
    s_score_history[s_history_idx % KWS_SMOOTH_FRAMES] = score;
    s_history_idx++;

    float avg = 0.0f;
    for (int i = 0; i < KWS_SMOOTH_FRAMES; i++) {
        avg += s_score_history[i];
    }
    avg /= KWS_SMOOTH_FRAMES;

    if (avg >= KWS_THRESHOLD) {
        ESP_LOGI(TAG, "KWS TRIGGERED: avg_score=%.3f (threshold=%.2f)",
                 avg, KWS_THRESHOLD);
        return true;
    }
    return false;
}


/* ── kws_reset_trigger ───────────────────────────────────────────────────── */
void kws_reset_trigger(void)
{
    memset(s_score_history, 0, sizeof(s_score_history));
    s_history_idx = 0;
    ESP_LOGD(TAG, "KWS trigger state reset");
}
