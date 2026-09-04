/* Owner: S2 + Integration
 * Module: KWS Inference Implementation (TFLM — fully wired)
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

static uint8_t s_arena[KWS_ARENA_BYTES] __attribute__((aligned(16)));
static tflite::MicroMutableOpResolver<6> s_resolver;
static tflite::MicroInterpreter         *s_interpreter = nullptr;

void kws_init(void)
{
    ESP_LOGI(TAG, "KWS init: model_len=%d, arena=%dKB",
             kws_model_data_len, KWS_ARENA_BYTES / 1024);

    const tflite::Model *model = tflite::GetModel(kws_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch: got %lu, expected %d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    s_resolver.AddConv2D();
    s_resolver.AddDepthwiseConv2D();
    s_resolver.AddMean();
    s_resolver.AddFullyConnected();
    s_resolver.AddSoftmax();
    s_resolver.AddReshape();

    static tflite::MicroInterpreter interpreter(
        model, s_resolver, s_arena, KWS_ARENA_BYTES);
    s_interpreter = &interpreter;

    TfLiteStatus status = s_interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed — arena too small?");
        s_interpreter = nullptr;
        return;
    }

    ESP_LOGI(TAG, "Arena used: %zuB / %dB",
             s_interpreter->arena_used_bytes(), KWS_ARENA_BYTES);

    TfLiteTensor *in  = s_interpreter->input(0);
    TfLiteTensor *out = s_interpreter->output(0);
    ESP_LOGI(TAG, "Input tensor dims: [%d,%d,%d,%d]",
             in->dims->data[0], in->dims->data[1], in->dims->data[2], in->dims->data[3]);
    ESP_LOGI(TAG, "Output tensor dims: [%d,%d]", out->dims->data[0], out->dims->data[1]);

    ESP_LOGI(TAG, "KWS init complete");
}

float kws_infer(const int8_t mfcc[49][13])
{
    if (s_interpreter == nullptr) {
        ESP_LOGE(TAG, "kws_infer() called before kws_init()");
        return 0.0f;
    }

    TfLiteTensor *input = s_interpreter->input(0);
    memcpy(input->data.int8, mfcc, 49 * 13 * sizeof(int8_t));

    int64_t t0 = esp_timer_get_time();
    TfLiteStatus status = s_interpreter->Invoke();
    int64_t t1 = esp_timer_get_time();

    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed");
        return 0.0f;
    }

    ESP_LOGD(TAG, "Inference time: %lldus", (long long)(t1 - t0));

    TfLiteTensor *output = s_interpreter->output(0);
    int8_t  raw_score = output->data.int8[1];          /* class 1 = keyword */
    float   scale     = output->params.scale;
    int32_t zero_pt   = output->params.zero_point;

    float score = (float)(raw_score - zero_pt) * scale;
    if (score < 0.0f) score = 0.0f;
    if (score > 1.0f) score = 1.0f;

    return score;
}

bool kws_is_triggered(float score)
{
    /* Single-window evaluation threshold for 500ms non-overlapping frames */
    if (score >= KWS_THRESHOLD) {
        ESP_LOGI(TAG, "KWS TRIGGERED! score=%.3f >= threshold=%.2f",
                 score, (double)KWS_THRESHOLD);
        return true;
    }
    return false;
}

void kws_reset_trigger(void)
{
    ESP_LOGD(TAG, "KWS trigger state reset");
}
