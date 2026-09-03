/* Owner: S2
 * Module: KWS TFLM Deployment — Model Data Header
 * How to generate kws_model_data.cc:
 *   xxd -i ml/models/naad_kws_int8.tflite > firmware/kws_tflm/kws_model_data.cc
 * Then edit kws_model_data.cc:
 *   Rename: ml_models_naad_kws_int8_tflite[]  → kws_model_data[]
 *   Rename: ml_models_naad_kws_int8_tflite_len → kws_model_data_len
 *   Add:    const uint8_t  and  const int  qualifiers
 * Last tested: [date, firmware commit]
 */
#pragma once
#include <stdint.h>

extern const uint8_t kws_model_data[];
extern const int     kws_model_data_len;
