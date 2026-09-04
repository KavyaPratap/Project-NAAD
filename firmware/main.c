/* Owner: S3
 * Module: Main Application Entry Point
 * Purpose: Wires S1 (DSP) + S2 (KWS) + S3 (transport) together
 * Last tested: [date, firmware commit]
 */
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/event_groups.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "esp_timer.h"
#include <math.h>

#include "audio_i2s/audio_i2s.h"
#include "adpcm/adpcm.h"
#include "transport/preroll.h"
#include "transport/tcp_client.h"
#include "telemetry/telemetry.h"
#include "kws_tflm/kws_infer.h"

/* S1's DSP task entry point (defined in micro_vad/dsp_task.c) */
extern void dsp_task_start(void);

static const char *TAG = "MAIN";

/* ── WiFi Config — CHANGE THESE ─────────────────────────────── */
#define WIFI_SSID     "C_BLOCK_F2"      // 2.4GHz band (ESP32 cannot do 5GHz)
#define WIFI_PASSWORD "welcome2ars"      // WiFi password

static QueueHandle_t     s_trigger_queue    = NULL;
static QueueHandle_t     s_transport_queue  = NULL;  // dedicated PCM queue for transport task
static EventGroupHandle_t s_wifi_events      = NULL;
#define WIFI_CONNECTED_BIT  BIT0

/* Transport queue depth: 32 frames = 640ms buffer */
#define TRANSPORT_QUEUE_DEPTH  32

/* Read one 320-sample frame from the transport-dedicated PCM queue */
static bool transport_read_frame(int16_t *out, TickType_t timeout_ticks) {
    return xQueueReceive(s_transport_queue, out, timeout_ticks) == pdTRUE;
}

void notify_kws_trigger(float score) {
    if (xQueueSend(s_trigger_queue, &score, 0) != pdTRUE) {
        float dummy;
        xQueueReceive(s_trigger_queue, &dummy, 0);
        xQueueSend(s_trigger_queue, &score, 0);
    }
}

static void transport_task(void *arg) {
    (void)arg;
    adpcm_state_t adpcm_state;
    static int16_t preroll_buf[PREROLL_SAMPLES];
    static uint8_t adpcm_out[160];

    /* Smart silence endpointing thresholds
     * --- these are the knobs you can tune ---
     * SPEECH_RMS_THRESHOLD : RMS above this counts as "speech heard"
     * SILENCE_STOP_MS      : consecutive silence ms AFTER speech to stop (5 s)
     * MAX_STREAM_MS        : hard ceiling — never stream longer than this
     */
    #define SPEECH_RMS_THRESHOLD  250      // int16 RMS (~1% of full scale)
    #define SILENCE_STOP_FRAMES   250      // 250 x 20ms = 5 000ms (5 s)
    #define MAX_STREAM_MS         12000    // 12 s absolute max
    #define FRAME_MS              20
    #define MAX_FRAMES            (MAX_STREAM_MS / FRAME_MS)

    ESP_LOGI(TAG, "Transport task started — waiting for KWS trigger");

    while (1) {
        float score = 0.0f;

        if (xQueueReceive(s_trigger_queue, &score, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        ESP_LOGI(TAG, "KWS TRIGGER! score=%.3f — starting stream", score);
        telemetry_record_trigger();

        adpcm_state_reset(&adpcm_state);
        kws_reset_trigger();

        if (!naad_tcp_is_connected()) {
            /* Wait up to 15s for WiFi to be up before attempting TCP */
            EventBits_t bits = xEventGroupWaitBits(
                s_wifi_events, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE,
                pdMS_TO_TICKS(15000)
            );
            if (!(bits & WIFI_CONNECTED_BIT)) {
                ESP_LOGW(TAG, "WiFi not ready — skipping trigger");
                preroll_reset();
                continue;
            }
            ESP_LOGI(TAG, "Connecting to server %s:%d ...", SERVER_IP, SERVER_PORT);
            if (!naad_tcp_connect()) {
                ESP_LOGE(TAG, "TCP connect failed — retrying in 1s");
                preroll_reset();
                vTaskDelay(pdMS_TO_TICKS(1000));
                continue;
            }
        }

        /* SEND PRE-ROLL AUDIO FIRST */
        int preroll_samples = 0;
        preroll_flush(preroll_buf, &preroll_samples);
        ESP_LOGI(TAG, "Sending pre-roll: %d samples (%d ms)",
                 preroll_samples, preroll_samples / 16);

        for (int i = 0; i + 320 <= preroll_samples; i += 320) {
            adpcm_encode_block(&adpcm_state, &preroll_buf[i], 320, adpcm_out);
            uint32_t ts = (uint32_t)(esp_timer_get_time() / 1000);
            naad_tcp_send_frame(adpcm_out, 160, ts);
        }

        telemetry_record_first_packet();

        /* STREAM LIVE AUDIO WITH SMART SILENCE ENDPOINTING
         *
         * Phase 1 (speech not yet heard): ignore silence, keep streaming.
         * Phase 2 (speech heard):         count consecutive silence frames.
         *                                 If >= SILENCE_STOP_FRAMES (5s), stop.
         * This means a 2-3 second human pause is fully tolerated.
         * The stream only cuts when truly 5 consecutive seconds of silence occur.
         */
        static int16_t live_frame[320];
        int silence_frames   = 0;
        bool speech_heard    = false;  // have we seen any speech yet?

        ESP_LOGI(TAG, "Streaming live audio (smart endpoint, max %dms) ...", MAX_STREAM_MS);

        for (int f = 0; f < MAX_FRAMES; f++) {
            /* Read from transport's dedicated queue (100ms timeout = 5 frame misses) */
            if (!transport_read_frame(live_frame, pdMS_TO_TICKS(100))) {
                ESP_LOGW(TAG, "Transport audio timeout at frame %d — stopping", f);
                break;
            }

            /* Compute RMS energy of this frame */
            int64_t energy = 0;
            for (int s = 0; s < 320; s++) {
                energy += (int64_t)live_frame[s] * live_frame[s];
            }
            float rms = 0.0f;
            if (energy > 0) rms = sqrtf((float)(energy / 320));

            bool is_silence = (rms < SPEECH_RMS_THRESHOLD);

            if (!is_silence) {
                speech_heard  = true;  // latch: speech detected at least once
                silence_frames = 0;    // reset silence counter on any speech
            } else if (speech_heard) {
                /* Only count silence AFTER we've heard speech at least once */
                silence_frames++;
                if (silence_frames >= SILENCE_STOP_FRAMES) {
                    ESP_LOGI(TAG, "Smart endpoint: %dms silence after speech — stopping stream",
                             silence_frames * FRAME_MS);
                    /* Encode and send the current frame before stopping */
                    adpcm_encode_block(&adpcm_state, live_frame, 320, adpcm_out);
                    uint32_t ts = (uint32_t)(esp_timer_get_time() / 1000);
                    naad_tcp_send_frame(adpcm_out, 160, ts);
                    break;
                }
            }
            /* else: silence before any speech — keep streaming (pre-speech window) */

            adpcm_encode_block(&adpcm_state, live_frame, 320, adpcm_out);
            uint32_t ts = (uint32_t)(esp_timer_get_time() / 1000);
            if (!naad_tcp_send_frame(adpcm_out, 160, ts)) {
                ESP_LOGW(TAG, "TCP send failed at frame %d — stopping stream", f);
                break;
            }

            if (f % 25 == 0) {
                ESP_LOGD(TAG, "Stream frame %d, rms=%.0f, silence_frames=%d, speech=%d",
                         f, (double)rms, silence_frames, (int)speech_heard);
            }
        }

        ESP_LOGI(TAG, "Stream complete. Disconnecting socket.");
        naad_tcp_disconnect();
        preroll_reset();

        /* Flush queued duplicate triggers accumulated during streaming */
        float dummy_score;
        while (xQueueReceive(s_trigger_queue, &dummy_score, 0) == pdTRUE) {}

        /* Flush old transport queue frames */
        int16_t dummy_frame[320];
        while (xQueueReceive(s_transport_queue, dummy_frame, 0) == pdTRUE) {}
    }
}

static void wifi_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data) {
    (void)arg;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        /* Clear connected bit and retry */
        xEventGroupClearBits(s_wifi_events, WIFI_CONNECTED_BIT);
        ESP_LOGW(TAG, "WiFi disconnected — retrying...");
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "✅ WiFi connected! IP: " IPSTR, IP2STR(&event->ip_info.ip));
        /* Signal all tasks that WiFi is up */
        xEventGroupSetBits(s_wifi_events, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init(void) {
    s_wifi_events = xEventGroupCreate();

    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                               &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                               &wifi_event_handler, NULL));

    wifi_config_t wifi_cfg = {
        .sta = {
            .ssid     = WIFI_SSID,
            .password = WIFI_PASSWORD,
            /* Improve roaming stability */
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
            .pmf_cfg = { .capable = true, .required = false },
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_cfg));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "WiFi connecting to SSID: %s (waiting up to 30s)...", WIFI_SSID);
    /* Block until connected or 30s timeout — don't start tasks blind */
    EventBits_t bits = xEventGroupWaitBits(
        s_wifi_events, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, pdMS_TO_TICKS(30000)
    );
    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "WiFi ready!");
    } else {
        ESP_LOGW(TAG, "WiFi timeout — continuing anyway (will retry in background)");
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "=== PROJECT NAAD FIRMWARE STARTING ===");

    wifi_init();
    /* Create dedicated PCM queue for transport task (separate from DSP queue) */
    s_transport_queue = xQueueCreate(TRANSPORT_QUEUE_DEPTH, AUDIO_DMA_SAMPLES * sizeof(int16_t));
    if (s_transport_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create transport PCM queue!");
    }

    preroll_init();
    audio_i2s_start(s_transport_queue);  // capture pushes to BOTH internal (DSP) and this queue
    ESP_LOGI(TAG, "I2S audio capture started");

    kws_init();
    ESP_LOGI(TAG, "KWS inference engine initialized");

    s_trigger_queue = xQueueCreate(4, sizeof(float));

    xTaskCreatePinnedToCore(
        transport_task, "Transport", 8192, NULL,
        6,    // priority 6 — ABOVE DSP (priority 5) so it gets CPU immediately on trigger
        NULL, 0
    );

    dsp_task_start();
    telemetry_start();

    ESP_LOGI(TAG, "=== ALL TASKS STARTED — Speak the keyword! ===");
    ESP_LOGI(TAG, "Server: %s:%d", SERVER_IP, SERVER_PORT);
    ESP_LOGI(TAG, "KWS threshold: %.2f", (double)KWS_THRESHOLD);
}
