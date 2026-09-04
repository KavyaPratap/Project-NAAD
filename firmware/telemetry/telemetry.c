/* Owner: S3
 * Module: Telemetry Task
 * Last tested: [date, firmware commit]
 */
#include "telemetry.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "TELEMETRY";

static int64_t s_trigger_time_us     = 0;
static int64_t s_first_packet_us     = 0;
static uint32_t s_trigger_count      = 0;

static void telemetry_task(void *arg) {
    (void)arg;
    uint32_t report_num = 0;

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(5000));  // Report every 5 seconds
        report_num++;

        /* ── Heap metrics ──────────────────────────────────────── */
        uint32_t free_heap     = esp_get_free_heap_size();
        uint32_t min_free_heap = esp_get_minimum_free_heap_size();
        uint32_t largest_block = heap_caps_get_largest_free_block(MALLOC_CAP_DEFAULT);

        /* ── Print report ─────────────────────────────────────── */
        ESP_LOGI(TAG, "=== TELEMETRY REPORT #%lu ===", (unsigned long)report_num);
        ESP_LOGI(TAG, "  RAM free:         %5lu bytes  (%lu KB)", (unsigned long)free_heap,     (unsigned long)(free_heap / 1024));
        ESP_LOGI(TAG, "  RAM min ever:     %5lu bytes  (%lu KB)", (unsigned long)min_free_heap, (unsigned long)(min_free_heap / 1024));
        ESP_LOGI(TAG, "  Largest block:    %5lu bytes  (%lu KB)", (unsigned long)largest_block, (unsigned long)(largest_block / 1024));
        ESP_LOGI(TAG, "  KWS triggers:     %lu total",  (unsigned long)s_trigger_count);

        if (s_trigger_count > 0 && s_first_packet_us > s_trigger_time_us) {
            uint32_t latency_ms = (uint32_t)((s_first_packet_us - s_trigger_time_us) / 1000);
            ESP_LOGI(TAG, "  Last trig->pkt:   %lu ms", (unsigned long)latency_ms);
        }

        ESP_LOGI(TAG, "=============================");
    }
}

void telemetry_start(void) {
    xTaskCreatePinnedToCore(
        telemetry_task,
        "Telemetry",
        4096,
        NULL,
        2,    // Low priority — doesn't interfere with audio/DSP
        NULL,
        0     // Core 0
    );
    ESP_LOGI(TAG, "Telemetry task started (reports every 5s)");
}

void telemetry_record_trigger(void) {
    s_trigger_time_us = esp_timer_get_time();
    s_trigger_count++;
    ESP_LOGI(TAG, "KWS trigger #%lu recorded at %lld us", (unsigned long)s_trigger_count, s_trigger_time_us);
}

void telemetry_record_first_packet(void) {
    s_first_packet_us = esp_timer_get_time();
    ESP_LOGI(TAG, "First packet sent at %lld us", s_first_packet_us);
}

uint32_t telemetry_get_trigger_to_packet_ms(void) {
    if (s_first_packet_us <= s_trigger_time_us) return 0;
    return (uint32_t)((s_first_packet_us - s_trigger_time_us) / 1000);
}
