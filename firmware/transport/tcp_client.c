/* Owner: S3
 * Module: TCP Client — ADPCM stream transport
 * Last tested: [date, firmware commit]
 */
#include "tcp_client.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include <string.h>
#include <arpa/inet.h>
#include <errno.h>

static const char *TAG = "TCP_CLIENT";

static int s_sock = -1;  // -1 = not connected

#define HEADER_SIZE 17

static uint32_t s_seq_num = 0;

static void write_u32_be(uint8_t *buf, uint32_t v) {
    buf[0] = (v >> 24) & 0xFF;
    buf[1] = (v >> 16) & 0xFF;
    buf[2] = (v >>  8) & 0xFF;
    buf[3] = (v      ) & 0xFF;
}

static void write_u16_be(uint8_t *buf, uint16_t v) {
    buf[0] = (v >> 8) & 0xFF;
    buf[1] = (v     ) & 0xFF;
}

bool naad_tcp_connect(void) {
    if (s_sock >= 0) {
        ESP_LOGW(TAG, "Already connected");
        return true;
    }

    struct sockaddr_in dest;
    dest.sin_family      = AF_INET;
    dest.sin_port        = htons(SERVER_PORT);
    dest.sin_addr.s_addr = inet_addr(SERVER_IP);

    s_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s_sock < 0) {
        ESP_LOGE(TAG, "Socket creation failed: errno=%d", errno);
        return false;
    }

    int ret = connect(s_sock, (struct sockaddr *)&dest, sizeof(dest));
    if (ret != 0) {
        ESP_LOGE(TAG, "Connect to %s:%d failed: errno=%d", SERVER_IP, SERVER_PORT, errno);
        close(s_sock);
        s_sock = -1;
        return false;
    }

    s_seq_num = 0;
    ESP_LOGI(TAG, "Connected to %s:%d", SERVER_IP, SERVER_PORT);
    return true;
}

void naad_tcp_disconnect(void) {
    if (s_sock >= 0) {
        shutdown(s_sock, SHUT_RDWR);
        close(s_sock);
        s_sock = -1;
        ESP_LOGI(TAG, "Disconnected. Total packets sent: %lu", (unsigned long)s_seq_num);
    }
}

bool naad_tcp_send_frame(const uint8_t *adpcm_payload, uint16_t payload_len, uint32_t timestamp_ms) {
    if (s_sock < 0) {
        ESP_LOGW(TAG, "Not connected — cannot send");
        return false;
    }

    /* Build header */
    uint8_t header[HEADER_SIZE];
    header[0]  = PACKET_MAGIC_0;
    header[1]  = PACKET_MAGIC_1;
    header[2]  = PACKET_VERSION;
    write_u32_be(&header[3],  s_seq_num);
    write_u32_be(&header[7],  timestamp_ms);
    header[11] = CODEC_IMA_ADPCM;
    write_u16_be(&header[12], 16000);  // sample rate
    header[14] = 0x01;                 // mono
    write_u16_be(&header[15], payload_len);

    /* Send header */
    int sent = send(s_sock, header, HEADER_SIZE, 0);
    if (sent != HEADER_SIZE) {
        ESP_LOGE(TAG, "Header send failed: sent=%d errno=%d", sent, errno);
        naad_tcp_disconnect();
        return false;
    }

    /* Send payload */
    int sent_payload = send(s_sock, adpcm_payload, payload_len, 0);
    if (sent_payload != payload_len) {
        ESP_LOGE(TAG, "Payload send failed: sent=%d errno=%d", sent_payload, errno);
        naad_tcp_disconnect();
        return false;
    }

    s_seq_num++;

    if (s_seq_num % 50 == 0) {
        ESP_LOGI(TAG, "Packets sent: %lu", (unsigned long)s_seq_num);
    }

    return true;
}

bool naad_tcp_is_connected(void) {
    return s_sock >= 0;
}
