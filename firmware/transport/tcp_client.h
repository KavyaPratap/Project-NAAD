/* Owner: S3
 * Module: TCP Client — ADPCM stream transport
 * Interface: Sends IMA-ADPCM framed packets to Python server
 * Packet format: magic(2)+version(1)+seq(4)+ts(4)+codec(1)+rate(2)+ch(1)+len(2)+payload(N)
 * Last tested: [date, firmware commit]
 */
#pragma once
#include <stdint.h>
#include <stdbool.h>

/* ── Server connection config — CHANGE THESE ─────────────────── */
#define SERVER_IP    "192.168.1.128"   // Laptop IP on C_BLOCK_F2_5G network
#define SERVER_PORT  5555

/* ── Packet protocol constants ───────────────────────────────── */
#define PACKET_MAGIC_0   0xAD
#define PACKET_MAGIC_1   0x9A
#define PACKET_VERSION   0x01
#define CODEC_IMA_ADPCM  0x01

/* ── Public API ─────────────────────────────────────────────── */

/** Connect to Python TCP server */
bool naad_tcp_connect(void);

/** Disconnect cleanly */
void naad_tcp_disconnect(void);

/** Send ADPCM packet */
bool naad_tcp_send_frame(const uint8_t *adpcm_payload, uint16_t payload_len, uint32_t timestamp_ms);

/** Returns true if connected */
bool naad_tcp_is_connected(void);
