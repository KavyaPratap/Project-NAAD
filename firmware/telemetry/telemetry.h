/* Owner: S3
 * Module: Telemetry Task — RAM/CPU/Latency logging
 * Logs system health to UART every 5 seconds
 * Results saved to: docs/test_results/ram_measurement.md
 * Last tested: [date, firmware commit]
 */
#pragma once
#include <stdint.h>

/** Start telemetry logging task (call once at boot, after all other tasks started) */
void telemetry_start(void);

/** Record timestamp of KWS trigger (call from transport task on trigger) */
void telemetry_record_trigger(void);

/** Record timestamp of first TCP packet sent (call from tcp_client on first send) */
void telemetry_record_first_packet(void);

/** Returns end-to-end latency of last trigger->first_packet in ms */
uint32_t telemetry_get_trigger_to_packet_ms(void);
