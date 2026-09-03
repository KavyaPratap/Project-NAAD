# mocks/kws_mock.py
# Owner: S3
# Purpose: Simulates keyword trigger events so S3 can test the
#          pre-roll + streaming + server pipeline end-to-end.
#
# In firmware: replace the real KWS task with this Python trigger
#              when testing the server independently.

import time
import threading
import random

class MockKWSTrigger:
    """
    Simulates keyword trigger events from the firmware side.
    S3 uses this to test: trigger → pre-roll flush → ADPCM stream → server
    
    Replace with: real TFLM inference once S2 merges kws_infer.h
    """

    def __init__(self, trigger_interval_s: float = 10.0, jitter_s: float = 2.0):
        """
        trigger_interval_s: fire a fake trigger every N seconds
        jitter_s: add random jitter to make it realistic
        """
        self._interval = trigger_interval_s
        self._jitter   = jitter_s
        self._callback = None
        self._running  = False
        self._thread   = None

    def set_trigger_callback(self, callback):
        """callback(score: float) is called on trigger."""
        self._callback = callback

    def start(self):
        """Start firing fake triggers in background."""
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[kws_mock] Mock KWS trigger started — firing every ~10s")

    def stop(self):
        self._running = False

    def fire_now(self, score: float = 0.92):
        """Manually fire a trigger (useful for testing)."""
        if self._callback:
            self._callback(score)

    def _loop(self):
        while self._running:
            wait = self._interval + random.uniform(-self._jitter, self._jitter)
            time.sleep(max(1.0, wait))
            if self._running and self._callback:
                fake_score = random.uniform(0.87, 0.99)
                print(f"[kws_mock] TRIGGER! Score={fake_score:.3f}")
                self._callback(fake_score)


# ============================================================
# SWAP GUIDE for S3:
# ============================================================
# While S2 is training (server-side simulation of triggers):
#   kws = MockKWSTrigger()
#   kws.set_trigger_callback(on_trigger)
#   kws.start()
#
# Once S2 merges firmware/kws_tflm/ files:
#   Remove the mock. Trigger comes from the real ESP32 TCP stream header.
#   Server detects trigger from a "trigger=true" field in packet header.
# ============================================================
