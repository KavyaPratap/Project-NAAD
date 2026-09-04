# Owner: S3
# Module: Utterance Endpointing (silence-based segmentation)
# Purpose: Detect when the speaker has finished talking
#          so we send exactly one utterance to ASR (not continuous audio)
# Last validated: [date, commit hash]

import numpy as np


# ── Tunable thresholds — adjust based on your mic/room ────────
SILENCE_THRESHOLD_RMS = 300    # RMS below this = silence (tune from your audio levels)
SILENCE_FRAMES        = 15     # 15 x 20ms = 300ms silence = utterance end
MIN_UTTERANCE_FRAMES  = 25     # Require at least 500ms of audio before we can endpoint


class Endpointer:
    """
    Simple energy-based endpointer.
    Call update() with every decoded PCM block (20ms = 320 samples).
    Returns True when utterance end is detected.

    Typical usage:
        ep = Endpointer()
        for pcm_block in stream:
            if ep.update(pcm_block):
                # utterance complete — send to ASR
                ep.reset()
                break
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset for a new utterance."""
        self._silence_count = 0
        self._total_frames  = 0

    def update(self, pcm_samples: list) -> bool:
        """
        Process one PCM frame (int16 list, typically 320 samples = 20ms).
        Returns True if end-of-utterance detected.
        """
        self._total_frames += 1
        arr = np.array(pcm_samples, dtype=np.int16)
        rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))

        if rms < SILENCE_THRESHOLD_RMS:
            self._silence_count += 1
        else:
            self._silence_count = 0  # Reset silence counter on speech

        # Only endpoint if we have enough audio AND enough trailing silence
        if (self._silence_count >= SILENCE_FRAMES and
                self._total_frames >= MIN_UTTERANCE_FRAMES):
            return True

        return False

    def is_speech(self, pcm_samples: list) -> bool:
        """Check if current frame is speech (for logging/debugging)."""
        arr = np.array(pcm_samples, dtype=np.int16)
        rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
        return rms >= SILENCE_THRESHOLD_RMS

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def silence_frames(self) -> int:
        return self._silence_count


# ── Self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    ep = Endpointer()

    # Simulate: 30 speech frames then 20 silence frames
    speech_frame  = (np.ones(320) * 1000).astype(np.int16).tolist()
    silence_frame = (np.zeros(320)).astype(np.int16).tolist()

    triggered = False
    for i in range(50):
        frame = speech_frame if i < 30 else silence_frame
        if ep.update(frame):
            print(f"✅ Endpoint detected at frame {i} (after {ep.silence_frames} silence frames)")
            triggered = True
            break

    if not triggered:
        print("❌ Endpoint NOT detected — check thresholds")
    else:
        print(f"   Total frames processed: {ep.total_frames}")
        print(f"   SILENCE_THRESHOLD_RMS = {SILENCE_THRESHOLD_RMS}")
        print(f"   SILENCE_FRAMES        = {SILENCE_FRAMES}")
        print(f"   MIN_UTTERANCE_FRAMES  = {MIN_UTTERANCE_FRAMES}")
