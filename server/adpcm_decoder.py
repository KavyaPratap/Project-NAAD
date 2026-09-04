# Owner: S3
# Module: IMA-ADPCM Decoder (Python)
# CRITICAL: This MUST match firmware/adpcm/adpcm.c exactly.
#           Any mismatch in step_table, index_table, or nibble order = corrupted audio.
# Last validated: [date, commit hash]

STEP_TABLE = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34,
    37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143,
    157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494,
    544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552,
    1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428,
    4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635,
    13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
]
INDEX_TABLE = [-1, -1, -1, -1, 2, 4, 6, 8]


class AdpcmDecoder:
    """
    Stateful IMA-ADPCM decoder — one instance per audio stream.
    State (predictor + step_index) persists across packets; reset only
    when the stream starts or an error occurs.
    """

    def __init__(self) -> None:
        self.predictor: int = 0
        self.step_index: int = 0

    def reset(self, predictor: int = 0, step_index: int = 0) -> None:
        """Reset decoder state. Call at stream start or on reconnect."""
        self.predictor = predictor
        self.step_index = step_index

    def decode_sample(self, code: int) -> int:
        """
        Decode a single 4-bit ADPCM code to int16 PCM sample.
        Input:  code — integer 0-15 (4-bit)
        Output: int16 PCM sample in [-32768, 32767]
        """
        step = STEP_TABLE[self.step_index]
        diff = step >> 3
        if code & 4: diff += step
        if code & 2: diff += step >> 1
        if code & 1: diff += step >> 2
        if code & 8: diff = -diff
        self.predictor  = max(-32768, min(32767, self.predictor + diff))
        self.step_index = max(0, min(88, self.step_index + INDEX_TABLE[code & 7]))
        return self.predictor

    def decode(self, adpcm_bytes: bytes) -> list:
        """
        Decode a block of ADPCM bytes to int16 PCM samples.
        Input:  N bytes (each byte packs 2 samples: low nibble first, high nibble second)
        Output: list of int16 samples (length = 2*N)
        """
        samples = []
        for byte in adpcm_bytes:
            lo = byte & 0x0F
            hi = (byte >> 4) & 0x0F
            samples.append(self.decode_sample(lo))
            samples.append(self.decode_sample(hi))
        return samples


# ── Quick self-test ────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    # Encode in Python (mirrors C logic) then decode and check SNR
    def encode_py(pcm_list):
        pred, idx = 0, 0
        codes = []
        for s in pcm_list:
            step = STEP_TABLE[idx]
            diff = int(s) - pred
            code = 0
            if diff < 0: code = 8; diff = -diff
            if diff >= step:        code |= 4; diff -= step
            if diff >= step >> 1:   code |= 2; diff -= step >> 1
            if diff >= step >> 2:   code |= 1
            r = step >> 3
            if code & 4: r += step
            if code & 2: r += step >> 1
            if code & 1: r += step >> 2
            if code & 8: r = -r
            pred = max(-32768, min(32767, pred + r))
            idx  = max(0, min(88, idx + INDEX_TABLE[code & 7]))
            codes.append(code & 0xF)
        return codes

    # Generate test signal: 440Hz sine wave at 16kHz, 1 second
    t = np.linspace(0, 1, 16000)
    pcm = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)

    # Encode
    codes = encode_py(pcm.tolist())

    # Pack into bytes (low nibble first)
    packed = bytes([(codes[i]) | (codes[i+1] << 4) for i in range(0, len(codes)-1, 2)])

    # Decode
    dec = AdpcmDecoder()
    decoded = np.array(dec.decode(packed), dtype=np.int16)

    # SNR
    min_len = min(len(pcm), len(decoded))
    signal_power = np.mean(pcm[:min_len].astype(np.float64) ** 2)
    noise_power  = np.mean((pcm[:min_len] - decoded[:min_len]).astype(np.float64) ** 2) + 1e-9
    snr = 10 * np.log10(signal_power / noise_power)

    print(f"ADPCM round-trip SNR: {snr:.1f} dB")
    if snr > 30:
        print("✅ PASS — SNR > 30 dB (acceptable for speech)")
    else:
        print("❌ FAIL — SNR too low, check encoder/decoder mismatch!")
