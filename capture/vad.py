import numpy as np


class EnergyVAD:
    def __init__(self, sample_rate=16000, frame_ms=30, threshold=0.01):
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self.threshold = threshold

    def rms(self, chunk):
        if chunk is None or len(chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk))))

    def is_speech(self, chunk):
        return self.rms(chunk) >= self.threshold
