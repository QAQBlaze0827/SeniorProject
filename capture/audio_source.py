import queue

import sounddevice as sd


class AudioSource:
    def __init__(self, sample_rate=16000, channels=1, block_ms=30):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = int(sample_rate * block_ms / 1000)
        self.queue = queue.Queue()
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        self.queue.put(indata.copy())

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.block_size,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()

    def read(self, timeout=1.0):
        try:
            return self.queue.get(timeout=timeout)
        except Exception:
            return None

    def drain(self, max_items=64):
        chunks = []
        while len(chunks) < max_items:
            try:
                chunks.append(self.queue.get_nowait())
            except Exception:
                break
        return chunks

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
