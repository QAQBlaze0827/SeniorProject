import cv2
import numpy as np
import onnxruntime as ort


class ArcFaceEmbedder:
    def __init__(self, model_file, input_size=(112, 112), provider="CPUExecutionProvider"):
        self.session = ort.InferenceSession(model_file, providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = tuple(input_size)

    def get_embedding(self, face_bgr):
        img = cv2.resize(face_bgr, self.input_size)
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / 127.5, self.input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        emb = self.session.run([self.output_name], {self.input_name: blob})[0]
        emb = np.asarray(emb, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(emb))
        if norm > 0.0:
            emb = emb / norm
        return emb
