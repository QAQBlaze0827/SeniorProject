import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

IDX_TO_EMOTION = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happiness",
    5: "Neutral",
    6: "Sadness",
    7: "Surprise",
}

FOUR_CLASS_INDEX = {"happy": 4, "angry": 0, "sad": 6, "neutral": 5}


class FaceEmotionExtractor:
    def __init__(self, onnx_path, provider="CPUExecutionProvider", img_size=224):
        ort.set_default_logger_severity(3)
        model = onnx.load(onnx_path)
        graph = model.graph
        gemm = graph.node[-1]
        if gemm.op_type != "Gemm" or len(gemm.input) < 3:
            raise RuntimeError(f"unexpected last node: {gemm.op_type}")

        feature_name = gemm.input[0]
        weight = next((t for t in graph.initializer if t.name == gemm.input[1]), None)
        bias = next((t for t in graph.initializer if t.name == gemm.input[2]), None)
        if weight is None or bias is None:
            raise RuntimeError("classifier weights/bias not found in graph")

        self.classifier_weights = numpy_helper.to_array(weight).astype(np.float32)
        self.classifier_bias = numpy_helper.to_array(bias).astype(np.float32)

        graph.node.remove(gemm)
        graph.output.remove(graph.output[0])
        graph.output.append(
            helper.make_tensor_value_info(
                feature_name,
                TensorProto.FLOAT,
                [None, self.classifier_weights.shape[1]],
            )
        )
        self.session = ort.InferenceSession(
            model.SerializeToString(), providers=[provider]
        )
        self.input_name = self.session.get_inputs()[0].name

        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, face_bgr):
        x = cv2.resize(face_bgr, (self.img_size, self.img_size)).astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def extract_features(self, face_bgr):
        return self.session.run(None, {self.input_name: self._preprocess(face_bgr)})[0]

    def logits8(self, face_bgr):
        features = self.extract_features(face_bgr)
        return features @ self.classifier_weights.T + self.classifier_bias

    @staticmethod
    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def probs8(self, face_bgr):
        return self.softmax(self.logits8(face_bgr))[0]

    def probs4(self, face_bgr):
        p8 = self.probs8(face_bgr)
        p4 = {name: float(p8[idx]) for name, idx in FOUR_CLASS_INDEX.items()}
        total = sum(p4.values())
        if total <= 0:
            return {name: 0.25 for name in FOUR_CLASS_INDEX}
        return {name: value / total for name, value in p4.items()}

    @staticmethod
    def top4(probs4):
        return max(probs4, key=probs4.get)
