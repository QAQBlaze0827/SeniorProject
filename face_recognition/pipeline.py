from config import load_config
from face_recognition.detector import SCRFD
from face_recognition.align import norm_crop
from face_recognition.embedder import ArcFaceEmbedder


class FacePipeline:
    def __init__(self, cfg=None):
        cfg = cfg or load_config()
        fr = cfg["face_recognition"]
        self.detector = SCRFD(fr["det_model"], provider=fr["provider"])
        self.detector.prepare(
            input_size=tuple(fr["det_size"]),
            det_thresh=fr.get("det_thresh", 0.5),
            nms_thresh=fr.get("nms_thresh", 0.4),
        )
        self.embedder = ArcFaceEmbedder(fr["rec_model"], provider=fr["provider"])
        self.similarity_threshold = fr.get("similarity_threshold", 0.35)

    def process(self, frame):
        dets, kpss = self.detector.detect(frame)
        results = []
        for i in range(dets.shape[0]):
            bbox = dets[i, :4].astype(int)
            score = float(dets[i, 4])
            kps = kpss[i] if kpss is not None else None
            aligned = self._aligned(frame, bbox, kps)
            embedding = self.embedder.get_embedding(aligned)
            results.append(
                {"bbox": bbox, "score": score, "kps": kps, "embedding": embedding}
            )
        return results

    @staticmethod
    def _aligned(frame, bbox, kps):
        if kps is not None:
            return norm_crop(frame, kps, 112)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return frame
        return crop
