import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from config import load_config
from face_recognition.detector import SCRFD
from face_emotion.extractor import FaceEmotionExtractor


def main():
    image = sys.argv[1] if len(sys.argv) > 1 else "tests/data/t1.jpg"
    cfg = load_config()

    det = SCRFD(cfg["face_recognition"]["det_model"])
    det.prepare(input_size=tuple(cfg["face_recognition"]["det_size"]))
    extractor = FaceEmotionExtractor(
        cfg["face_emotion"]["onnx_model"],
        provider=cfg["face_emotion"]["provider"],
        img_size=cfg["face_emotion"]["img_size"],
    )

    frame = cv2.imread(image)
    if frame is None:
        raise SystemExit(f"cannot read image: {image}")

    dets, _ = det.detect(frame)
    print("faces detected:", dets.shape[0])
    for i in range(dets.shape[0]):
        x1, y1, x2, y2 = dets[i, :4].astype(int)
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        feats = extractor.extract_features(crop)
        probs8 = extractor.probs8(crop)
        probs4 = extractor.probs4(crop)
        top = extractor.top4(probs4)
        print(
            f"  [{i}] feat_dim={feats.shape[1]} "
            f"top4={top} "
            f"p4={{"
            + ", ".join(f"{k}:{v:.2f}" for k, v in probs4.items())
            + "} "
            f"p8_max={probs8.max():.2f}"
        )

    print("OK")


if __name__ == "__main__":
    main()
