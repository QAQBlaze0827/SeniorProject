import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from config import load_config
from face_recognition.pipeline import FacePipeline
from face_recognition.gallery import FaceGallery


def main():
    image = sys.argv[1] if len(sys.argv) > 1 else "tests/data/t1.jpg"
    cfg = load_config()
    pipeline = FacePipeline(cfg)

    frame = cv2.imread(image)
    if frame is None:
        raise SystemExit(f"cannot read image: {image}")

    results = pipeline.process(frame)
    print("faces detected:", len(results))
    for i, r in enumerate(results):
        x1, y1, x2, y2 = r["bbox"]
        print(
            f"  [{i}] bbox=({x1},{y1},{x2},{y2}) det={r['score']:.3f} "
            f"dim={r['embedding'].shape[0]} norm={np.linalg.norm(r['embedding']):.4f}"
        )

    if not results:
        raise SystemExit("no face detected")

    tmp_db = Path(tempfile.mkdtemp()) / "face_gallery.db"
    gallery = FaceGallery(tmp_db)
    gallery.add("self_test", results[0]["embedding"])
    name, score = gallery.match(
        results[0]["embedding"], cfg["face_recognition"]["similarity_threshold"]
    )
    print(f"self match -> {name} ({score:.4f})")
    gallery.close()

    if name != "self_test":
        raise SystemExit("FAIL: self match did not return registered name")
    print("OK")


if __name__ == "__main__":
    main()
