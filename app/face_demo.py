import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from config import load_config
from capture.video_source import VideoSource
from face_recognition.pipeline import FacePipeline
from face_recognition.gallery import FaceGallery


def draw(frame, bbox, label, color):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    cfg = load_config()
    cam = cfg["camera"]
    pipeline = FacePipeline(cfg)
    gallery = FaceGallery(cfg["face_recognition"]["db_path"])
    threshold = pipeline.similarity_threshold

    with VideoSource(
        index=cam["index"],
        width=cam["width"],
        height=cam["height"],
        fps=cam["fps"],
    ) as video:
        prev = time.time()
        fps = 0.0
        print("press q to quit")

        while True:
            frame = video.read()
            if frame is None:
                break

            results = pipeline.process(frame)
            for r in results:
                name, score = gallery.match(r["embedding"], threshold)
                color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
                draw(frame, r["bbox"], f"{name} {score:.2f}", color)

            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.putText(
                frame,
                f"FPS {fps:4.1f}  faces {len(results)}  gallery {gallery.count()}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("face recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    gallery.close()


if __name__ == "__main__":
    main()
