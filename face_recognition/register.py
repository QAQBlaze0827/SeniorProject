import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from config import load_config
from face_recognition.pipeline import FacePipeline
from face_recognition.gallery import FaceGallery


def build():
    cfg = load_config()
    pipeline = FacePipeline(cfg)
    gallery = FaceGallery(cfg["face_recognition"]["db_path"])
    return cfg, pipeline, gallery


def register_folder(pipeline, gallery, folder, name):
    folder = Path(folder)
    images = sorted(
        p
        for p in folder.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )
    if not images:
        print(f"no images in {folder}")
        return
    added = 0
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        results = pipeline.process(frame)
        if not results:
            print(f"  no face: {path.name}")
            continue
        best = max(results, key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]))
        gallery.add(name, best["embedding"])
        added += 1
    print(f"registered {added} face(s) for '{name}'")


def register_camera(cfg, pipeline, gallery, name, count):
    from capture.video_source import VideoSource

    cam = cfg["camera"]
    with VideoSource(cam["index"], cam["width"], cam["height"], cam["fps"]) as video:
        captured = 0
        print("press space to capture, q to quit")
        while added < count if False else added < count if False else True:
            frame = video.read()
            if frame is None:
                break
            results = pipeline.process(frame)
            for r in results:
                x1, y1, x2, y2 = r["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"name={name} captured={added}/{count}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("register", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and results:
                gallery.add(name, results[0]["embedding"])
                added += 1
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--folder", help="register every image in this folder")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    cfg, pipeline, gallery = build()
    if args.folder:
        register_folder(pipeline, gallery, args.folder, args.name)
    elif args.camera:
        register_camera(pipeline, gallery, args.name, args.count)
    else:
        parser.error("choose --folder or --camera")
    print("gallery size:", gallery.count(), "names:", gallery.names())


if __name__ == "__main__":
    main()
