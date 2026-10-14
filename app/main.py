import cv2
import numpy as np

from config import load_config
from capture.video_source import VideoSource
from capture.audio_source import AudioSource


def main():
    cfg = load_config()
    cam = cfg["camera"]
    aud = cfg["audio"]
    emotions = cfg["emotions"]

    with VideoSource(
        index=cam["index"],
        width=cam["width"],
        height=cam["height"],
        fps=cam["fps"],
    ) as video, AudioSource(
        sample_rate=aud["sample_rate"],
        channels=aud["channels"],
        block_ms=aud["block_ms"],
    ) as mic:
        level = 0.0
        print("press q to quit")

        while True:
            frame = video.read()
            if frame is None:
                break

            for chunk in mic.drain():
                rms = float(np.sqrt(np.mean(np.square(chunk))))
                level = 0.85 * level + 0.15 * rms

            cv2.putText(
                frame,
                f"emotions: {', '.join(emotions)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            bar_w = int(min(level * 800.0, 1.0) * 200)
            cv2.rectangle(frame, (10, 40), (210, 55), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, 40), (10 + bar_w, 55), (0, 200, 255), -1)
            cv2.putText(
                frame,
                f"mic level {level:.3f}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("multimodal emotion capture", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
