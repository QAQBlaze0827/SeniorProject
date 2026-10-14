import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import sounddevice as sd

from config import load_config


def main():
    cfg = load_config()
    print("config ok:", cfg["emotions"])
    print("opencv:", cv2.__version__)
    print("numpy:", np.__version__)
    print("sounddevice:", sd.__version__)

    print("audio devices:")
    for i, d in enumerate(sd.query_devices()):
        print("  ", i, d["name"], "in:", d["max_input_channels"])
    try:
        default_in, _ = sd.default.device
        print("default input device:", default_in)
    except Exception as exc:
        print("no default input device:", exc)

    print("cameras:")
    for i in range(3):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        if ok:
            ret, frame = cap.read()
            print("   index", i, "ok", "read:", ret, None if frame is None else frame.shape)
        cap.release()


if __name__ == "__main__":
    main()
