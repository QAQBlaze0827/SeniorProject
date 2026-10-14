import cv2
import numpy as np

ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_norm(landmark, image_size=112):
    lmk = np.asarray(landmark, dtype=np.float32)
    if lmk.shape != (5, 2):
        raise ValueError(f"expected 5 landmarks with shape (5, 2), got {lmk.shape}")
    ratio = float(image_size) / 112.0
    dst = ARCFACE_DST * ratio
    m, _ = cv2.estimateAffinePartial2D(lmk, dst, method=cv2.LMEDS)
    if m is None:
        m = cv2.estimateAffinePartial2D(lmk, dst, method=cv2.RANSAC)[0]
    return m


def norm_crop(img, landmark, image_size=112):
    m = estimate_norm(landmark, image_size)
    return cv2.warpAffine(img, m, (image_size, image_size), borderValue=0.0)
