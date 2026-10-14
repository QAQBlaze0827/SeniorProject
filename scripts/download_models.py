import sys
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config

BUFFALO_SC_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
)
BUFFALO_SC_FILES = ["det_500m.onnx", "w600k_mbf.onnx"]

HSE_ONNX_URL = (
    "https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/main/"
    "models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx"
)


def download_buffalo_sc(models_dir):
    dest = Path(models_dir) / "buffalo_sc"
    dest.mkdir(parents=True, exist_ok=True)
    if all((dest / name).exists() for name in BUFFALO_SC_FILES):
        print("buffalo_sc: already present")
        return

    print("downloading buffalo_sc ...")
    data = urllib.request.urlopen(BUFFALO_SC_URL, timeout=180).read()
    with zipfile.ZipFile(BytesIO(data)) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in BUFFALO_SC_FILES:
                target = dest / name
                with zf.open(member) as src, open(target, "wb") as out:
                    out.write(src.read())
                print("  extracted", target)

    missing = [name for name in BUFFALO_SC_FILES if not (dest / name).exists()]
    if missing:
        raise SystemExit(f"missing after download: {missing}")


def download_hsemotion(models_dir):
    target = Path(models_dir) / "hsemotion" / "enet_b0_8_best_vgaf.onnx"
    if target.exists():
        print("hsemotion: already present")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    print("downloading hsemotion onnx ...")
    data = urllib.request.urlopen(HSE_ONNX_URL, timeout=180).read()
    target.write_bytes(data)
    print("  extracted", target)


def main():
    models_dir = load_config()["paths"]["models_dir"]
    download_buffalo_sc(models_dir)
    download_hsemotion(models_dir)
    print("done")


if __name__ == "__main__":
    main()
