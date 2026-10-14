import sys
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config


BUFFALO_SC_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
)
WANTED = ["det_500m.onnx", "w600k_mbf.onnx"]


def main():
    cfg = load_config()
    dest = Path(cfg["paths"]["models_dir"]) / "buffalo_sc"
    dest.mkdir(parents=True, exist_ok=True)

    if all((dest / name).exists() for name in WANTED):
        print("models already present:", dest)
        return

    print("downloading", BUFFALO_SC_URL)
    data = urllib.request.urlopen(BUFFALO_SC_URL, timeout=120).read()
    with zipfile.ZipFile(__import__("io").BytesIO(data)) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in WANTED:
                target = dest / name
                with zf.open(member) as src, open(target, "wb") as out:
                    out.write(src.read())
                print("extracted", target)

    missing = [name for name in WANTED if not (dest / name).exists()]
    if missing:
        raise SystemExit(f"missing after download: {missing}")
    print("done")


if __name__ == "__main__":
    main()
