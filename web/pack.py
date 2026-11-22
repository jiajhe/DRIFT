"""Package the built demo for any static web host, with no remote API dependency."""

from pathlib import Path
import json
import zipfile

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"
OUT = HERE.parent / "reports/marketflow-static.zip"
FILES = (
    "app.js", "archive.js", "config.js", "historical-demo.json", "index.html",
    "model-card.json", "multiday-replay.json", "multiday.js", "starmap-demo.json",
    "starmap.js", "style.css",
)


def pack():
    archive = json.loads((DIST / "historical-demo.json").read_bytes())
    result = json.loads((DIST / "model-card.json").read_bytes())["evaluation"]
    if (archive["checkpoint_sha256"], archive["gate"]) != (result["checkpoint_sha256"], result["gate"]):
        raise ValueError("Replay and evaluation reference different models")
    if "window.MARKETFLOW_API = \"\"" not in (DIST / "config.js").read_text():
        raise ValueError("Build the static archive with PUBLIC_MARKETFLOW_API_URL unset")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w") as out:
        for name in FILES:
            info = zipfile.ZipInfo(name)
            info.external_attr = 0o100644 << 16
            out.writestr(info, (DIST / name).read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None or tuple(z.namelist()) != FILES:
            raise IOError("Packaged assets failed their integrity check")
    tmp.replace(OUT)
    print(f"{OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    pack()
