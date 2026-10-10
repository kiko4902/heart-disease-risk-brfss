"""Download CVD_cleaned.csv from Kaggle into data/.

The dataset is public (CC0). Kaggle's download endpoint currently serves it
without authentication; if Kaggle credentials are available (KAGGLE_USERNAME
and KAGGLE_KEY, or ~/.kaggle/kaggle.json) they are sent as well. If the
download fails, the script prints manual instructions.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DATASET = "alphiree/cardiovascular-diseases-risk-prediction-dataset"
URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
FILENAME = "CVD_cleaned.csv"
SHA256 = "0c3c399c95c9df8ba65fa36821a95c8f27609508dc6927b6a6e7034f25a492c8"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

MANUAL = f"""
Could not download the dataset automatically.

Manual steps:
  1. Open https://www.kaggle.com/datasets/{DATASET}
  2. Click "Download" (a free Kaggle account may be required).
  3. Unzip and place {FILENAME} in: {DATA_DIR}

Or set up Kaggle API credentials (https://www.kaggle.com/docs/api):
  put kaggle.json in ~/.kaggle/ or export KAGGLE_USERNAME and KAGGLE_KEY, then rerun.
"""


def kaggle_credentials() -> tuple[str, str] | None:
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key
    config = Path.home() / ".kaggle" / "kaggle.json"
    if config.exists():
        content = json.loads(config.read_text())
        return content["username"], content["key"]
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download() -> bytes:
    request = urllib.request.Request(URL, headers={"User-Agent": "cvd-download/0.1"})
    credentials = kaggle_credentials()
    if credentials:
        token = base64.b64encode(":".join(credentials).encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> int:
    target = DATA_DIR / FILENAME
    if target.exists():
        if sha256(target) == SHA256:
            print(f"{target} already present, checksum OK.")
            return 0
        print(f"{target} exists but its checksum differs from the expected file; re-downloading.")

    print(f"Downloading {DATASET} ...")
    try:
        payload = download()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            DATA_DIR.mkdir(exist_ok=True)
            target.write_bytes(archive.read(FILENAME))
    except (urllib.error.URLError, zipfile.BadZipFile, KeyError, TimeoutError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        print(MANUAL, file=sys.stderr)
        return 1

    actual = sha256(target)
    if actual != SHA256:
        print(
            f"WARNING: checksum {actual} differs from the file this project was built on.\n"
            "The dataset may have been updated; numbers in the README may not reproduce exactly.",
            file=sys.stderr,
        )
    print(f"Saved {target} ({target.stat().st_size / 1e6:.1f} MB).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
