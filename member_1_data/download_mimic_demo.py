import csv
import os
from pathlib import Path
import tempfile
import urllib.request
import sys

BASE_URL = "https://physionet.org/files/mimiciii-demo/1.4/"
DEMO_FILES = [
    "ADMISSIONS.csv", "PATIENTS.csv", "ICUSTAYS.csv", "DIAGNOSES_ICD.csv",
    "MICROBIOLOGYEVENTS.csv", "D_LABITEMS.csv", "D_ITEMS.csv",
]


def valid_csv(path):
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader)
            if "row_id" not in header or len(header) < 2:
                return False
            count = 0
            for row in reader:
                if len(row) != len(header):
                    return False
                count += 1
            return count > 0
    except (OSError, UnicodeError, csv.Error, StopIteration):
        return False


def download_demo_tables(target_dir="data/mimic_demo", files=None):
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for filename in DEMO_FILES if files is None else files:
        if filename not in DEMO_FILES:
            raise ValueError(f"Unsupported demo table: {filename}")
        dest = target / filename
        if dest.exists() and valid_csv(dest):
            downloaded.append(str(dest))
            continue
        temporary = None
        try:
            req = urllib.request.Request(BASE_URL + filename, headers={"User-Agent": "AIH301m-coursework"})
            with urllib.request.urlopen(req, timeout=30) as response:
                with tempfile.NamedTemporaryFile(dir=target, suffix=".part", delete=False) as handle:
                    temporary = Path(handle.name)
                    size = 0
                    while chunk := response.read(64 * 1024):
                        handle.write(chunk)
                        size += len(chunk)
                expected = response.headers.get("Content-Length")
                if expected is not None and size != int(expected):
                    raise OSError(f"Incomplete download: {filename}")
            if not valid_csv(temporary):
                raise ValueError(f"Invalid CSV download: {filename}")
            os.replace(temporary, dest)
            downloaded.append(str(dest))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return downloaded


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/mimic_demo"
    print(f"Available files: {len(download_demo_tables(out_dir))}")
