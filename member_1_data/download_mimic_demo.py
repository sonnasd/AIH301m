import os
import urllib.request
import sys

BASE_URL = "https://physionet.org/files/mimiciii-demo/1.4/"

DEMO_FILES = [
    "ADMISSIONS.csv",
    "PATIENTS.csv",
    "ICUSTAYS.csv",
    "DIAGNOSES_ICD.csv",
    "MICROBIOLOGYEVENTS.csv",
    "D_LABITEMS.csv",
    "D_ITEMS.csv"
]

def download_demo_tables(target_dir="data/mimic_demo", files=None):
    os.makedirs(target_dir, exist_ok=True)
    if files is None:
        files = DEMO_FILES
        
    downloaded = []
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for filename in files:
        dest_path = os.path.join(target_dir, filename)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            downloaded.append(dest_path)
            continue
            
        url = BASE_URL + filename
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
                chunk_size = 64 * 1024
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
            downloaded.append(dest_path)
        except Exception as exc:
            pass
            
    return downloaded

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/mimic_demo"
    paths = download_demo_tables(out_dir)
    print(f"Downloaded {len(paths)} files to {out_dir}")
