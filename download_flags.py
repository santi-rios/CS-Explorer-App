from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests

FLAGS_DIR = Path(__file__).resolve().parent / "www" / "flags"
FLAGS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("./data/data.parquet")

# Collect all 2-letter ISO codes from both solo rows and collaboration strings
iso_set = set()
for val in df["iso2c"].dropna().unique():
    for part in str(val).strip().split("-"):
        code = part.strip().lower()
        if len(code) == 2 and code.isalpha():
            iso_set.add(code)

print(f"Downloading {len(iso_set)} country flags into {FLAGS_DIR} ...")

def fetch_flag(iso: str):
    dest = FLAGS_DIR / f"{iso}.png"
    if dest.exists() and dest.stat().st_size > 100:
        return iso, "cached"
    url = f"https://flagcdn.com/w80/{iso}.png"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            return iso, "downloaded"
        return iso, f"HTTP {r.status_code}"
    except Exception as e:
        return iso, str(e)

with ThreadPoolExecutor(max_workers=12) as pool:
    results = list(pool.map(fetch_flag, sorted(iso_set)))

ok = sum(1 for _, status in results if status in ("downloaded", "cached"))
print(f"✅ Ready! {ok}/{len(iso_set)} flags available locally in ./www/flags/")