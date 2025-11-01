import os, json, hashlib, datetime, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
MANIFEST = ROOT / "manifest" / "datasets.json"

IST_OFFSET = datetime.timedelta(hours=5, minutes=30)

def md5_of_file(path, chunk=1024*1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def sniff_columns(path, nrows=5):
    p = Path(path)
    ext = p.suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(p, nrows=nrows)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(p, nrows=nrows)
        else:
            return {"columns": [], "sample_rows": []}
        cols = df.columns.tolist()
        rows = df.head(min(nrows, len(df))).to_dict(orient="records")
        return {"columns": cols, "sample_rows": rows}
    except Exception as e:
        return {"columns": [], "sample_rows": [], "error": str(e)}

def now_ist_iso():
    utc = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    ist = utc + IST_OFFSET
    return ist.replace(tzinfo=None).isoformat(timespec="seconds") + " IST"

def enrich_entry(entry):
    expected = {
        "crop_district": ["district_wise_raw_crop_data.csv", "district_wise_raw_crop_data.xlsx"],
        "imd_rainfall":  ["raw_rainfall_data.csv",  "raw_rainfall_data.xlsx"],
    }
    files = expected.get(entry.get("id"), [])
    found = None
    for f in files:
        p = RAW / f
        if p.exists():
            found = p
            break

    e = dict(entry)
    e.setdefault("catalog_url", entry.get("url", ""))
    e.setdefault("resource_url", "")
    e.setdefault("resource_id", "")
    e.setdefault("license", "")
    e.setdefault("attribution", "")
    e.setdefault("notes", "")

    if found:
        e["file_name"] = found.name
        e["format"] = found.suffix.lstrip(".").upper()
        e["filesize_bytes"] = found.stat().st_size
        e["md5"] = md5_of_file(found)
        sniff = sniff_columns(found)
        e["columns"] = sniff.get("columns", [])
        if sniff.get("error"):
            e["sniff_error"] = sniff["error"]
    else:
        e["file_name"] = ""
        e["format"] = ""
        e["filesize_bytes"] = 0
        e["md5"] = ""

    e["downloaded_at_ist"] = now_ist_iso()
    e["url"] = e.get("catalog_url", e.get("url", ""))

    return e

def main():
    if not MANIFEST.exists():
        print(f"ERROR: {MANIFEST} not found. Create it first.")
        sys.exit(1)

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print("ERROR: datasets.json must be a JSON array.")
        sys.exit(1)

    enriched = [enrich_entry(entry) for entry in data]

    MANIFEST.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated manifest written to {MANIFEST}")
    for e in enriched:
        print("—"*60)
        print(f"id: {e.get('id')}")
        print(f"file: {e.get('file_name')}  size: {e.get('filesize_bytes')} bytes  md5: {e.get('md5')}")
        print(f"catalog_url: {e.get('catalog_url')}")
        print(f"resource_url: {e.get('resource_url')}")
        print(f"resource_id: {e.get('resource_id')}")
        print(f"columns: {e.get('columns')[:8]}{' ...' if len(e.get('columns',[]))>8 else ''}")
        if e.get("sniff_error"):
            print(f"sniff_error: {e.get('sniff_error')}")
    print("Done.")

if __name__ == "__main__":
    main()
