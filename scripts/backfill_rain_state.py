# scripts/backfill_rain_state.py
from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "cleaned_data"
sub_parq = CLEAN / "rain_subdivision_year.parquet"
out_parq = CLEAN / "rain_state_year.parquet"

if not sub_parq.exists():
    raise SystemExit(f"Missing {sub_parq}. Run scripts/etl.py first.")

# Load subdivision data
con = duckdb.connect()
sub = con.execute(f"SELECT * FROM '{sub_parq.as_posix()}'").fetchdf()

# mapping: IMD subdivision -> state label
sub_to_state = {
    # Karnataka (3 subs)
    "Coastal Karnataka": "Karnataka",
    "North Interior Karnataka": "Karnataka",
    "South Interior Karnataka": "Karnataka",
    # 1:1 examples
    "Tamil Nadu": "Tamil Nadu",
    "Kerala": "Kerala",
    "Telangana": "Telangana",
    "Punjab": "Punjab",
    "Himachal Pradesh": "Himachal Pradesh",
    "Jammu & Kashmir": "Jammu & Kashmir",
    "Arunachal Pradesh": "Arunachal Pradesh",
    "Bihar": "Bihar",
    "Jharkhand": "Jharkhand",
    "Odisha": "Odisha",
    "Lakshadweep": "Lakshadweep",
    "Andaman & Nicobar Islands": "Andaman And Nicobar Islands",
    # Andhra Pradesh (2 subs)
    "Coastal Andhra Pradesh": "Andhra Pradesh",
    "Rayalaseema": "Andhra Pradesh",
    # Maharashtra (4 subs; “Konkan & Goa” spans two — we’ll include under Maharashtra/Goa combined)
    "Madhya Maharashtra": "Maharashtra",
    "Marathwada": "Maharashtra",
    "Vidarbha": "Maharashtra",
    "Konkan & Goa": "Maharashtra/Goa",
    # Gujarat (2 subs)
    "Gujarat Region": "Gujarat",
    "Saurashtra & Kutch": "Gujarat",
    # Uttar Pradesh (2 subs)
    "East Uttar Pradesh": "Uttar Pradesh",
    "West Uttar Pradesh": "Uttar Pradesh",
    # Delhi/Haryana/Chandigarh combined in IMD
    "Haryana Delhi & Chandigarh": "Haryana/Delhi/Chandigarh",
    # West Bengal splits
    "Gangetic West Bengal": "West Bengal",
    "Sub Himalayan West Bengal & Sikkim": "West Bengal/Sikkim",
    # North-East combined
    "Assam & Meghalaya": "Assam/Meghalaya",
    "Naga Mani Mizo Tripura": "NE-4 States",
    # MP, Rajasthan split
    "East Madhya Pradesh": "Madhya Pradesh",
    "West Madhya Pradesh": "Madhya Pradesh",
    "East Rajasthan": "Rajasthan",
    "West Rajasthan": "Rajasthan",
}

# normalize
sub["subdivision"] = sub["subdivision"].astype(str).str.strip().str.title()
sub["state"] = sub["subdivision"].map(sub_to_state)

# keep mapped rows only
cols = [c for c in ["state","year","annual_mm","jan_feb_mm","mar_may_mm","jun_sep_mm","oct_dec_mm"] if c in sub.columns]
mapped = sub.dropna(subset=["state"])[cols].copy()

# aggregate to state = simple mean across its subdivisions (approximation)
state = (
    mapped.groupby(["state","year"], as_index=False)
    .agg({
        "annual_mm": "mean",
        **({k:"mean" for k in ["jan_feb_mm","mar_may_mm","jun_sep_mm","oct_dec_mm"] if k in mapped.columns})
    })
    .sort_values(["state","year"])
)

state.to_parquet(out_parq, index=False)
print(f"Wrote {out_parq} with {len(state)} rows")
