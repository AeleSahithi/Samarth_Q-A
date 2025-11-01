import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data"
CLEAN = ROOT / "cleaned_data"
CLEAN.mkdir(parents=True, exist_ok=True)

def read_any(path_csv: Path, alt_xlsx: Path) -> pd.DataFrame:
    if path_csv.exists():
        if path_csv.suffix.lower() == ".csv":
            return pd.read_csv(path_csv)
    if alt_xlsx.exists():
        return pd.read_excel(alt_xlsx)
    raise FileNotFoundError(f"Neither {path_csv.name} nor {alt_xlsx.name} found in {RAW}")

def std_text(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    # Title-case for names but preserve ALL CAPS acronyms reasonably
    return s.title()

def coerce_num(x):
    # convert to numeric safely; returns NaN if fails
    return pd.to_numeric(x, errors="coerce")

def etl_crops():
    crop_csv = RAW / "district_wise_raw_crop_data.csv"
    crop_xlsx = RAW / "district_wise_raw_crop_data.xlsx"
    df = read_any(crop_csv, crop_xlsx)
    # Normalize column names (lowercase, strip)
    df.columns = [c.strip().lower() for c in df.columns]

    # Expected columns you showed:
    # state_name, district_name, crop_year, season, crop, area_, production_
    # Some datasets have trailing spaces; also map common variants
    colmap = {
        "state_name": "state",
        "district_name": "district",
        "crop_year": "year",
        "season": "season",
        "crop": "crop",
        "area_": "area_ha",
        "production_": "production_tonnes",
        # fallbacks (if variants appear)
        "area": "area_ha",
        "production": "production_tonnes",
    }
    for k, v in list(colmap.items()):
        if k not in df.columns:
            # try to find close match (e.g., 'area ' with space)
            matches = [c for c in df.columns if c.replace(" ", "") == k.replace(" ", "")]
            if matches:
                colmap[matches[0]] = v

    # Apply mapping for available keys only
    apply_map = {k: v for k, v in colmap.items() if k in df.columns}
    df = df.rename(columns=apply_map)

    # Keep only relevant columns if present
    keep = [c for c in ["state", "district", "year", "season", "crop", "area_ha", "production_tonnes"] if c in df.columns]
    df = df[keep].copy()

    # Clean text fields
    for txt_col in ["state", "district", "season", "crop"]:
        if txt_col in df.columns:
            df[txt_col] = df[txt_col].map(std_text)

    # Coerce numeric
    if "year" in df.columns:
        df["year"] = coerce_num(df["year"]).astype("Int64")
    if "area_ha" in df.columns:
        df["area_ha"] = coerce_num(df["area_ha"])
    if "production_tonnes" in df.columns:
        df["production_tonnes"] = coerce_num(df["production_tonnes"])

    # Remove fully empty rows
    df = df.dropna(how="all")
    # Remove rows where year is missing
    if "year" in df.columns:
        df = df.dropna(subset=["year"])

    # Compute yield_kg_per_ha if possible (production in tonnes → kg)
    if {"area_ha", "production_tonnes"}.issubset(df.columns):
        # Avoid div-by-zero
        df["yield_kg_per_ha"] = np.where(
            (df["area_ha"] > 0) & ~df["area_ha"].isna() & ~df["production_tonnes"].isna(),
            (df["production_tonnes"] * 1000.0) / df["area_ha"],
            np.nan
        )
    else:
        df["yield_kg_per_ha"] = np.nan

    # Save district-level annual, per crop
    out_district = CLEAN / "crop_district_year.parquet"
    df.to_parquet(out_district, index=False)

    # Aggregate to state×year×crop
    group_keys = [k for k in ["state", "year", "crop"] if k in df.columns]
    if group_keys:
        agg = df.groupby(group_keys, dropna=True).agg(
            area_ha=("area_ha", "sum"),
            production_tonnes=("production_tonnes", "sum")
        ).reset_index()
        agg["yield_kg_per_ha"] = np.where(
            (agg["area_ha"] > 0) & ~agg["area_ha"].isna() & ~agg["production_tonnes"].isna(),
            (agg["production_tonnes"] * 1000.0) / agg["area_ha"],
            np.nan
        )
        out_state_crop = CLEAN / "crop_state_year.parquet"
        agg.to_parquet(out_state_crop, index=False)
    else:
        out_state_crop = None

    # Aggregate to state×year (totals across crops)
    group_keys2 = [k for k in ["state", "year"] if k in df.columns]
    if group_keys2:
        tot = df.groupby(group_keys2, dropna=True).agg(
            area_ha=("area_ha", "sum"),
            production_tonnes=("production_tonnes", "sum")
        ).reset_index()
        tot["yield_kg_per_ha"] = np.where(
            (tot["area_ha"] > 0) & ~tot["area_ha"].isna() & ~tot["production_tonnes"].isna(),
            (tot["production_tonnes"] * 1000.0) / tot["area_ha"],
            np.nan
        )
        out_state_total = CLEAN / "crop_state_totals.parquet"
        tot.to_parquet(out_state_total, index=False)
    else:
        out_state_total = None

    return {
        "district_file": str(out_district),
        "state_crop_file": str(out_state_crop) if out_state_crop else None,
        "state_totals_file": str(out_state_total) if out_state_total else None,
        "rows_in": len(df),
    }

def etl_rain():
    rain_csv = RAW / "raw_rainfall_data.csv"
    rain_xlsx = RAW / "raw_rainfall_data.xlsx"
    df = read_any(rain_csv, rain_xlsx)
    # normalize column names
    df.columns = [c.strip().upper() for c in df.columns]
    df.columns = [c.replace(" ", "_") for c in df.columns]

    # Detect subdivision/state-wise file vs India-only
    has_sub = "SUBDIVISION" in df.columns

    # Standardize → lower
    df.columns = [c.lower() for c in df.columns]
    rename_map = {
        "year": "year",
        "subdivision": "subdivision",
        "parameter": "parameter",
        "annual": "annual_mm",
        "jf": "jan_feb_mm",
        "mam": "mar_may_mm",
        "jjas": "jun_sep_mm",
        "ond": "oct_dec_mm",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Coerce numerics
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for mm_col in ["annual_mm","jan_feb_mm","mar_may_mm","jun_sep_mm","oct_dec_mm"]:
        if mm_col in df.columns:
            df[mm_col] = pd.to_numeric(df[mm_col], errors="coerce")

    # Keep only Actual rainfall
    if "parameter" in df.columns:
        df["parameter"] = df["parameter"].astype(str).str.strip().str.title()
        df = df[df["parameter"] == "Actual"].copy()

    # If no subdivision present → India-only file (keep compatibility)
    if "subdivision" not in df.columns:
        keep = [c for c in ["year","annual_mm","jan_feb_mm","mar_may_mm","jun_sep_mm","oct_dec_mm"] if c in df.columns]
        india = df[keep].dropna(subset=["year"]).sort_values("year")
        out_india = CLEAN / "rain_india_year.parquet"
        india.to_parquet(out_india, index=False)
        return {"india_file": str(out_india), "subdivision_file": None, "rows_in": len(india)}

    # Clean subdivision text
    df["subdivision"] = df["subdivision"].astype(str).str.strip().str.title()

    # Save subdivision-level
    keep_sub = [c for c in ["year","subdivision","annual_mm","jan_feb_mm","mar_may_mm","jun_sep_mm","oct_dec_mm"] if c in df.columns]
    sub = df[keep_sub].dropna(subset=["year","subdivision"]).drop_duplicates().sort_values(["subdivision","year"])
    out_sub = CLEAN / "rain_subdivision_year.parquet"
    sub.to_parquet(out_sub, index=False)

    # ------- NEW: Aggregate to state-level via alias map -------
    # Note: IMD subdivisions don't always match states 1:1. We create a practical mapping.
    # For states spanning multiple subdivisions we take a simple mean of subdivision annual_mm.
    # (Proper area-weighted averaging would need subdivision area weights; if you have them, we can plug them in.)
    sub_to_state = {
        # Karnataka splits into three:
        "Coastal Karnataka": "Karnataka",
        "North Interior Karnataka": "Karnataka",
        "South Interior Karnataka": "Karnataka",
        # Tamil Nadu is 1:1:
        "Tamil Nadu": "Tamil Nadu",
        # Add a few common ones (you can extend this list as needed)
        "Kerala": "Kerala",
        "Coastal Andhra Pradesh": "Andhra Pradesh",
        "Rayalaseema": "Andhra Pradesh",
        "Telangana": "Telangana",
        "Madhya Maharashtra": "Maharashtra",
        "Marathwada": "Maharashtra",
        "Vidarbha": "Maharashtra",
        "Konkan & Goa": "Maharashtra/Goa",  # spans two; we keep as combined unless you want to split
        "Gujarat Region": "Gujarat",
        "Saurashtra & Kutch": "Gujarat",
        "East Uttar Pradesh": "Uttar Pradesh",
        "West Uttar Pradesh": "Uttar Pradesh",
        "Haryana Delhi & Chandigarh": "Haryana/Delhi/Chandigarh",
        "Punjab": "Punjab",
        "Himachal Pradesh": "Himachal Pradesh",
        "Jammu & Kashmir": "Jammu & Kashmir",
        "Arunachal Pradesh": "Arunachal Pradesh",
        "Assam & Meghalaya": "Assam/Meghalaya",
        "Naga Mani Mizo Tripura": "NE-4 States",
        "Sub Himalayan West Bengal & Sikkim": "West Bengal/Sikkim",
        "Gangetic West Bengal": "West Bengal",
        "Bihar": "Bihar",
        "Jharkhand": "Jharkhand",
        "Odisha": "Odisha",
        "East Madhya Pradesh": "Madhya Pradesh",
        "West Madhya Pradesh": "Madhya Pradesh",
        "East Rajasthan": "Rajasthan",
        "West Rajasthan": "Rajasthan",
        "Andaman & Nicobar Islands": "Andaman And Nicobar Islands",
        "Lakshadweep": "Lakshadweep",
    }

    sub["state"] = sub["subdivision"].map(sub_to_state)
    stateable = sub.dropna(subset=["state", "annual_mm"])

    # Aggregate: simple mean across mapped subdivisions for each state-year
    state = (
        stateable
        .groupby(["state","year"], as_index=False)
        .agg(
            annual_mm=("annual_mm", "mean"),
            jan_feb_mm=("jan_feb_mm", "mean"),
            mar_may_mm=("mar_may_mm", "mean"),
            jun_sep_mm=("jun_sep_mm", "mean"),
            oct_dec_mm=("oct_dec_mm", "mean"),
        )
        .sort_values(["state","year"])
    )
    out_state = CLEAN / "rain_state_year.parquet"
    state.to_parquet(out_state, index=False)

    return {
        "india_file": None,
        "subdivision_file": str(out_sub),
        "state_file": str(out_state),
        "rows_in": len(sub)
    }


def main():
    print("ETL: Crops ...")
    crop_out = etl_crops()
    print("Crops cleaned:", crop_out)

    print("ETL: Rainfall ...")
    rain_out = etl_rain()
    print("Rainfall cleaned:", rain_out)

    print("All done. Clean files are in:", CLEAN)

if __name__ == "__main__":
    main()
