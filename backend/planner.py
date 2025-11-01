# backend/planner.py
from __future__ import annotations
import re, json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import duckdb
import pandas as pd
import numpy as np
# Canonical crop aliases (expand anytime)
CROP_ALIAS = {
    "Rice": ["Rice", "Paddy"],
    "Cotton": ["Cotton", "Cotton(Lint)", "Cotton Lint"],
    "Rapeseed And Mustard": ["Rapeseed And Mustard", "Rapeseed & Mustard"],
}

def _crop_variants(crop: str) -> list[str]:
    key = crop.strip().title()
    return CROP_ALIAS.get(key, [key])

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "cleaned_data"
DB    = ROOT / "samarth.duckdb"
MANIFEST = ROOT / "manifest" / "datasets.json"

# ---------- helpers ----------

def _sql_path(p: Path) -> str:
    s = str(p).replace("\\", "\\\\")
    return f"'{s}'"

def _register_views(con: duckdb.DuckDBPyConnection) -> Dict[str, bool]:
    con.execute(f"CREATE OR REPLACE VIEW crop_state_year AS SELECT * FROM read_parquet({_sql_path(CLEAN/'crop_state_year.parquet')})")
    con.execute(f"CREATE OR REPLACE VIEW crop_state_totals AS SELECT * FROM read_parquet({_sql_path(CLEAN/'crop_state_totals.parquet')})")
    con.execute(f"CREATE OR REPLACE VIEW crop_district_year AS SELECT * FROM read_parquet({_sql_path(CLEAN/'crop_district_year.parquet')})")
    if (CLEAN / 'rain_india_year.parquet').exists():
        con.execute(f"CREATE OR REPLACE VIEW rain_india_year AS SELECT * FROM read_parquet({_sql_path(CLEAN/'rain_india_year.parquet')})")
    have_sub = (CLEAN / "rain_subdivision_year.parquet").exists()
    if have_sub:
        con.execute(f"CREATE OR REPLACE VIEW rain_subdivision_year AS SELECT * FROM read_parquet({_sql_path(CLEAN/'rain_subdivision_year.parquet')})")
    have_state = (CLEAN / "rain_state_year.parquet").exists()
    if have_state:
        con.execute(f"CREATE OR REPLACE VIEW rain_state_year AS SELECT * FROM read_parquet({_sql_path(CLEAN/'rain_state_year.parquet')})")
    return {"have_subdivision": have_sub, "have_state": have_state}

def _load_manifest() -> Dict[str, Dict[str, Any]]:
    if MANIFEST.exists():
        arr = json.loads(MANIFEST.read_text(encoding="utf-8"))
        return {e.get("id",""): e for e in arr if isinstance(e, dict)}
    return {}

def _latest_year(con, table: str, where: str = "1=1") -> Optional[int]:
    row = con.execute(f"SELECT MAX(year) FROM {table} WHERE {where}").fetchone()
    return int(row[0]) if row and row[0] is not None else None

def _years_range(con, table: str) -> Tuple[Optional[int], Optional[int]]:
    row = con.execute(f"SELECT MIN(year), MAX(year) FROM {table}").fetchone()
    if not row: return (None, None)
    return (int(row[0]) if row[0] is not None else None,
            int(row[1]) if row[1] is not None else None)

def _norm_state(s: str) -> str:
    return s.strip().title()

# ---------- very small NLU (states whitelist) ----------
PAT_STATE = r"(andaman and nicobar islands|andhra pradesh|arunachal pradesh|assam|bihar|chandigarh|chhattisgarh|dadra and nagar haveli and daman and diu|delhi|goa|gujarat|haryana|himachal pradesh|jammu and kashmir|jharkhand|karnataka|kerala|ladakh|lakshadweep|madhya pradesh|maharashtra|manipur|meghalaya|mizoram|nagaland|odisha|puducherry|punjab|rajasthan|sikkim|tamil nadu|telangana|tripura|uttar pradesh|uttarakhand|west bengal)"
PAT_CROP  = r"([a-z][a-z \(\)\-\/]+?)"  # loose crop capture; we'll .title() later

# ---------- parsing ----------
def parse_question(q: str) -> Dict[str, Any]:
    ql = q.lower().strip()

    # --- Template B1: district extremes in ONE state, state's name BEFORE crop ---
    m = re.search(
        rf"district\s+in\s+(?P<state1>{PAT_STATE}).*?(?P<mode>highest|lowest|max|min).*?production\s+of\s+(?P<crop>.+?)\s+(?:in|for)?\s+(?P<year>\d{{4}})",
        ql
    )
    if m:
        mode = m.group("mode")
        if mode == "max": mode = "highest"
        if mode == "min": mode = "lowest"
        return {
            "type": "district_extremes_one",
            "crop": m.group("crop").strip().title(),
            "state1": _norm_state(m.group("state1")),
            "year": int(m.group("year")),
            "mode": mode
        }

    # --- Template B2: district extremes in ONE state, crop BEFORE state ---
    m = re.search(
        rf"district.*?(?P<mode>highest|lowest|max|min).*?production\s+of\s+(?P<crop>.+?)\s+in\s+(?P<state1>{PAT_STATE})\s+(?:in|for)?\s+(?P<year>\d{{4}})",
        ql
    )
    if m:
        mode = m.group("mode")
        if mode == "max": mode = "highest"
        if mode == "min": mode = "lowest"
        return {
            "type": "district_extremes_one",
            "crop": m.group("crop").strip().title(),
            "state1": _norm_state(m.group("state1")),
            "year": int(m.group("year")),
            "mode": mode
        }

    # --- Template B3: TWO states (highest in X vs lowest in Y), optional year ---
    m = re.search(
        rf"district\s+in\s+(?P<state1>{PAT_STATE}).*?(?P<mode1>highest|max).*?production\s+of\s+(?P<crop>.+?)"
        rf"(?:\s+(?:in|for)\s+(?P<year>\d{{4}}))?.*?district\s+(?:in|of)\s+(?P<state2>{PAT_STATE}).*?(?P<mode2>lowest|min).*?production",
        ql
    )
    if m:
        year = m.group("year")
        return {
            "type": "district_extremes_two",
            "crop": m.group("crop").strip().title(),
            "state1": _norm_state(m.group("state1")),
            "state2": _norm_state(m.group("state2")),
            "year": int(year) if year else None
        }

    # --- Template A: rainfall compare + top crops ---
    m = re.search(
        rf"compare.*?rainfall.*?(?P<state1>{PAT_STATE}).*?(?P<state2>{PAT_STATE}).*?last\s+(?P<N>\d+)",
        ql
    )
    if m:
        mm = re.search(r"top\s+(?P<M>\d+)", ql)
        M = int(mm.group("M")) if mm else 3
        return {
            "type":"rainfall_compare_topcrops",
            "state1": _norm_state(m.group("state1")),
            "state2": _norm_state(m.group("state2")),
            "N": int(m.group("N")), "M": M
        }

    # --- Template C: India trend + correlation ---
    m = re.search(r"(trend|correlate|correlation).*?crop\s+(?P<crop>"+PAT_CROP+r")\s+.*?(india|national)", ql)
    if m:
        return {"type":"india_trend_corr", "crop": m.group("crop").strip().title(), "years": 10}

    # --- Simple: Top M crops in a state (latest year) ---
    m = re.search(r"top\s+(?P<M>\d+)\s+crops.*?(?P<state>"+PAT_STATE+r")", ql)
    if m:
        return {"type":"top_crops_state_latest", "state": _norm_state(m.group("state")), "M": int(m.group("M"))}

    return {
        "type": "unknown",
        "hint": ("Try:\n"
                 "- 'Which district in Maharashtra has the highest production of Sugarcane in 2014?'\n"
                 "- 'Top 5 crops in Karnataka'\n"
                 "- 'Analyze trend/correlation for crop Rice in India'\n"
                 "- 'Compare rainfall in Karnataka and Tamil Nadu for last 5 years and top 3 crops' (requires subdivision rainfall)")
    }

# ---------- planners ----------

def answer_district_extremes_one(con, crop: str, state1: str, year: int, mode: str, manifest) -> Dict[str, Any]:
    order = "DESC NULLS LAST" if mode == "highest" else "ASC NULLS FIRST"
    variants = _crop_variants(crop)

    def extreme_for_year(y):
        placeholders = ",".join(["?"] * len(variants))
        q = f"""
        SELECT district, SUM(production_tonnes) AS production_tonnes
        FROM crop_district_year
        WHERE state = ? AND crop IN ({placeholders}) AND year = ?
        GROUP BY 1
        ORDER BY production_tonnes {order}
        LIMIT 1;
        """
        return con.execute(q, [state1, *variants, int(y)]).fetchone()

    row = extreme_for_year(year)
    used_year = year

    if not row:
        y_alt = con.execute(
            "SELECT MAX(year) FROM crop_district_year WHERE state = ? AND crop IN ({})".format(",".join(["?"]*len(variants))),
            [state1, *variants]
        ).fetchone()[0]
        if y_alt is not None:
            row = extreme_for_year(y_alt)
            used_year = int(y_alt)

    if not row:
        return {"answer_text": f"No rows found for {crop} in {state1}. Try another crop or state.",
                "tables": [], "citations": _cite(manifest)}

    title = f"{mode.title()} production district"
    note = "" if used_year == year else f" (requested {year} not available; using {used_year})"
    return {
        "answer_text": f"{title} for {crop} in {state1}{note}",
        "tables": [{"title": f"{title} ({used_year})",
                    "rows": [{"district": row[0], "production_tonnes": float(row[1])}]}],
        "citations": _cite(manifest)
    }


def answer_district_extremes_two(con, crop: str, state1: str, state2: str, year: Optional[int], manifest) -> Dict[str, Any]:
    # If year not given, use each state's latest available year for that crop
    if year is None:
        y1 = con.execute("SELECT MAX(year) FROM crop_district_year WHERE state=? AND crop=?", [state1, crop]).fetchone()[0]
        y2 = con.execute("SELECT MAX(year) FROM crop_district_year WHERE state=? AND crop=?", [state2, crop]).fetchone()[0]
    else:
        y1 = y2 = year

    def extreme(state, yr, order):
        if yr is None:
            return None
        q = f"""
        SELECT district, SUM(production_tonnes) AS production_tonnes
        FROM crop_district_year
        WHERE state = ? AND crop = ? AND year = ?
        GROUP BY 1
        ORDER BY production_tonnes {order}
        LIMIT 1;
        """
        return con.execute(q, [state, crop, int(yr)]).fetchone()

    max_row = extreme(state1, y1, "DESC NULLS LAST")
    min_row = extreme(state2, y2, "ASC NULLS FIRST")

    if not max_row and not min_row:
        return {"answer_text": f"No rows found for {crop} in {state1} or {state2}.",
                "tables": [], "citations": _cite(manifest)}

    tables = []
    if max_row:
        tables.append({"title": f"Highest production in {state1} ({y1})",
                       "rows": [{"district": max_row[0], "production_tonnes": float(max_row[1])}]})
    if min_row:
        tables.append({"title": f"Lowest production in {state2} ({y2})",
                       "rows": [{"district": min_row[0], "production_tonnes": float(min_row[1])}]})

    return {
        "answer_text": f"Comparison for {crop}: highest in {state1} vs lowest in {state2}.",
        "tables": tables,
        "citations": _cite(manifest)
    }

def answer_rainfall_compare_topcrops(con, state1: str, state2: str, N: int, M: int, flags, manifest) -> Dict[str, Any]:
    """
    Compare avg annual rainfall in two states over last N available years,
    and list top M crops by total production in that same window.
    Prefers rain_state_year (aggregated in ETL); falls back to subdivision-level, and
    if a state is missing, computes on-the-fly as the mean of its subdivisions.
    """
    # ---- 1) Pick rainfall source and compute window ----
    use_state = flags.get("have_state")
    if use_state:
        ymax = con.execute("SELECT MAX(year) FROM rain_state_year").fetchone()[0]
        if ymax is None:
            use_state = False
    if not use_state and not flags.get("have_subdivision"):
        return {"answer_text": "Rainfall file is India-level only. Load subdivision/state-wise IMD and re-run ETL.",
                "tables": [], "citations": _cite(manifest)}

    if use_state:
        y1 = ymax - N + 1
        rain_df = con.execute("""
            SELECT state, AVG(annual_mm) AS avg_annual_mm
            FROM rain_state_year
            WHERE state IN (?, ?) AND year BETWEEN ? AND ?
            GROUP BY 1
        """, [state1, state2, y1, ymax]).fetchdf()
    else:
        ymax = con.execute("SELECT MAX(year) FROM rain_subdivision_year").fetchone()[0]
        if ymax is None:
            return {"answer_text": "No rainfall years available.", "tables": [], "citations": _cite(manifest)}
        y1 = ymax - N + 1
        rain_df = con.execute("""
            SELECT subdivision AS state, AVG(annual_mm) AS avg_annual_mm
            FROM rain_subdivision_year
            WHERE subdivision IN (?, ?) AND year BETWEEN ? AND ?
            GROUP BY 1
        """, [state1, state2, y1, ymax]).fetchdf()

    # ---- 1b) Fallback if a state missing but subdivision data exists ----
    if flags.get("have_subdivision"):
        needed = {state1, state2} - set(rain_df["state"].tolist())
        if needed:
            sub_groups = {
                "Karnataka": ["Coastal Karnataka","North Interior Karnataka","South Interior Karnataka"],
                "Andhra Pradesh": ["Coastal Andhra Pradesh","Rayalaseema"],
                "Gujarat": ["Gujarat Region","Saurashtra & Kutch"],
                "Uttar Pradesh": ["East Uttar Pradesh","West Uttar Pradesh"],
            }
            for miss in list(needed):
                subs = sub_groups.get(miss)
                if subs:
                    row = con.execute(f"""
                        SELECT '{miss}' AS state, AVG(annual_mm) AS avg_annual_mm
                        FROM rain_subdivision_year
                        WHERE subdivision IN ({','.join(['?']*len(subs))})
                          AND year BETWEEN ? AND ?
                    """, subs + [y1, ymax]).fetchdf()
                    if not row.empty:
                        rain_df = pd.concat([rain_df, row], ignore_index=True)

    # ---- 2) Top M crops in same window; fallback if empty ----
    def top_crops_window(y_from, y_to):
        return con.execute("""
            SELECT *
            FROM (
                SELECT
                    state,
                    crop,
                    SUM(production_tonnes) AS total_tonnes,
                    ROW_NUMBER() OVER (
                        PARTITION BY state
                        ORDER BY SUM(production_tonnes) DESC NULLS LAST
                    ) AS rnk
                FROM crop_state_year
                WHERE state IN (?, ?)
                  AND year BETWEEN ? AND ?
                GROUP BY state, crop
            ) t
            WHERE rnk <= ?
            ORDER BY state, total_tonnes DESC NULLS LAST
        """, [state1, state2, y_from, y_to, M]).fetchdf()

    crops_df = top_crops_window(y1, ymax)

    if crops_df.empty:
        # fallback: use latest common crop window present
        y_max_crop = con.execute("""
            SELECT MIN(maxy) FROM (
                SELECT state, MAX(year) AS maxy
                FROM crop_state_year
                WHERE state IN (?, ?)
                GROUP BY state
            )
        """, [state1, state2]).fetchone()[0]
        y_min_crop = con.execute("""
            SELECT MAX(miny) FROM (
                SELECT state, MIN(year) AS miny
                FROM crop_state_year
                WHERE state IN (?, ?)
                GROUP BY state
            )
        """, [state1, state2]).fetchone()[0]
        if y_max_crop and y_min_crop:
            y1c = max(y_min_crop, y_max_crop - (N - 1))
            crops_df = top_crops_window(y1c, y_max_crop)
            crop_window_note = f"{y1c}-{y_max_crop}"
        else:
            crop_window_note = None
    else:
        crop_window_note = f"{y1}-{ymax}"

    # ---- 3) Build response ----
    tables = []
    tables.append({
        "title": f"Avg annual rainfall {y1}-{ymax}",
        "rows": rain_df.round(2).to_dict(orient="records")
    })
    if not crops_df.empty:
        title = f"Top {M} crops by total production {crop_window_note}" if crop_window_note else f"Top {M} crops (latest available)"
        tables.append({
            "title": title,
            "rows": crops_df.to_dict(orient="records")
        })

    return {
        "answer_text": f"Comparison for {state1} vs {state2} over {y1}-{ymax}.",
        "tables": tables,
        "citations": _cite(manifest)
    }

def answer_india_trend_corr(con, crop: str, years: int, manifest) -> Dict[str, Any]:
    ymax = _latest_year(con, "crop_state_year")
    if ymax is None:
        return {"answer_text": "No crop years available.", "tables": [], "citations": _cite(manifest)}
    y1 = ymax - years + 1

    qc = """
    SELECT year, SUM(production_tonnes) AS production_tonnes
    FROM crop_state_year
    WHERE crop = ? AND year BETWEEN ? AND ?
    GROUP BY 1 ORDER BY 1;
    """
    prod = con.execute(qc, [crop, y1, ymax]).fetchdf()

    qr = """
    SELECT year, annual_mm
    FROM rain_india_year
    WHERE year BETWEEN ? AND ?
    ORDER BY year;
    """
    rain = con.execute(qr, [y1, ymax]).fetchdf()

    tables = []
    if not prod.empty:
        tables.append({"title": f"Production of {crop} in India ({y1}-{ymax})",
                       "rows": prod.to_dict(orient="records")})
    if not rain.empty:
        tables.append({"title": f"Annual rainfall (India) ({y1}-{ymax})",
                       "rows": rain.to_dict(orient="records")})

    merged = pd.merge(prod, rain, on="year", how="inner")
    corr_txt = "Correlation not computed (insufficient overlap)."
    if len(merged) >= 3 and merged["production_tonnes"].notna().sum() >= 3 and merged["annual_mm"].notna().sum() >= 3:
        r = merged["production_tonnes"].corr(merged["annual_mm"])
        corr_txt = f"Pearson correlation (production vs rainfall) over {y1}-{ymax}: r = {r:.2f}"

    return {"answer_text": f"India trend for {crop}. {corr_txt}",
            "tables": tables, "citations": _cite(manifest)}

def answer_top_crops_state_latest(con, state: str, M: int, manifest) -> Dict[str, Any]:
    # Try latest year globally, then latest for that state
    ymax_global = _latest_year(con, "crop_state_year", "state IS NOT NULL")
    yr_try = ymax_global if ymax_global is not None else _latest_year(con, "crop_state_year")
    q = """
    SELECT crop, production_tonnes
    FROM crop_state_year
    WHERE state = ? AND year = ?
    ORDER BY production_tonnes DESC NULLS LAST
    LIMIT ?;
    """
    df = pd.DataFrame()
    if yr_try is not None:
        df = con.execute(q, [state, yr_try, M]).fetchdf()

    if df.empty:
        yr_state = con.execute("SELECT MAX(year) FROM crop_state_year WHERE state = ?", [state]).fetchone()[0]
        if yr_state is None:
            return {"answer_text": f"No data found for {state}.", "tables": [], "citations": _cite(manifest)}
        df = con.execute(q, [state, yr_state, M]).fetchdf()
        header = f"Top {M} crops in {state} (latest available year for state: {yr_state})"
    else:
        header = f"Top {M} crops in {state} (year {yr_try})"

    return {"answer_text": header, "tables": [{"title": header, "rows": df.to_dict(orient="records")}], "citations": _cite(manifest)}

def _cite(manifest: Dict[str, Any]) -> List[Dict[str, str]]:
    cites = []
    for k in ("crop_district", "imd_rainfall"):
        if k in manifest:
            e = manifest[k]
            cites.append({
                "dataset_id": k,
                "title": e.get("title",""),
                "catalog_url": e.get("catalog_url") or e.get("url",""),
                "resource_id": e.get("resource_id","")
            })
    return cites

# ---------- main entry ----------
def answer_question(q: str) -> Dict[str, Any]:
    manifest = _load_manifest()
    con = duckdb.connect(str(DB))
    flags = _register_views(con)
    parsed = parse_question(q)

    t = parsed.get("type")
    if t == "district_extremes_one":
        return answer_district_extremes_one(con, parsed["crop"], parsed["state1"], parsed["year"], parsed["mode"], manifest)
    elif t == "district_extremes_two":
        return answer_district_extremes_two(con, parsed["crop"], parsed["state1"], parsed["state2"], parsed["year"], manifest)
    elif t == "rainfall_compare_topcrops":
        return answer_rainfall_compare_topcrops(con, parsed["state1"], parsed["state2"], parsed["N"], parsed["M"], flags, manifest)
    elif t == "india_trend_corr":
        return answer_india_trend_corr(con, parsed["crop"], parsed["years"], manifest)
    elif t == "top_crops_state_latest":
        return answer_top_crops_state_latest(con, parsed["state"], parsed["M"], manifest)
    else:
        return {"answer_text": parsed.get("hint") or "Sorry, I couldn't understand the question.",
                "tables": [], "citations": _cite(manifest)}
