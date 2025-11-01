# scripts/duckdb_load_and_test.py
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "cleaned_data"
DB    = ROOT / "samarth.duckdb"

def sql_path(p: Path) -> str:
    """Return a SQL-quoted path literal, escaping backslashes for Windows."""
    s = str(p)
    s = s.replace("\\", "\\\\")  # escape backslashes
    return f"'{s}'"

con = duckdb.connect(str(DB))

print("Registering Parquet datasets as views ...")

con.execute(f"""
CREATE OR REPLACE VIEW crop_district_year AS
SELECT * FROM read_parquet({sql_path(CLEAN / "crop_district_year.parquet")})
""")

con.execute(f"""
CREATE OR REPLACE VIEW crop_state_year AS
SELECT * FROM read_parquet({sql_path(CLEAN / "crop_state_year.parquet")})
""")

con.execute(f"""
CREATE OR REPLACE VIEW crop_state_totals AS
SELECT * FROM read_parquet({sql_path(CLEAN / "crop_state_totals.parquet")})
""")

con.execute(f"""
CREATE OR REPLACE VIEW rain_india_year AS
SELECT * FROM read_parquet({sql_path(CLEAN / "rain_india_year.parquet")})
""")

# Optional subdivision/state rainfall
rain_sub_file = CLEAN / "rain_subdivision_year.parquet"
have_sub = rain_sub_file.exists()
if have_sub:
    con.execute(f"""
    CREATE OR REPLACE VIEW rain_subdivision_year AS
    SELECT * FROM read_parquet({sql_path(rain_sub_file)})
    """)

def one(sql, params=None):
    return con.execute(sql, params or {}).fetchone()

def df(sql, params=None):
    return con.execute(sql, params or {}).fetchdf()

print("\n=== Row counts ===")
print("crop_state_year:", one("SELECT COUNT(*) FROM crop_state_year")[0])
print("crop_state_totals:", one("SELECT COUNT(*) FROM crop_state_totals")[0])
print("rain_india_year:", one("SELECT COUNT(*) FROM rain_india_year")[0])
print("rain_subdivision_year:" if have_sub else "rain_subdivision_year: (not available)", 
      one("SELECT COUNT(*) FROM rain_subdivision_year")[0] if have_sub else "")

print("\n=== Sample rows (crop_state_year) ===")
print(df("SELECT * FROM crop_state_year LIMIT 5"))

print("\n=== Sample rows (rain_india_year) ===")
print(df("SELECT * FROM rain_india_year LIMIT 5"))

print("\n=== Q1: Top 5 crops by total production per state (latest year per state) ===")
q1 = """
WITH latest AS (
  SELECT state, MAX(year) AS latest_year
  FROM crop_state_year
  WHERE state IS NOT NULL AND year IS NOT NULL
  GROUP BY 1
),
ranked AS (
  SELECT c.state, c.year, c.crop, c.production_tonnes,
         ROW_NUMBER() OVER (PARTITION BY c.state, c.year ORDER BY c.production_tonnes DESC NULLS LAST) AS rnk
  FROM crop_state_year c
  JOIN latest l ON c.state = l.state AND c.year = l.latest_year
)
SELECT state, year, crop, production_tonnes
FROM ranked
WHERE rnk <= 5
ORDER BY state, production_tonnes DESC;
"""
print(df(q1))

print("\n=== Q2: State totals (production & yield) for a chosen year (example: 2018) ===")
yr = 2018
q2 = """
SELECT state, year, 
       SUM(area_ha) AS area_ha, 
       SUM(production_tonnes) AS production_tonnes,
       CASE WHEN SUM(area_ha) > 0 THEN (SUM(production_tonnes)*1000.0)/SUM(area_ha) ELSE NULL END AS yield_kg_per_ha
FROM crop_state_year
WHERE year = ?
GROUP BY 1,2
ORDER BY production_tonnes DESC NULLS LAST;
"""
print(df(q2, [yr]))

if have_sub:
    print("\n=== Q3: Avg annual rainfall by subdivision for last N years (example N=5) ===")
    ymax = one("SELECT MAX(year) FROM rain_subdivision_year")[0]
    if ymax is not None:
        N = 5
        q3 = """
        SELECT subdivision, AVG(annual_mm) AS avg_annual_mm, MIN(year) AS from_year, MAX(year) AS to_year
        FROM rain_subdivision_year
        WHERE year BETWEEN ?-?+1 AND ?
        GROUP BY 1
        ORDER BY avg_annual_mm DESC NULLS LAST
        LIMIT 20;
        """
        print(df(q3, [ymax, N, ymax]))
    else:
        print("No years in rain_subdivision_year.")
else:
    print("\n(Q3 skipped) Subdivision rainfall not loaded yet (India-level file in use).")

print("\nAll tests done. DB file at:", DB)
