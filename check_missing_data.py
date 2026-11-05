import pandas as pd
import numpy as np

# 1. Load your local data.parquet
df_parquet = pd.read_parquet("./data/data.parquet")
print("=== 1. LOCAL DATA.PARQUET SUMMARY ===")
solo_df = df_parquet[df_parquet['is_collab'] == False]
print(f"Total rows in data.parquet: {len(df_parquet):,}")
print(f"Unique solo countries (by 'country' name): {solo_df['country'].nunique()}")
print(f"Unique solo 'iso2c' codes (non-null):      {solo_df['iso2c'].dropna().nunique()}")
if 'iso3c' in solo_df.columns:
    print(f"Unique solo 'iso3c' codes (non-null):      {solo_df['iso3c'].dropna().nunique()}")
    missing_iso3 = solo_df[solo_df['iso3c'].isna() | (solo_df['iso3c'] == '')][['country', 'iso2c']].drop_duplicates()
    if not missing_iso3.empty:
        print("\n⚠️ COUNTRIES PRESENT IN PARQUET BUT MISSING 'iso3c' (Invisible on Choropleth!):")
        print(missing_iso3.to_string(index=False))

# Check if coordinates (lat/lng) are missing for any country (which drops them in load_country_list!)
missing_coords = solo_df[solo_df['lat'].isna() | solo_df['lng'].isna()][['country', 'iso2c', 'iso3c']].drop_duplicates()
if not missing_coords.empty:
    print("\n⚠️ COUNTRIES WITH NaN LAT/LNG (Dropped by drop_duplicates/filtering!):")
    print(missing_coords.to_string(index=False))

# 2. Compare against the Original GitHub Raw TSVs
BASE_RAW_URL = "https://raw.githubusercontent.com/mbermudezmoTec/China-s-great-leap-forward-in-the-chemical-space/main/data/raw"

for filename in ["Absolute_collab_allSubs.tsv", "Country_weighted_allSubs.tsv"]:
    url = f"{BASE_RAW_URL}/{filename}"
    print(f"\n=== 2. COMPARING WITH GITHUB RAW FILE: {filename} ===")
    try:
        # IMPORTANT: keep_default_na=False prevents Namibia ('NA') from turning into NaN!
        raw_df = pd.read_csv(url, sep="\t", index_col=0, keep_default_na=False)
        raw_entities = [str(idx).strip() for idx in raw_df.index if str(idx).strip() != ""]

        # Single countries are rows without a comma (',') or hyphen ('-')
        raw_single_isos = set([
            code for code in raw_entities
            if ("," not in code) and ("-" not in code) and len(code) == 2
        ])
        parquet_isos = set(solo_df['iso2c'].dropna().str.strip().str.upper().unique())

        missing_in_parquet = sorted(list(raw_single_isos - parquet_isos))
        extra_in_parquet = sorted(list(parquet_isos - raw_single_isos))

        print(f"Single-country ISO2 codes in {filename}: {len(raw_single_isos)}")
        print(f"Single-country ISO2 codes in data.parquet: {len(parquet_isos)}")
        print(f"❌ Missing in data.parquet ({len(missing_in_parquet)}): {missing_in_parquet}")
        if extra_in_parquet:
            print(f"➕ Extra in data.parquet ({len(extra_in_parquet)}): {extra_in_parquet}")

        # Check how many countries in data.parquet have 0.0% across all years
        zero_countries = (
            solo_df[solo_df['chemical'] == 'All']
            .groupby(['iso2c', 'country'])['percentage']
            .max()
        )
        all_zero = zero_countries[zero_countries == 0]
        print(f"Countries in data.parquet with 0.0% max contribution: {len(all_zero)}")

    except Exception as e:
        print(f"Could not fetch {filename}: {e}")