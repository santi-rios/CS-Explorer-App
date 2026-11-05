import shutil
import pandas as pd
import numpy as np
import requests
import io

PARQUET_PATH = "./data/data.parquet"
BACKUP_PATH = "./data/data_backup.parquet"

# 1. Create a safe backup first
shutil.copyfile(PARQUET_PATH, BACKUP_PATH)
print(f"✅ Backed up original parquet to {BACKUP_PATH}")

df = pd.read_parquet(PARQUET_PATH)

# 2. Metadata dictionary for the 50 missing ISO2 codes:
# (iso2c: (Country Name, iso3c, Region, lat, lng))
MISSING_META = {
    "NA": ("Namibia", "NAM", "Sub-Saharan Africa", -22.9576, 18.4904),
    "AD": ("Andorra", "AND", "Europe & Central Asia", 42.5462, 1.6016),
    "AF": ("Afghanistan", "AFG", "South Asia", 33.9391, 67.7100),
    "AG": ("Antigua and Barbuda", "ATG", "Latin America & Caribbean", 17.0608, -61.7964),
    "AL": ("Albania", "ALB", "Europe & Central Asia", 41.1533, 20.1683),
    "AO": ("Angola", "AGO", "Sub-Saharan Africa", -11.2027, 17.8739),
    "BI": ("Burundi", "BDI", "Sub-Saharan Africa", -3.3731, 29.9189),
    "BJ": ("Benin", "BEN", "Sub-Saharan Africa", 9.3077, 2.3158),
    "BS": ("Bahamas", "BHS", "Latin America & Caribbean", 25.0343, -77.3963),
    "BT": ("Bhutan", "BTN", "South Asia", 27.5142, 90.4336),
    "BZ": ("Belize", "BLZ", "Latin America & Caribbean", 17.1899, -88.4976),
    "CF": ("Central African Republic", "CAF", "Sub-Saharan Africa", 6.6111, 20.9394),
    "CG": ("Republic of the Congo", "COG", "Sub-Saharan Africa", -0.2280, 15.8277),
    "DJ": ("Djibouti", "DJI", "Middle East & North Africa", 11.8251, 42.5903),
    "DM": ("Dominica", "DMA", "Latin America & Caribbean", 15.4150, -61.3710),
    "ER": ("Eritrea", "ERI", "Sub-Saharan Africa", 15.1794, 39.7823),
    "FO": ("Faroe Islands", "FRO", "Europe & Central Asia", 61.8926, -6.9118),
    "GA": ("Gabon", "GAB", "Sub-Saharan Africa", -0.8037, 11.6094),
    "GD": ("Grenada", "GRD", "Latin America & Caribbean", 12.2628, -61.6042),
    "GF": ("French Guiana", "GUF", "Latin America & Caribbean", 3.9339, -53.1258),
    "GL": ("Greenland", "GRL", "Europe & Central Asia", 71.7069, -42.6043),
    "GM": ("Gambia", "GMB", "Sub-Saharan Africa", 13.4432, -15.3101),
    "GP": ("Guadeloupe", "GLP", "Latin America & Caribbean", 16.9950, -62.0676),
    "GU": ("Guam", "GUM", "East Asia & Pacific", 13.4443, 144.7937),
    "GY": ("Guyana", "GUY", "Latin America & Caribbean", 4.8604, -58.9302),
    "JE": ("Jersey", "JEY", "Europe & Central Asia", 49.2144, -2.1312),
    "KH": ("Cambodia", "KHM", "East Asia & Pacific", 12.5657, 104.9910),
    "KP": ("North Korea", "PRK", "East Asia & Pacific", 40.3399, 127.5101),
    "LA": ("Laos", "LAO", "East Asia & Pacific", 19.8563, 102.4955),
    "LC": ("Saint Lucia", "LCA", "Latin America & Caribbean", 13.9094, -60.9789),
    "MC": ("Monaco", "MCO", "Europe & Central Asia", 43.7503, 7.4128),
    "MM": ("Myanmar", "MMR", "East Asia & Pacific", 21.9140, 95.9562),
    "MO": ("Macau", "MAC", "East Asia & Pacific", 22.1987, 113.5439),
    "MQ": ("Martinique", "MTQ", "Latin America & Caribbean", 14.6415, -61.0242),
    "MR": ("Mauritania", "MRT", "Sub-Saharan Africa", 21.0079, -10.9408),
    "MW": ("Malawi", "MWI", "Sub-Saharan Africa", -13.2543, 34.3015),
    "NF": ("Norfolk Island", "NFK", "East Asia & Pacific", -29.0408, 167.9547),
    "NI": ("Nicaragua", "NIC", "Latin America & Caribbean", 12.8654, -85.2072),
    "PW": ("Palau", "PLW", "East Asia & Pacific", 7.5150, 134.5825),
    "RE": ("Reunion", "REU", "Sub-Saharan Africa", -21.1151, 55.5364),
    "SO": ("Somalia", "SOM", "Sub-Saharan Africa", 5.1521, 46.1996),
    "SR": ("Suriname", "SUR", "Latin America & Caribbean", 3.9193, -56.0278),
    "SV": ("El Salvador", "SLV", "Latin America & Caribbean", 13.7942, -88.8965),
    "SZ": ("Eswatini", "SWZ", "Sub-Saharan Africa", -26.5225, 31.4659),
    "TD": ("Chad", "TCD", "Sub-Saharan Africa", 15.4542, 18.7322),
    "TM": ("Turkmenistan", "TKM", "Europe & Central Asia", 38.9697, 59.5563),
    "TO": ("Tonga", "TON", "East Asia & Pacific", -21.1790, -175.1982),
    "UG": ("Uganda", "UGA", "Sub-Saharan Africa", 1.3733, 32.2903),
    "VI": ("U.S. Virgin Islands", "VIR", "Latin America & Caribbean", 18.3358, -64.8963),
    "XK": ("Kosovo", "XKX", "Europe & Central Asia", 42.6026, 20.9030),
    "ZM": ("Zambia", "ZMB", "Sub-Saharan Africa", -13.1339, 27.8493),
}

# 3. Fix existing Namibia rows where country == 'Namibia' but iso2c is NaN
namibia_mask = (df['country'] == 'Namibia') & (df['iso2c'].isna())
if namibia_mask.any():
    df.loc[namibia_mask, 'iso2c'] = 'NA'
    if 'iso3c' in df.columns:
        df.loc[namibia_mask, 'iso3c'] = 'NAM'
    df.loc[namibia_mask, 'lat'] = -22.9576
    df.loc[namibia_mask, 'lng'] = 18.4904
    df.loc[namibia_mask, 'region'] = 'Sub-Saharan Africa'
    print(f"✅ Repaired {namibia_mask.sum()} existing Namibia ('NA') rows in data.parquet")

# 4. Inspect raw files in GitHub repo to pull data for the remaining missing countries
BASE_URL = "https://raw.githubusercontent.com/mbermudezmoTec/China-s-great-leap-forward-in-the-chemical-space/main/data/raw"
# Map chemical category in your app to the raw TSV filenames in the repo
RAW_FILES = {
    "All": "Country_weighted_allSubs.tsv",
    "Organic": "Country_weighted_organic.tsv",
    "Organometallic": "Country_weighted_organometallic.tsv",
    "Rare-Earths": "Country_weighted_rareEarths.tsv"
}

existing_solo_pairs = set(
    zip(
        df[df['is_collab'] == False]['iso2c'].dropna().str.upper(),
        df[df['is_collab'] == False]['chemical'].dropna()
    )
)

new_rows = []
for chem_cat, fname in RAW_FILES.items():
    url = f"{BASE_URL}/{fname}"
    resp = requests.get(url)
    if resp.status_code != 200:
        # Fallback to Absolute_collab if Country_weighted_* has a slightly different name
        print(f"Skipping {fname} (HTTP {resp.status_code})")
        continue

    raw_tsv = pd.read_csv(io.StringIO(resp.text), sep="\t", index_col=0, keep_default_na=False)
    # Year columns (1996..2022)
    year_cols = [c for c in raw_tsv.columns if str(c).strip().isdigit()]

    # In Country_weighted_*.tsv, check if values are fractions (0-1) or percentages (0-100) or counts
    # Sum of column for a year: if ~1.0, multiply by 100; if ~100.0, keep as is; if >1000, divide by col sum * 100
    col_sums = raw_tsv[year_cols].apply(pd.to_numeric, errors='coerce').sum(axis=0)

    for iso_code, (c_name, iso3, region, lat, lng) in MISSING_META.items():
        if (iso_code, chem_cat) in existing_solo_pairs:
            continue
        if iso_code not in raw_tsv.index:
            continue

        series = pd.to_numeric(raw_tsv.loc[iso_code, year_cols], errors='coerce').fillna(0.0)
        for yr_str, val in series.items():
            yr = int(yr_str)
            col_total = float(col_sums[yr_str])
            if col_total > 150.0:
                pct = (float(val) / col_total) * 100.0
            elif col_total <= 1.5 and col_total > 0:
                pct = float(val) * 100.0
            else:
                pct = float(val)

            if pct > 0:
                row_dict = {
                    'country': c_name,
                    'iso2c': iso_code,
                    'iso3c': iso3,
                    'year': yr,
                    'percentage': round(pct, 6),
                    'chemical': chem_cat,
                    'is_collab': False,
                    'lat': lat,
                    'lng': lng,
                    'region': region,
                    'cc': '#3b82f6'
                }
                new_rows.append(row_dict)

if new_rows:
    new_df = pd.DataFrame(new_rows)
    # Align columns with existing df
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df[df.columns]
    df_combined = pd.concat([df, new_df], ignore_index=True)
else:
    df_combined = df

df_combined.to_parquet(PARQUET_PATH, index=False)
solo_after = df_combined[df_combined['is_collab'] == False]
print(f"\n🎉 DONE! Updated {PARQUET_PATH}:")
print(f"   Unique solo countries before: 138 -> AFTER: {solo_after['iso2c'].dropna().nunique()}")
print(f"   Added {len(new_rows):,} new country-year records across chemical subspaces.")