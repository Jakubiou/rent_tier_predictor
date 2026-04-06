import os, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw_listings1.csv")
OUTPUT_CLEAN = os.path.join(PROJECT_ROOT, "data", "clean_listings1.csv")
CITY_INDEX_FILE = os.path.join(PROJECT_ROOT, "data", "city_price_index1.json")
TIER_META_FILE = os.path.join(PROJECT_ROOT, "data", "tier_meta1.json")
PHOTO_SCORE_PATH = os.path.join(PROJECT_ROOT, "data", "photo_quality1.json")

POI_COLS = ["skola_m", "park_m", "mhd_m", "lekarna_m", "supermarket_m"]
MIN_CITY_DISP = 100


def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def clean(df):
    df = df.drop_duplicates(subset="id")

    df["plocha_m2"] = df["nazev"].str.extract(r"(\d+)\s*m²").astype(float)
    df["cena_kc"] = pd.to_numeric(df["cena_kc"], errors="coerce")

    df = df[df["cena_kc"].between(2_000, 150_000)].copy()
    df = df[df["plocha_m2"].between(10, 500)].copy()

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    mask = df["lat"].between(48.5, 51.1) & df["lon"].between(12.1, 18.9)
    df = df[mask].copy()

    for col in POI_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(upper=5000)

    return df.reset_index(drop=True)


DISPOSITION_MAP = {
    "garsonka": 1, "garsoniéra": 1,
    "1+kk": 2, "1+1": 2,
    "2+kk": 3, "2+1": 3,
    "3+kk": 4, "3+1": 4,
    "4+kk": 5, "4+1": 5, "5+kk": 5, "5+1": 5,
}


def extract_disposition_score(nazev):
    if not isinstance(nazev, str):
        return 3
    nazev_lower = nazev.lower()
    for key, score in DISPOSITION_MAP.items():
        if key in nazev_lower:
            return score
    return 3


def add_features(df):
    df["dispozice_skore"] = df["nazev"].apply(extract_disposition_score)

    df["mesto"] = (
        df["lokalita"].str.split(",").str[-1]
        .str.split(" - ").str[0].str.strip()
    )

    df["mhd_dostupnost"] = (df["mhd_m"].fillna(1000) / 1000).clip(0, 1).round(3)

    for col in POI_COLS:
        med = df[col].median()
        df[col] = df[col].fillna(med)

    return df


def add_photo_score(df):
    if os.path.exists(PHOTO_SCORE_PATH):
        with open(PHOTO_SCORE_PATH, encoding="utf-8") as f:
            photo_map = json.load(f)
        df["photo_score"] = df["id"].astype(str).map(photo_map)
        df["photo_score"] = df["photo_score"].fillna(df["photo_score"].median())
    else:
        df["photo_score"] = 0.5
    return df


def build_city_price_index(df):
    df = df.copy()
    df["_cpm2"] = df["cena_kc"] / df["plocha_m2"].replace(0, np.nan)
    counts = df.groupby("mesto")["_cpm2"].count()
    means = df.groupby("mesto")["_cpm2"].mean()
    global_mean = df["_cpm2"].mean()

    index = {
        city: round(float(means[city]), 1) if counts[city] >= 10
              else round(float(global_mean), 1)
        for city in means.index
    }
    with open(CITY_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump({"global_mean": round(global_mean, 1), "cities": index},
                  f, ensure_ascii=False, indent=2)
    return index, global_mean


def encode_mesto(df):
    le = LabelEncoder()
    df["mesto_enc"] = le.fit_transform(df["mesto"].fillna("neznámo"))
    return df, le


def add_price_tier(df):
    counts = df.groupby(["mesto", "dispozice_skore"]).size()
    city_med = df.groupby(["mesto", "dispozice_skore"])["cena_kc"].median()
    disp_med = df.groupby("dispozice_skore")["cena_kc"].median()

    ref_medians = []
    for _, row in df.iterrows():
        key = (row["mesto"], row["dispozice_skore"])
        if counts.get(key, 0) >= MIN_CITY_DISP:
            ref_medians.append(float(city_med.get(key, disp_med[row["dispozice_skore"]])))
        else:
            ref_medians.append(float(disp_med[row["dispozice_skore"]]))

    df["ref_median"] = ref_medians
    df["cena_rel"] = (df["cena_kc"] / df["ref_median"]).round(4)

    t1_rel = float(df["cena_rel"].quantile(0.30))
    t2_rel = float(df["cena_rel"].quantile(0.70))

    df["tier"] = pd.cut(
        df["cena_rel"],
        bins=[-np.inf, t1_rel, t2_rel, np.inf],
        labels=[0, 1, 2],
    ).astype(int)

    disp_names = {1: "Garsoniéra", 2: "1+kk / 1+1", 3: "2+kk / 2+1",
                  4: "3+kk / 3+1", 5: "4+kk a více"}
    tier_meta = {
        "t1_rel": t1_rel,
        "t2_rel": t2_rel,
        "per_disp": {
            int(d): {
                "name": disp_names.get(d, str(d)),
                "median": round(float(disp_med[d])),
                "t1_abs": round(float(disp_med[d]) * t1_rel),
                "t2_abs": round(float(disp_med[d]) * t2_rel),
                "n": int((df["dispozice_skore"] == d).sum()),
            }
            for d in sorted(df["dispozice_skore"].unique())
        },
        "city_disp_medians": {
            f"{m}|{int(d)}": round(float(v))
            for (m, d), v in city_med.items()
            if counts.get((m, d), 0) >= MIN_CITY_DISP
        },
        "disp_medians": {int(d): round(float(v)) for d, v in disp_med.items()},
    }
    with open(TIER_META_FILE, "w", encoding="utf-8") as f:
        json.dump(tier_meta, f, ensure_ascii=False, indent=2)

    return df, t1_rel, t2_rel, city_med, disp_med, counts


def main():
    df = load(INPUT_FILE)
    df = clean(df)
    df = add_features(df)
    df = add_photo_score(df)
    df, _ = encode_mesto(df)

    build_city_price_index(df)

    df, t1, t2, city_med, disp_med, counts = add_price_tier(df)

    df.to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()