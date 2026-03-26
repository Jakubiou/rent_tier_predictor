import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("cleaner.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_FILE   = os.path.join(PROJECT_ROOT, "data", "raw_listings.csv")
OUTPUT_CLEAN = os.path.join(PROJECT_ROOT, "data", "clean_listings.csv")
OUTPUT_NORM  = os.path.join(PROJECT_ROOT, "data", "clean_listings_normalized.csv")
POI_COLS     = ["skola_m", "park_m", "mhd_m", "lekarna_m", "supermarket_m"]

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    log.info(f"Načteno {len(df)} řádků, {len(df.columns)} sloupců")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    original_len = len(df)

    df = df.drop_duplicates(subset="id")
    log.info(f"Po odstranění duplicit: {len(df)} (smazáno {original_len - len(df)})")

    df["plocha_m2"] = df["nazev"].str.extract(r"(\d+)\s*m²").astype(float)
    log.info(f"Plocha parsována z názvu: {df['plocha_m2'].notna().sum()} / {len(df)}")

    df["cena_kc"] = pd.to_numeric(df["cena_kc"], errors="coerce")
    df = df[df["cena_kc"].between(2_000, 150_000)].copy()
    log.info(f"Po filtraci cen [2k–150k Kč]: {len(df)}")

    df = df[df["plocha_m2"].between(10, 500)].copy()
    log.info(f"Po filtraci plochy [10–500 m²]: {len(df)}")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    mask_gps = df["lat"].between(48.5, 51.1) & df["lon"].between(12.1, 18.9)
    df = df[mask_gps].copy()
    log.info(f"Po filtraci GPS: {len(df)}")

    for col in POI_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(upper=5000)

    log.info(f"Čištění hotovo. Výsledek: {len(df)} řádků")
    return df.reset_index(drop=True)

DISPOSITION_MAP = {
    "garsonka": 1, "garsoniéra": 1,
    "1+kk": 2, "1+1": 2,
    "2+kk": 3, "2+1": 3,
    "3+kk": 4, "3+1": 4,
    "4+kk": 5, "4+1": 5, "5+kk": 5, "5+1": 5,
}


def extract_disposition_score(nazev: str) -> int:
    if not isinstance(nazev, str):
        return 3
    nazev_lower = nazev.lower()
    for key, score in DISPOSITION_MAP.items():
        if key in nazev_lower:
            return score
    return 3


def add_features(df: pd.DataFrame) -> pd.DataFrame:

    df["dispozice_skore"] = df["nazev"].apply(extract_disposition_score)

    df["mesto"] = (
        df["lokalita"]
        .str.split(",").str[-1]
        .str.split(" - ").str[0]
        .str.strip()
    )

    df["velke_mesto"] = df["mesto"].str.lower().str.contains(
        r"prah|brno|ostrava", na=False
    ).astype(int)

    df["mhd_dostupnost"] = (df["mhd_m"].fillna(1000) / 1000).clip(0, 1).round(3)

    log.info("Feature engineering hotov")
    return df


def add_price_tier(df: pd.DataFrame) -> pd.DataFrame:
    t1 = df["cena_kc"].quantile(0.33)
    t2 = df["cena_kc"].quantile(0.66)

    df["cenove_pasmo"] = pd.cut(
        df["cena_kc"],
        bins=[-np.inf, t1, t2, np.inf],
        labels=[1, 2, 3],
    ).astype(int)

    log.info(f"Tercilové hranice: levný < {t1:.0f} Kč | střední < {t2:.0f} Kč | drahý")
    dist = df["cenove_pasmo"].value_counts().sort_index()
    log.info(f"Distribuce cenových pásem:\n{dist.to_string()}")
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_mesta = df["mesto"].value_counts().nlargest(15).index
    df["mesto_group"] = df["mesto"].where(df["mesto"].isin(top_mesta), other="ostatni")
    mesto_dummies = pd.get_dummies(df["mesto_group"], prefix="mesto")
    df = pd.concat([df, mesto_dummies], axis=1)

    drop_cols = [
        "id", "nazev", "lokalita", "zdroj",
        "lat", "lon",
        "mesto", "mesto_group",
        "cena_kc",
    ]
    df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])

    numeric_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        missing = df_model[col].isna().sum()
        if missing > 0:
            med = df_model[col].median()
            log.info(f"  Imputuji '{col}': {missing} NaN → {med:.1f}")
            df_model[col] = df_model[col].fillna(med)

    feature_cols = [
        c for c in numeric_cols
        if c != "cenove_pasmo" and not c.startswith("mesto_")
    ]
    scaler = MinMaxScaler()
    df_normed = df_model.copy()
    df_normed[feature_cols] = scaler.fit_transform(df_model[feature_cols])

    log.info(f"Preprocessing hotov. Features: {feature_cols}")
    return df_model, df_normed


def main():
    log.info("=" * 60)
    log.info("Spuštěn cleaner.py")

    df = load(INPUT_FILE)
    df = clean(df)
    df = add_features(df)
    df = add_price_tier(df)

    df.to_csv(OUTPUT_CLEAN, index=False, encoding="utf-8-sig")
    log.info(f"Čistá data uložena → {OUTPUT_CLEAN}")

    df_model, df_normed = preprocess(df)
    df_normed.to_csv(OUTPUT_NORM, index=False, encoding="utf-8-sig")
    log.info(f"Normalizovaná data uložena → {OUTPUT_NORM}")

    log.info("\n--- Přehled ---")
    log.info(f"Řádků: {len(df)}")
    log.info(f"Sloupců modelu: {len(df_model.columns)}")
    log.info(f"Chybějící POI hodnoty:\n{df[POI_COLS].isna().sum().to_string()}")
    log.info(f"Statistiky ceny:\n{df['cena_kc'].describe().round(0).to_string()}")
    log.info(f"Statistiky plochy:\n{df['plocha_m2'].describe().round(1).to_string()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()