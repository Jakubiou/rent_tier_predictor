import time
import random
import json
import math
import os
import logging
from datetime import datetime

import requests
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("collector.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_listings.csv")
TARGET_ROWS = 3500
SLEEP_MIN = 1.2
SLEEP_MAX = 3.5
MAX_RETRIES = 3

SREALITY_PER_PAGE = 20

SREALITY_BASE = (
    "https://www.sreality.cz/api/cs/v2/estates"
    "?category_main_cb=1"
    "&category_type_cb=2"
    "&per_page={per_page}"
    "&page={page}"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "cs-CZ,cs;q=0.9,en;q=0.8",
    "Referer": "https://www.sreality.cz/",
}

POI_QUERIES = {
    "skola_m":       '[amenity~"school|kindergarten"]',
    "park_m":        '[leisure~"park|playground|garden"]',
    "mhd_m":         '[highway~"bus_stop|tram_stop"][public_transport~"stop_position|platform"]',
    "lekarna_m":     '[amenity="pharmacy"]',
    "supermarket_m": '[shop~"supermarket|grocery|convenience"]',
}


def human_sleep(min_s=SLEEP_MIN, max_s=SLEEP_MAX):
    t = random.uniform(min_s, max_s)
    time.sleep(t)


def safe_get(url: str, params=None, attempt=1) -> requests.Response | None:
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        if r.status_code == 200:
            return r
        elif r.status_code == 429:
            wait = 60 * attempt          # při rate-limit čekáme déle
            log.warning(f"429 Rate limit – čekám {wait}s (pokus {attempt})")
            time.sleep(wait)
            if attempt < MAX_RETRIES:
                return safe_get(url, params, attempt + 1)
        elif r.status_code in (403, 503):
            log.warning(f"HTTP {r.status_code} – čekám 30s a zkouším znovu")
            time.sleep(30)
            if attempt < MAX_RETRIES:
                return safe_get(url, params, attempt + 1)
        else:
            log.error(f"HTTP {r.status_code} pro {url}")
    except requests.exceptions.RequestException as e:
        log.error(f"Síťová chyba ({e}) – pokus {attempt}/{MAX_RETRIES}")
        if attempt < MAX_RETRIES:
            time.sleep(10 * attempt)
            return safe_get(url, params, attempt + 1)
    return None


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_page(page: int) -> list[dict]:
    url = SREALITY_BASE.format(per_page=SREALITY_PER_PAGE, page=page)
    r = safe_get(url)
    if r is None:
        return []
    try:
        data = r.json()
    except json.JSONDecodeError:
        log.error(f"Špatný JSON na stránce {page}")
        return []

    estates = data.get("_embedded", {}).get("estates", [])
    records = []
    for e in estates:
        disposition = e.get("name", "")
        price_raw   = e.get("price", 0)

        locality_raw = e.get("locality", "")
        if isinstance(locality_raw, dict):
            locality = locality_raw.get("value", "")
        else:
            locality = str(locality_raw) if locality_raw else ""

        gps_raw = e.get("gps", {})
        if isinstance(gps_raw, dict):
            lat = gps_raw.get("lat")
            lon = gps_raw.get("lon")
        else:
            lat, lon = None, None

        area_m2 = None
        for item in e.get("labelsRichtext", []) or []:
            if not isinstance(item, dict):
                continue
            txt = item.get("value", "")
            if "m²" in txt:
                try:
                    area_m2 = float(
                        txt.replace("\xa0", "").replace("m²", "").strip()
                    )
                except ValueError:
                    pass
        records.append({
            "id":          e.get("hash_id"),
            "nazev":       disposition,
            "cena_kc":     price_raw,
            "lokalita":    locality,
            "lat":         lat,
            "lon":         lon,
            "plocha_m2":   area_m2,
            "zdroj":       "sreality",
        })
    return records


def collect_listings(target: int = TARGET_ROWS) -> pd.DataFrame:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_records = []
    page = 1

    log.info(f"Startuji sběr dat (cíl: {target} inzerátů)…")

    with tqdm(total=target, desc="Inzeráty") as pbar:
        while len(all_records) < target:
            records = fetch_page(page)
            if not records:
                log.warning(f"Stránka {page} vrátila 0 záznamů – pravděpodobně konec dat")
                break

            all_records.extend(records)
            pbar.update(len(records))
            log.info(f"Stránka {page}: +{len(records)} → celkem {len(all_records)}")
            page += 1
            human_sleep()

    df = pd.DataFrame(all_records).drop_duplicates(subset="id")
    log.info(f"Sesbíráno {len(df)} unikátních inzerátů")
    return df


OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
POI_CACHE_FILE = os.path.join(OUTPUT_DIR, "poi_czech_republic.json")


def download_czech_poi() -> dict | None:
    if os.path.exists(POI_CACHE_FILE):
        log.info(f"POI cache nalezena → načítám {POI_CACHE_FILE}")
        with open(POI_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)

    log.info("Stahuji POI data pro celou ČR jedním dotazem (~1–3 minuty)…")

    query = """
[out:json][timeout:180];
area["ISO3166-1"="CZ"][admin_level=2]->.cr;
(
  node["amenity"~"school|kindergarten"](area.cr);
  node["leisure"~"park|playground|garden"](area.cr);
  node["highway"="bus_stop"](area.cr);
  node["public_transport"="stop_position"](area.cr);
  node["amenity"="pharmacy"](area.cr);
  node["shop"~"supermarket|grocery|convenience"](area.cr);
);
out body;
"""
    for attempt in range(1, 5):
        for server in OVERPASS_SERVERS:
            try:
                log.info(f"  Zkouším {server} (pokus {attempt})…")
                r = requests.post(
                    server,
                    data={"data": query},
                    headers={"User-Agent": HEADERS["User-Agent"]},
                    timeout=200,
                )
                if r.status_code == 200:
                    raw = r.json()
                    poi = _parse_poi_to_buckets(raw["elements"])
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    with open(POI_CACHE_FILE, "w", encoding="utf-8") as f:
                        json.dump(poi, f)
                    total = sum(len(v["lat"]) for v in poi.values())
                    log.info(f"  Hotovo: {total} POI bodů staženo a uloženo do cache")
                    return poi
                else:
                    log.warning(f"  HTTP {r.status_code} od {server}")
            except Exception as e:
                log.warning(f"  Chyba: {e}")
            time.sleep(15 * attempt)

    log.error("Nepodařilo se stáhnout POI – zkus spustit znovu za chvíli")
    return None


def _parse_poi_to_buckets(elements: list) -> dict:
    buckets = {col: {"lat": [], "lon": []} for col in POI_QUERIES}

    for el in elements:
        if el.get("type") != "node":
            continue
        elat = el.get("lat")
        elon = el.get("lon")
        if elat is None or elon is None:
            continue
        tags = el.get("tags", {})
        amenity = tags.get("amenity", "")
        leisure = tags.get("leisure", "")
        shop    = tags.get("shop", "")
        hw      = tags.get("highway", "")
        pt      = tags.get("public_transport", "")

        if amenity in ("school", "kindergarten"):
            buckets["skola_m"]["lat"].append(elat)
            buckets["skola_m"]["lon"].append(elon)
        if leisure in ("park", "playground", "garden"):
            buckets["park_m"]["lat"].append(elat)
            buckets["park_m"]["lon"].append(elon)
        if hw == "bus_stop" or pt == "stop_position":
            buckets["mhd_m"]["lat"].append(elat)
            buckets["mhd_m"]["lon"].append(elon)
        if amenity == "pharmacy":
            buckets["lekarna_m"]["lat"].append(elat)
            buckets["lekarna_m"]["lon"].append(elon)
        if shop in ("supermarket", "grocery", "convenience"):
            buckets["supermarket_m"]["lat"].append(elat)
            buckets["supermarket_m"]["lon"].append(elon)

    return buckets


def nearest_from_bucket(lat: float, lon: float, bucket: dict) -> float | None:
    lats = bucket["lat"]
    lons = bucket["lon"]
    if not lats:
        return None

    R = 6_371_000
    lat_r = math.radians(lat)
    min_d = float("inf")

    for plat, plon in zip(lats, lons):
        if abs(plat - lat) > 0.05 or abs(plon - lon) > 0.05:
            continue
        dphi = math.radians(plat - lat)
        dlam = math.radians(plon - lon)
        a = (math.sin(dphi/2)**2
             + math.cos(lat_r) * math.cos(math.radians(plat)) * math.sin(dlam/2)**2)
        d = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        if d < min_d:
            min_d = d

    return round(min_d, 1) if min_d != float("inf") else None


def enrich_with_poi(df: pd.DataFrame) -> pd.DataFrame:
    poi = download_czech_poi()

    if poi is None:
        log.error("POI data nedostupná – ukládám bez POI sloupců")
        for col in POI_QUERIES:
            df[col] = None
        return df

    log.info(f"Počítám vzdálenosti lokálně pro {len(df)} bytů…")

    for col in POI_QUERIES:
        df[col] = None

    results = {col: [] for col in POI_QUERIES}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Vzdálenosti"):
        if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
            for col in POI_QUERIES:
                results[col].append(None)
            continue
        for col in POI_QUERIES:
            results[col].append(nearest_from_bucket(row["lat"], row["lon"], poi[col]))

    for col in POI_QUERIES:
        df[col] = results[col]

    log.info("Obohacení o POI hotovo")
    return df


def main():
    log.info("=" * 60)
    log.info(f"Spuštěno: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = collect_listings(TARGET_ROWS)

    df = enrich_with_poi(df)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    log.info(f"Hotovo! Uloženo {len(df)} řádků → {OUTPUT_FILE}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()