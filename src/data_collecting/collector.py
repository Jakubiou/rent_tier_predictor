import time, random, json, math, os
from datetime import datetime

import requests
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_listings1.csv")
PHOTO_DIR = os.path.join(OUTPUT_DIR, "photos")

TARGET_SREALITY = 7500
TARGET_BEZREALITKY = 5000
SLEEP_MIN, SLEEP_MAX = 1.2, 3.5
MAX_RETRIES = 3
PER_PAGE = 20

HEADERS_SREALITY = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "cs-CZ,cs;q=0.9,en;q=0.8",
    "Referer": "https://www.sreality.cz/",
}

HEADERS_BEZ = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "cs-CZ,cs;q=0.9",
    "Content-Type": "application/json",
    "Origin": "https://www.bezrealitky.cz",
    "Referer": "https://www.bezrealitky.cz/",
}

SREALITY_BASE = (
    "https://www.sreality.cz/api/cs/v2/estates"
    "?category_main_cb=1&category_type_cb=2"
    "&per_page={per_page}&page={page}"
)

POI_QUERIES = {
    "skola_m": '[amenity~"school|kindergarten"]',
    "park_m": '[leisure~"park|playground|garden"]',
    "mhd_m": '[highway~"bus_stop|tram_stop"][public_transport~"stop_position|platform"]',
    "lekarna_m": '[amenity="pharmacy"]',
    "supermarket_m": '[shop~"supermarket|grocery|convenience"]',
}

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
POI_CACHE_FILE = os.path.join(OUTPUT_DIR, "poi_czech_republic.json")


def human_sleep(min_s=None, max_s=None):
    time.sleep(random.uniform(min_s or SLEEP_MIN, max_s or SLEEP_MAX))


def safe_get(url, params=None, headers=None, attempt=1):
    try:
        r = requests.get(url, headers=headers or HEADERS_SREALITY,
                         params=params, timeout=20)
        if r.status_code == 200:
            return r
        if r.status_code == 429:
            time.sleep(60 * attempt)
            if attempt < MAX_RETRIES:
                return safe_get(url, params, headers, attempt + 1)
        elif r.status_code in (403, 503):
            time.sleep(30)
            if attempt < MAX_RETRIES:
                return safe_get(url, params, headers, attempt + 1)
    except requests.exceptions.RequestException as e:
        if attempt < MAX_RETRIES:
            time.sleep(10 * attempt)
            return safe_get(url, params, headers, attempt + 1)
    return None


def safe_post(url, payload, headers=None, attempt=1):
    try:
        r = requests.post(url, json=payload,
                          headers=headers or HEADERS_BEZ, timeout=20)
        if r.status_code == 200:
            return r
        if r.status_code == 429:
            time.sleep(60 * attempt)
            if attempt < MAX_RETRIES:
                return safe_post(url, payload, headers, attempt + 1)
    except requests.exceptions.RequestException as e:
        if attempt < MAX_RETRIES:
            time.sleep(10 * attempt)
            return safe_post(url, payload, headers, attempt + 1)
    return None


def download_photo(url, path, headers=None):
    try:
        r = requests.get(url, headers=headers or {"User-Agent": HEADERS_SREALITY["User-Agent"]},
                         timeout=15)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except:
        pass
    return False


def fetch_sreality_page(page):
    url = SREALITY_BASE.format(per_page=PER_PAGE, page=page)
    r = safe_get(url)
    if r is None:
        return []
    try:
        data = r.json()
    except json.JSONDecodeError:
        return []

    estates = data.get("_embedded", {}).get("estates", [])
    records = []
    for e in estates:
        hash_id = e.get("hash_id")
        locality_raw = e.get("locality", "")
        locality = locality_raw.get("value", "") if isinstance(locality_raw, dict) else str(locality_raw or "")
        gps = e.get("gps", {})
        lat = gps.get("lat") if isinstance(gps, dict) else None
        lon = gps.get("lon") if isinstance(gps, dict) else None

        photo_url = None
        try:
            images = e.get("_links", {}).get("images", [])
            if images and isinstance(images, list):
                href = images[0].get("href", "")
                if href:
                    photo_url = href.replace("{res}", "400x300").replace("{lang}", "cs")
        except:
            pass
        if not photo_url:
            try:
                imgs = e.get("_embedded", {}).get("images", [])
                if imgs:
                    photo_url = imgs[0].get("_links", {}).get("self", {}).get("href", "")
            except:
                pass

        records.append({
            "id": f"sr_{hash_id}",
            "nazev": e.get("name", ""),
            "cena_kc": e.get("price", 0),
            "lokalita": locality,
            "lat": lat,
            "lon": lon,
            "plocha_m2": None,
            "zdroj": "sreality",
            "photo_url": photo_url,
        })
    return records


def collect_sreality(target=TARGET_SREALITY):
    all_records, page = [], 1
    with tqdm(total=target, desc="Sreality") as pbar:
        while len(all_records) < target:
            records = fetch_sreality_page(page)
            if not records:
                break
            all_records.extend(records)
            pbar.update(len(records))
            page += 1
            human_sleep()
    df = pd.DataFrame(all_records).drop_duplicates(subset="id")
    return df


BEZ_GRAPHQL_URL = "https://www.bezrealitky.cz/api/graphql"

BEZ_QUERY = """
query AdvertList($limit: Int, $offset: Int, $offerType: OfferType, $estateType: [EstateType]) {
  advertList(
    limit: $limit
    offset: $offset
    offerType: $offerType
    estateType: $estateType
    regionOsmIds: []
  ) {
    list {
      id
      uri
      price
      charges
      gps {
        lat
        lng
      }
      address
      surface
      disposition
      mainOfferPhoto {
        url
      }
    }
    totalCount
  }
}
"""

BEZ_DISPOSITION_MAP = {
    "DISP_1_KK": "1+kk",
    "DISP_1_1": "1+1",
    "DISP_2_KK": "2+kk",
    "DISP_2_1": "2+1",
    "DISP_3_KK": "3+kk",
    "DISP_3_1": "3+1",
    "DISP_4_KK": "4+kk",
    "DISP_4_1": "4+1",
    "DISP_5_KK": "5+kk",
    "DISP_5_1": "5+1",
    "GARSONIERA": "garsoniéra",
}


def fetch_bezrealitky_page(offset, limit=20):
    payload = {
        "query": BEZ_QUERY,
        "variables": {
            "limit": limit,
            "offset": offset,
            "offerType": "RENT",
            "estateType": ["FLAT"],
        },
    }
    r = safe_post(BEZ_GRAPHQL_URL, payload)
    if r is None:
        return [], 0
    try:
        data = r.json()
    except json.JSONDecodeError:
        return [], 0

    advert_list = data.get("data", {}).get("advertList", {})
    total = advert_list.get("totalCount", 0)
    items = advert_list.get("list", [])

    records = []
    for e in items:
        gps = e.get("gps") or {}
        lat = gps.get("lat")
        lon = gps.get("lng")
        disp_raw = e.get("disposition", "")
        disp_str = BEZ_DISPOSITION_MAP.get(disp_raw, disp_raw or "")
        surface = e.get("surface")
        address = e.get("address", "")
        price = e.get("price", 0) or 0
        charges = e.get("charges", 0) or 0
        total_price = price + charges if charges else price
        nazev = f"Pronájem bytu {disp_str} {surface} m²" if surface and disp_str else ""

        photo_url = None
        try:
            photo_url = e["mainOfferPhoto"]["url"]
        except (KeyError, TypeError):
            pass

        records.append({
            "id": f"bz_{e.get('id', '')}",
            "nazev": nazev,
            "cena_kc": total_price,
            "lokalita": address,
            "lat": lat,
            "lon": lon,
            "plocha_m2": float(surface) if surface else None,
            "zdroj": "bezrealitky",
            "photo_url": photo_url,
        })
    return records, total


def collect_bezrealitky(target=TARGET_BEZREALITKY):
    all_records = []
    offset = 0
    limit = 20
    total = None

    with tqdm(total=target, desc="Bezrealitky") as pbar:
        while len(all_records) < target:
            records, total_count = fetch_bezrealitky_page(offset, limit)

            if total is None:
                total = total_count

            if not records:
                break

            all_records.extend(records)
            pbar.update(len(records))
            offset += limit

            if offset >= (total or 0):
                break

            human_sleep(0.8, 2.0)

    df = pd.DataFrame(all_records).drop_duplicates(subset="id")
    return df


def download_photos(df):
    os.makedirs(PHOTO_DIR, exist_ok=True)
    photo_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fotky"):
        if row.get("photo_url"):
            path = os.path.join(PHOTO_DIR, f"{row['id']}.jpg")
            if os.path.exists(path):
                photo_paths.append(path)
            elif download_photo(row["photo_url"], path):
                photo_paths.append(path)
            else:
                photo_paths.append(None)
            time.sleep(0.3)
        else:
            photo_paths.append(None)

    df = df.copy()
    df["photo_path"] = photo_paths
    return df


def download_czech_poi():
    if os.path.exists(POI_CACHE_FILE):
        with open(POI_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
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
                r = requests.post(server, data={"data": query},
                                  headers={"User-Agent": HEADERS_SREALITY["User-Agent"]},
                                  timeout=200)
                if r.status_code == 200:
                    raw = r.json()
                    poi = _parse_poi(raw["elements"])
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    with open(POI_CACHE_FILE, "w", encoding="utf-8") as f:
                        json.dump(poi, f)
                    return poi
            except Exception as e:
                pass
            time.sleep(15 * attempt)
    return None


def _parse_poi(elements):
    buckets = {col: {"lat": [], "lon": []} for col in POI_QUERIES}
    for el in elements:
        if el.get("type") != "node":
            continue
        elat, elon = el.get("lat"), el.get("lon")
        if elat is None or elon is None:
            continue
        tags = el.get("tags", {})
        am = tags.get("amenity", "")
        le = tags.get("leisure", "")
        sh = tags.get("shop", "")
        hw = tags.get("highway", "")
        pt = tags.get("public_transport", "")
        if am in ("school", "kindergarten"):
            buckets["skola_m"]["lat"].append(elat)
            buckets["skola_m"]["lon"].append(elon)
        if le in ("park", "playground", "garden"):
            buckets["park_m"]["lat"].append(elat)
            buckets["park_m"]["lon"].append(elon)
        if hw == "bus_stop" or pt == "stop_position":
            buckets["mhd_m"]["lat"].append(elat)
            buckets["mhd_m"]["lon"].append(elon)
        if am == "pharmacy":
            buckets["lekarna_m"]["lat"].append(elat)
            buckets["lekarna_m"]["lon"].append(elon)
        if sh in ("supermarket", "grocery", "convenience"):
            buckets["supermarket_m"]["lat"].append(elat)
            buckets["supermarket_m"]["lon"].append(elon)
    return buckets


def nearest_from_bucket(lat, lon, bucket):
    lats, lons = bucket["lat"], bucket["lon"]
    if not lats:
        return None
    R, min_d = 6_371_000, float("inf")
    lat_r = math.radians(lat)
    for plat, plon in zip(lats, lons):
        if abs(plat - lat) > 0.05 or abs(plon - lon) > 0.05:
            continue
        dphi = math.radians(plat - lat)
        dlam = math.radians(plon - lon)
        a = (math.sin(dphi / 2) ** 2
             + math.cos(lat_r) * math.cos(math.radians(plat)) * math.sin(dlam / 2) ** 2)
        d = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        if d < min_d:
            min_d = d
    return round(min_d, 1) if min_d != float("inf") else None


def enrich_with_poi(df):
    poi = download_czech_poi()
    if poi is None:
        for col in POI_QUERIES:
            df[col] = None
        return df
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
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PHOTO_DIR, exist_ok=True)

    df_sr = collect_sreality(TARGET_SREALITY)
    df_bz = collect_bezrealitky(TARGET_BEZREALITKY)

    df = pd.concat([df_sr, df_bz], ignore_index=True)
    df = df.drop_duplicates(subset="id")

    df = download_photos(df)
    df = enrich_with_poi(df)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()