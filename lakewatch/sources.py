"""Open data sources. Everything here is free and needs no account or API key.

- Landsat Collection 2 Level-2, Sentinel-2 L2A, Sentinel-1 RTC: Microsoft Planetary Computer STAC
- JRC Global Surface Water v1.4 (used only to find the lake under the point the user picked)
- NASA POWER monthly rainfall
- Esri World Imagery Wayback release list and capture dates (tiles are loaded live by the report)
- OpenStreetMap place names via Overpass (optional, for naming neighbouring tanks)
"""
from __future__ import annotations

import calendar
import json
import os
import urllib.parse
import urllib.request

import numpy as np
from rasterio.enums import Resampling

from .geo import Grid, read_to_grid

for k, v in {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MULTIRANGE": "YES",
             "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES", "VSI_CACHE": "TRUE",
             "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "2",
             "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff"}.items():
    os.environ.setdefault(k, v)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
UA = {"User-Agent": "lakewatch/0.1 (+https://github.com/rohithreddy1095/lakes-and-land)"}
_cat = None


def catalog():
    global _cat
    if _cat is None:
        import pystac_client
        _cat = pystac_client.Client.open(STAC_URL)
    return _cat


def sign(item):
    import planetary_computer as pc
    return pc.sign(item)


def get_json(url: str, timeout: int = 40, data: bytes | None = None):
    req = urllib.request.Request(url, headers=UA, data=data)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


# ---------------------------------------------------------------- JRC Global Surface Water
def jrc_layer(layer: str, lon: float, lat: float, G: Grid) -> np.ndarray:
    lon0 = int(np.floor(lon / 10) * 10)
    lat0 = int(np.ceil(lat / 10) * 10)
    name = f"{abs(lon0)}{'E' if lon0 >= 0 else 'W'}_{abs(lat0)}{'N' if lat0 >= 0 else 'S'}"
    url = (f"/vsicurl/https://storage.googleapis.com/global-surface-water/downloads2021/"
           f"{layer}/{layer}_{name}v1_4_2021.tif")
    return read_to_grid(url, G)


# ---------------------------------------------------------------- Landsat
LANDSAT_SENSOR = {"landsat-4": "TM", "landsat-5": "TM", "landsat-7": "ETM+", "landsat-8": "OLI", "landsat-9": "OLI"}
SENSOR_LABEL = {"TM": "Landsat 4/5 · 30 m", "ETM+": "Landsat 7 · 30 m", "OLI": "Landsat 8/9 · 30 m"}


def search_landsat(bbox, start: str, end: str, cloud_lt: int = 80):
    items = catalog().search(collections=["landsat-c2-l2"], bbox=list(bbox), datetime=f"{start}/{end}",
                             query={"eo:cloud_cover": {"lt": cloud_lt}}).items()
    return [i for i in items if i.properties.get("platform") in LANDSAT_SENSOR
            and i.properties.get("landsat:collection_category", "T1") == "T1"]


def landsat_mndwi(item, G: Grid):
    """MNDWI x100 as int8 (-128 = no data / cloud / shadow) for one scene on grid G."""
    it = sign(item)
    qa = read_to_grid(it.assets["qa_pixel"].href, G).astype(np.uint16)
    valid = (qa & 0b11111) == 0  # not fill, dilated cloud, cirrus, cloud, cloud shadow
    out = np.full((G.h, G.w), -128, np.int8)
    if valid.mean() < 0.2:
        return out
    g = read_to_grid(it.assets["green"].href, G).astype(np.float32)
    s = read_to_grid(it.assets["swir16"].href, G).astype(np.float32)
    valid &= (g > 0) & (s > 0)
    g = g * 0.0000275 - 0.2
    s = s * 0.0000275 - 0.2
    mn = (g - s) / (g + s + 1e-6)
    out[valid] = np.clip(np.round(mn[valid] * 100), -100, 100).astype(np.int8)
    return out


def landsat_rgb(item, G: Grid) -> np.ndarray:
    it = sign(item)
    return np.dstack([read_to_grid(it.assets[b].href, G, Resampling.bilinear).astype(np.float32) * 0.0000275 - 0.2
                      for b in ("red", "green", "blue")])


# ---------------------------------------------------------------- Sentinel-2
def search_s2(bbox, start: str, end: str, cloud_lt: int = 80):
    items = list(catalog().search(collections=["sentinel-2-l2a"], bbox=list(bbox), datetime=f"{start}/{end}",
                                  query={"eo:cloud_cover": {"lt": cloud_lt}}).items())
    by = {}
    for it in items:  # keep the newest processing of each acquisition per tile
        k = it.datetime.strftime("%Y%m%dT%H%M") + it.properties.get("s2:mgrs_tile", "")
        pb = it.properties.get("s2:processing_baseline", "0")
        if k not in by or pb > by[k].properties.get("s2:processing_baseline", "0"):
            by[k] = it
    return sorted(by.values(), key=lambda i: i.datetime)


def _s2_offset(item) -> float:
    return -1000.0 if item.properties.get("s2:processing_baseline", "0") >= "04.00" else 0.0


def s2_indices(item, G: Grid):
    """(mndwi, ndvi) x100 as int8 (-128 = no data) for one Sentinel-2 scene on grid G."""
    it = sign(item)
    scl = read_to_grid(it.assets["SCL"].href, G)
    ok = ~np.isin(scl, [0, 1, 3, 8, 9, 10])
    mn = np.full((G.h, G.w), -128, np.int8)
    nv = mn.copy()
    if ok.mean() < 0.2:
        return mn, nv
    off = _s2_offset(item)
    b = {}
    for k in ("B03", "B04", "B08", "B11"):
        a = read_to_grid(it.assets[k].href, G).astype(np.float32)
        ok &= a > 0
        b[k] = (a + off) / 1e4
    m = (b["B03"] - b["B11"]) / (b["B03"] + b["B11"] + 1e-6)
    n = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + 1e-6)
    mn[ok] = np.clip(np.round(m[ok] * 100), -100, 100).astype(np.int8)
    nv[ok] = np.clip(np.round(n[ok] * 100), -100, 100).astype(np.int8)
    return mn, nv


def s2_rgb(item, G: Grid) -> np.ndarray:
    it = sign(item)
    off = _s2_offset(item)
    return np.dstack([(read_to_grid(it.assets[b].href, G, Resampling.bilinear).astype(np.float32) + off) / 1e4
                      for b in ("B04", "B03", "B02")])


# ---------------------------------------------------------------- Sentinel-1
def search_s1(bbox, start: str, end: str):
    return sorted(catalog().search(collections=["sentinel-1-rtc"], bbox=list(bbox),
                                   datetime=f"{start}/{end}").items(), key=lambda i: i.datetime)


def s1_vv_db(item, G: Grid) -> np.ndarray:
    a = read_to_grid(sign(item).assets["vv"].href, G).astype(np.float32)
    db = 10 * np.log10(np.maximum(a, 1e-4))
    db[a <= 0] = np.nan
    return db


# ---------------------------------------------------------------- rainfall
def rain_monthly(lon: float, lat: float, y0: int, y1: int) -> dict:
    """NASA POWER monthly precipitation, mm per month, keyed 'YYYY-MM'. Available from 1981."""
    y0 = max(1981, y0)
    url = ("https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=PRECTOTCORR&community=AG"
           f"&longitude={lon:.4f}&latitude={lat:.4f}&start={y0}&end={y1}&format=JSON")
    d = get_json(url, timeout=90)["properties"]["parameter"]["PRECTOTCORR"]
    out = {}
    for k, v in d.items():
        y, m = int(k[:4]), int(k[4:])
        if 1 <= m <= 12 and v is not None and v >= 0:
            out[f"{y}-{m:02d}"] = round(v * calendar.monthrange(y, m)[1], 1)
    return out


# ---------------------------------------------------------------- Esri Wayback (dates only)
WAYBACK_CFG = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
WAYBACK_TILEMAP = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tilemap"
WAYBACK_TILES = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
                 "default028mm/MapServer/tile/{rn}/{z}/{y}/{x}")


def wayback_captures(lon: float, lat: float, zoom: int = 16, workers: int = 12):
    """Distinct high-resolution captures available at a point in Esri World Imagery Wayback.

    Returns [{rn, release, date, src, res}] sorted by capture date. The report loads the tiles
    live from Esri; nothing is downloaded or redistributed here except these dates.
    """
    import concurrent.futures as cf
    import datetime as dt

    import mercantile
    cfg = get_json(WAYBACK_CFG)
    t = mercantile.tile(lon, lat, zoom)

    def changed(rn):
        try:
            tm = get_json(f"{WAYBACK_TILEMAP}/{rn}/{zoom}/{t.y}/{t.x}", timeout=25)
            if tm.get("data", [0])[0] != 1:
                return None
            return (tm.get("select") or [int(rn)])[0]
        except Exception:
            return None

    with cf.ThreadPoolExecutor(workers) as ex:
        uniq = sorted({r for r in ex.map(changed, list(cfg)) if r})

    def meta(rn):
        c = cfg.get(str(rn))
        if not c:
            return None
        q = urllib.parse.urlencode({"f": "json", "where": "1=1", "returnGeometry": "false",
                                    "outFields": "SRC_DATE2,SRC_DESC,SRC_RES,NICE_DESC",
                                    "geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint",
                                    "inSR": 4326, "spatialRel": "esriSpatialRelIntersects"})
        for lid in range(0, 12):  # finest metadata layer that has a footprint here
            try:
                f = get_json(f"{c['metadataLayerUrl']}/{lid}/query?{q}", timeout=25).get("features") or []
            except Exception:
                continue
            if f:
                a = f[0]["attributes"]
                if not a.get("SRC_DATE2"):
                    return None
                d = dt.datetime.fromtimestamp(a["SRC_DATE2"] / 1000, dt.timezone.utc).strftime("%Y-%m-%d")
                return dict(rn=int(rn), release=c["itemTitle"][-11:-1], date=d,
                            src=a.get("SRC_DESC") or a.get("NICE_DESC") or "", res=a.get("SRC_RES"))
        return None

    with cf.ThreadPoolExecutor(workers) as ex:
        rows = [m for m in ex.map(meta, uniq) if m]
    best = {}
    for m in sorted(rows, key=lambda m: m["release"]):  # first release that carried each capture
        best.setdefault(m["date"], m)
    return sorted(best.values(), key=lambda m: m["date"])


# ---------------------------------------------------------------- OpenStreetMap place names
def osm_places(west, south, east, north, timeout: int = 40):
    q = f"[out:json][timeout:{timeout - 5}];node({south},{west},{north},{east})[place~'village|town|hamlet|suburb|city'][name];out;"
    for host in ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"):
        try:
            d = get_json(host, timeout=timeout, data=("data=" + urllib.parse.quote(q)).encode())
            return [dict(name=e["tags"]["name"], lat=e["lat"], lon=e["lon"]) for e in d.get("elements", [])]
        except Exception:
            continue
    return []
