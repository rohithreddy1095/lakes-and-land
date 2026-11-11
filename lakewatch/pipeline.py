"""End-to-end analysis for one lake: point in, report folder out."""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import hashlib
import math
import os
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from . import analyse as an
from . import sources as src
from .geo import (Grid, block_fraction, grid_around, km_between, mask_to_geojson, mask_to_path, refine,
                  resample_nearest)


class LakeNotFound(Exception):
    pass


class NotEnoughData(Exception):
    pass


@dataclass
class Config:
    lat: float
    lon: float
    name: str = "Lake"
    out: str = "reports/lake"
    units: str = "acres"                      # "acres" or "ha" (display only; data is stored in hectares)
    peak_months: tuple = (9, 10, 11, 2, 3)    # months when the lake is fullest and fields are not flooded
    year_start: int = 7                       # first month of the hydrological year
    dry_months: tuple = (3, 4, 5, 6)          # months when a harvested field is bare
    rain_months: tuple = (6, 7, 8, 9, 10)     # rainy season, for the rainfall line
    start_year: int = 1984
    search_km: float = 10.0                   # how far around the point to look for the lake's full extent
    neighbours_km: float = 15.0               # radius of the neighbouring-tank scan; 0 turns it off
    radar_months: int = 26
    hires: bool = True                        # look up Esri Wayback capture dates for the report's photo layer
    workers: int = 10
    cache: str | None = None
    today: str | None = None                  # YYYY-MM-DD, for reproducible runs
    extra: dict = field(default_factory=dict)


def _log_default(stage, msg, frac=None):
    print(f"[{stage}] {msg}", flush=True)


def _cache_dir(cfg: Config, G: Grid) -> str:
    root = cfg.cache or os.path.join(os.path.expanduser("~"), ".cache", "lakewatch")
    key = hashlib.sha1(f"{G.crs}{G.xmin}{G.ymax}{G.w}{G.h}".encode()).hexdigest()[:12]
    d = os.path.join(root, key)
    os.makedirs(d, exist_ok=True)
    return d


def _cached(path: str, fn):
    if os.path.exists(path):
        try:
            return np.load(path)
        except Exception:
            pass
    a = fn()
    tmp = path + ".tmp.npy"
    np.save(tmp, a)
    os.replace(tmp, path)
    return a


def _bbox_px(mask: np.ndarray, pad: int):
    rr, cc = np.where(mask)
    return (max(0, rr.min() - pad), min(mask.shape[0], rr.max() + 1 + pad),
            max(0, cc.min() - pad), min(mask.shape[1], cc.max() + 1 + pad))


def run(cfg: Config, log=_log_default) -> dict:
    today = dt.date.fromisoformat(cfg.today) if cfg.today else dt.date.today()
    R: dict = {"cfg": cfg, "today": today.isoformat(), "notes": []}

    # ------------------------------------------------------------------ 1. find the lake
    log("locate", "Finding the water body at that point", 0.01)
    G0 = grid_around(cfg.lon, cfg.lat, cfg.search_km * 1000, 30.0)
    ext = src.jrc_layer("extent", cfg.lon, cfg.lat, G0) == 1
    occ = src.jrc_layer("occurrence", cfg.lon, cfg.lat, G0)
    r, c = (int(v) for v in G0.ll2px(cfg.lon, cfg.lat))
    found = an.find_lake(ext, occ, r, c)
    if found is None:
        raise LakeNotFound("No lake, tank or reservoir was found within about 500 m of that point. "
                           "Click on the open water and try again.")
    comp0, core0, clipped = found
    if clipped:
        R["notes"].append("clipped")
    r0, r1, c0, c1 = _bbox_px(comp0, 20)
    G = G0.crop(r0, r1, c0, c1)
    comp, core = comp0[r0:r1, c0:c1], core0[r0:r1, c0:c1]
    limit = ndi.binary_dilation(comp, iterations=15)
    near = ndi.binary_dilation(comp, iterations=4)
    cache = _cache_dir(cfg, G)
    log("locate", f"Lake found: {comp.sum() * G.px_ha:.0f} ha maximum extent on record", 0.03)

    # ------------------------------------------------------------------ 2. Landsat record
    items = src.search_landsat(G.bounds_ll(), f"{cfg.start_year}-01-01", today.isoformat())
    log("landsat", f"Reading {len(items)} Landsat scenes ({cfg.start_year} to today)", 0.04)
    os.makedirs(f"{cache}/landsat", exist_ok=True)

    def load_ls(it):
        try:
            return it, _cached(f"{cache}/landsat/{it.id}.npy", lambda: src.landsat_mndwi(it, G))
        except Exception:
            return it, None

    scenes = []
    with cf.ThreadPoolExecutor(cfg.workers) as ex:
        for n, (it, mn) in enumerate(ex.map(load_ls, items)):
            if n % 40 == 0:
                log("landsat", f"Landsat scene {n} of {len(items)}", 0.04 + 0.36 * n / max(1, len(items)))
            if mn is None or (mn[near] != an.NODATA).mean() < 0.5:
                continue
            date = it.datetime.strftime("%Y-%m-%d")
            sensor = src.LANDSAT_SENSOR[it.properties["platform"]]
            gappy = sensor == "ETM+" and date >= "2003-06-01"
            if gappy:
                mn = an.fill_small_gaps(mn, 3)
            scenes.append(dict(item=it, date=date, sensor=sensor, gappy=gappy, mn=mn, month=int(date[5:7]),
                               hy=an.hydro_year(date, cfg.year_start)))
    if len(scenes) < 30:
        raise NotEnoughData(f"Only {len(scenes)} usable Landsat scenes cover this lake; at least 30 are needed.")

    # lake zone: where water shows up in the peak season, connected to the lake's core
    peak_sc = [s for s in scenes if s["month"] in cfg.peak_months]
    wsum = sum(((s["mn"] > 0) for s in peak_sc)).astype(float)
    ksum = sum(((s["mn"] != an.NODATA) for s in peak_sc)).astype(float)
    freq = wsum / np.maximum(ksum, 1)
    zone = an.connected_to((((freq >= 0.08) & (ksum >= 5)) | comp) & limit, core)
    zone = ndi.binary_dilation(zone, iterations=2) & limit
    for s in scenes:
        known = s["mn"] != an.NODATA
        s["valid"] = float(known[zone].mean())
        s["lake"] = an.connected_to((s["mn"] > 0) & zone, core)
        s["area"] = int(s["lake"].sum())
    peaks = an.pick_peaks(scenes, cfg.peak_months)
    years = sorted(peaks)
    current_hy = an.hydro_year(today.isoformat(), cfg.year_start)
    ref, now = an.choose_eras(years, current_hy)
    if len(ref) < 3 or len(now) < 3 or (now and ref and now[0] - ref[-1] < 5):
        raise NotEnoughData("The clear-image record for this lake is too short to compare two periods "
                            f"(peak years found: {years}).")
    wet = {y: (peaks[y]["mn"] > 0) & zone for y in years}
    known = {y: peaks[y]["mn"] != an.NODATA for y in years}
    M = an.era_masks(wet, known, ref, now, core)
    if not M["full_ref"].any():
        raise NotEnoughData("The lake held no water at peak in the reference years.")
    lost = M["never"] | M["wettest"]
    ha = lambda m: round(float(m.sum()) * G.px_ha, 1)  # noqa: E731
    log("landsat", f"{len(scenes)} usable scenes, {len(years)} peak years; reference {ref[0]}–{ref[-1]}, "
                   f"now {now[0]}–{now[-1]}", 0.40)
    R.update(G=G, zone=zone, core=core, peaks=peaks, years=years, ref=ref, now=now, masks=M,
             n_scenes=len(scenes), current_hy=current_hy)

    # ------------------------------------------------------------------ 3. rainfall
    rain = {}
    try:
        mon = src.rain_monthly(cfg.lon, cfg.lat, cfg.start_year, today.year)
        for y in range(max(1981, cfg.start_year), today.year + 1):
            vals = [mon.get(f"{y}-{m:02d}") for m in cfg.rain_months]
            if all(v is not None for v in vals) and (y < today.year or max(cfg.rain_months) < today.month):
                rain[y] = round(sum(vals))
    except Exception as e:
        R["notes"].append("no_rain")
        log("rain", f"Rainfall unavailable ({type(e).__name__})", 0.41)
    R["rain"] = rain

    # ------------------------------------------------------------------ 4. Sentinel-2: what is on the lakebed now
    fr0, fr1, fc0, fc1 = _bbox_px(zone | M["full_ref"] | M["still"], 8)
    GF = G.crop(fr0, fr1, fc0, fc1)                       # frame / view window on the 30 m grid
    s2res = 10.0 if GF.w * GF.h * 9 <= 3.0e6 else 20.0
    S = refine(GF, s2res) if s2res == 10.0 else Grid(GF.crs, 20.0, GF.xmin, GF.ymax,
                                                     int(GF.w * 30 // 20), int(GF.h * 30 // 20))
    up = lambda m: resample_nearest(m, G, S, False)       # noqa: E731
    zoneS, coreS, fullS = up(zone), up(core), up(M["full_ref"])
    lostS, neverS = up(lost), up(M["never"])
    s2_start = max(2017, min(2018, current_hy - 3))
    s2_items = src.search_s2(S.bounds_ll(), f"{s2_start}-{cfg.year_start:02d}-01", today.isoformat())
    log("sentinel2", f"Reading {len(s2_items)} Sentinel-2 scenes at {s2res:.0f} m", 0.42)
    os.makedirs(f"{cache}/s2", exist_ok=True)

    def load_s2(it):
        try:
            a = _cached(f"{cache}/s2/{it.id}_{int(s2res)}.npy", lambda: np.stack(src.s2_indices(it, S)))
            return it, a
        except Exception:
            return it, None

    by_hy: dict = {}
    for it in s2_items:
        by_hy.setdefault(an.hydro_year(it.datetime.strftime("%Y-%m-%d"), cfg.year_start), []).append(it)
    cls_years, cls = [], []
    recent: dict = {}          # date -> (water, valid) for the last ~3.4 years
    latest = None
    done = 0
    keep_from = (today - dt.timedelta(days=int(365 * 3.4))).isoformat()
    for hy in sorted(by_hy):
        day: dict = {}
        with cf.ThreadPoolExecutor(cfg.workers) as ex:
            for it, a in ex.map(load_s2, by_hy[hy]):
                done += 1
                if a is None:
                    continue
                v = float((a[0][zoneS] != an.NODATA).mean())
                d = it.datetime.strftime("%Y-%m-%d")
                if v >= 0.5 and (d not in day or v > day[d][0]):
                    day[d] = (v, a, it)
        log("sentinel2", f"Sentinel-2 year {hy}: {len(day)} clear dates", 0.42 + 0.26 * done / max(1, len(s2_items)))
        if not day:
            continue
        dates = sorted(day)
        for d in dates:
            v, a, it = day[d]
            if d >= keep_from:
                recent[d] = (np.packbits(a[0] > 0), np.packbits(a[0] != an.NODATA))
            if v >= 0.9:
                latest = dict(date=d, mn=a[0], nv=a[1], item=it)
        if hy < current_hy and len(dates) >= 6:
            mn = np.stack([day[d][1][0] for d in dates])
            nv = np.stack([day[d][1][1] for d in dates])
            months = np.array([int(d[5:7]) for d in dates])
            cls.append(an.classify_cover_year(mn, nv, months, cfg.dry_months))
            cls_years.append(hy)
    S_ha = S.px_ha
    haS = lambda m: round(float(m.sum()) * S_ha, 1)       # noqa: E731
    if cls:
        farmed, weeds, need = an.summarise_cover(np.stack(cls))
    else:
        farmed = weeds = np.zeros((S.h, S.w), bool)
        need = 0
        R["notes"].append("no_cover")
    farmed &= fullS
    weeds &= fullS
    R.update(S=S, GF=GF, farmed=farmed, weeds=weeds, cover_years=cls_years, cover_need=need, latest=latest)

    areas = dict(full_ref=ha(M["full_ref"]), still=ha(M["still"] & M["full_ref"]), lost=ha(lost),
                 never=ha(M["never"]), wettest=ha(M["wettest"]), gained=ha(M["gained"]),
                 farmed_lost=haS(farmed & lostS), weeds_lost=haS(weeds & lostS), farmed_total=haS(farmed))
    R["ha"] = areas

    # ------------------------------------------------------------------ 5. this month's bulletin
    bul = None
    if latest:
        w_now = latest["mn"] > 0
        v_now = latest["mn"] != an.NODATA
        veg = v_now & ~w_now & (latest["nv"] >= 50)
        bul = dict(date=latest["date"], water=haS(w_now & fullS), crop=haS(veg & farmed), veg=haS(veg & fullS),
                   watch=0.0, patches=[])
        d0 = dt.date.fromisoformat(latest["date"])
        prior = []
        for back in (1, 2, 3):
            mid = d0.replace(year=d0.year - back) if not (d0.month == 2 and d0.day == 29) else d0.replace(year=d0.year - back, day=28)
            sel = [d for d in recent if abs((dt.date.fromisoformat(d) - mid).days) <= 25]
            if sel:
                n = S.h * S.w
                wany = np.zeros(n, bool)
                for d in sel:
                    wany |= np.unpackbits(recent[d][0])[:n].astype(bool) & np.unpackbits(recent[d][1])[:n].astype(bool)
                prior.append(wany.reshape(S.h, S.w))
        if len(prior) >= 2:
            watch = fullS & v_now & ~w_now & np.logical_and.reduce(prior)
            ps = an.patches(watch, min_px=int(math.ceil(0.4 / S_ha)), join=1)
            bul["watch"] = round(sum(float(p.sum()) for p in ps) * S_ha, 1)
            bul["years"] = len(prior)
            for p in ps[:3]:
                rr, cc = ndi.center_of_mass(p)
                lon, lat = S.px2ll(rr, cc)
                bul["patches"].append([round(float(p.sum()) * S_ha, 1), round(float(lat), 5), round(float(lon), 5)])
    R["bulletin"] = bul

    # ------------------------------------------------------------------ 6. radar (sees through cloud)
    radar = []
    try:
        start = (today - dt.timedelta(days=int(cfg.radar_months * 30.4))).isoformat()
        s1 = src.search_s1(S.bounds_ll(), start, today.isoformat())
        log("radar", f"Reading {len(s1)} Sentinel-1 radar passes", 0.69)
        zr = ndi.binary_dilation(zoneS, iterations=3)

        def load_s1(it):
            try:
                db = _cached(f"{cache}/s1_{it.id}_{int(s2res)}.npy", lambda: src.s1_vv_db(it, S).astype(np.float32))
                if np.isnan(db[zr]).mean() > 0.02:
                    return None
                db = ndi.median_filter(np.nan_to_num(db, nan=0.0), 5)
                t = an.otsu(db[zr], -20.0, -14.0)
                lake = an.connected_to((db < t) & zr, coreS)
                return it.datetime.strftime("%Y-%m-%d"), round(float(lake.sum()) * S_ha, 1)
            except Exception:
                return None

        with cf.ThreadPoolExecutor(cfg.workers) as ex:
            rows = [x for x in ex.map(load_s1, s1) if x]
        cap = 1.25 * max(areas["full_ref"], ha(M["still"]))
        best: dict = {}
        for d, a in rows:
            if 0 < a <= cap:
                best[d] = max(best.get(d, 0), a)
        radar = [dict(d=d, ha=a) for d, a in sorted(best.items())]
    except Exception as e:
        R["notes"].append("no_radar")
        log("radar", f"Radar unavailable ({type(e).__name__})", 0.74)
    R["radar"] = radar

    # ------------------------------------------------------------------ 7. hotspots and alert replay
    log("alerts", "Replaying alerts year by year", 0.75)
    zoom = min(480 / GF.w, 760 / GF.h)
    fw, fh = int(round(GF.w * zoom)), int(round(GF.h * zoom))
    crop = lambda m: m[fr0:fr1, fc0:fc1]                  # noqa: E731
    cr, cc_ = ndi.center_of_mass(M["full_ref"])
    hot = []
    used: dict = {}
    for q in an.patches(lost, min_px=int(math.ceil(0.6 / G.px_ha)), join=2)[:5]:
        rr, cc = ndi.center_of_mass(q)
        lon, lat = G.px2ll(rr, cc)
        d = an.compass(cr - rr, cc - cc_)
        used[d] = used.get(d, 0) + 1
        b0, b1, b2, b3 = _bbox_px(q, 6)
        w_, n_ = G.px2ll(b0, b2)
        e_, s_ = G.px2ll(b1 - 1, b3 - 1)
        qS = up(q)
        share = {}
        for y in years:
            if known[y][q].mean() >= 0.8:
                share[y] = round(float((wet[y] & q).sum() / q.sum()), 2)
        bef = [y for y in share if y <= ref[-1]]
        aft = [y for y in share if y > ref[-1]]
        hot.append(dict(key=f"h{len(hot) + 1}", dir=d, n=used[d], lat=round(float(lat), 5), lon=round(float(lon), 5),
                        bounds=[[round(float(s_), 6), round(float(w_), 6)], [round(float(n_), 6), round(float(e_), 6)]],
                        lost=ha(q), never=ha(q & M["never"]), farmed=haS(farmed & qS), weeds=haS(weeds & qS),
                        share=share, before=[sum(share[y] >= 0.5 for y in bef), len(bef)],
                        after=[sum(share[y] >= 0.5 for y in aft), len(aft)]))
    R["hotspots"] = hot

    alert = an.replay_alerts(wet, known, [y for y in years if y < current_hy], M["full_ref"])
    farm30 = block_fraction(farmed, S, G)
    alerts = []
    for y in sorted(set(np.unique(alert)) - {0}):
        for q in an.patches(alert == y, min_px=int(math.ceil(0.4 / G.px_ha)), join=1):
            rr, cc = ndi.center_of_mass(q)
            lon, lat = G.px2ll(rr, cc)
            still_dry = float((q & M["never"]).sum() / q.sum())
            fshare = float(farm30[q].mean())
            st = "farm" if fshare >= 0.5 else "dry" if still_dry >= 0.5 else "back"
            where = None
            if hot:
                dists = [km_between(lat, lon, h["lat"], h["lon"]) for h in hot]
                if min(dists) < 0.75:
                    where = hot[int(np.argmin(dists))]["key"]
            alerts.append(dict(y=int(y), ha=ha(q), lat=round(float(lat), 5), lon=round(float(lon), 5), where=where,
                               st=st, path=mask_to_path(crop(q), zoom)))
    R["alerts"] = alerts
    R["frame"] = dict(zoom=zoom, fw=fw, fh=fh, win=(fr0, fr1, fc0, fc1))

    # ------------------------------------------------------------------ 8. neighbouring tanks
    R["tanks"] = None
    if cfg.neighbours_km > 0:
        try:
            R["tanks"] = _neighbours(cfg, R, log)
        except Exception as e:
            R["notes"].append("no_tanks")
            log("tanks", f"Neighbour scan skipped ({type(e).__name__}: {e})", 0.9)

    # ------------------------------------------------------------------ 9. high-resolution photo dates
    R["hires"] = []
    if cfg.hires:
        try:
            log("photos", "Looking up high-resolution photo dates", 0.91)
            lon_c, lat_c = G.px2ll(cr, cc_)
            R["hires"] = src.wayback_captures(float(lon_c), float(lat_c))
        except Exception:
            R["notes"].append("no_hires")
    return R


def _neighbours(cfg: Config, R: dict, log) -> list:
    """Every water body within neighbours_km: area that fills in the reference era vs now."""
    G, ref, now = R["G"], R["ref"], R["now"]
    W = grid_around(cfg.lon, cfg.lat, cfg.neighbours_km * 1000, 30.0, crs=G.crs)
    items = []
    for a, b in ((ref[0], ref[-1]), (now[0], now[-1])):
        items += src.search_landsat(W.bounds_ll(), f"{a}-{cfg.year_start:02d}-01",
                                    f"{b + 1}-{cfg.year_start:02d}-01", cloud_lt=60)
    items = [i for i in items if i.datetime.month in cfg.peak_months and i.properties["platform"] != "landsat-7"]
    log("tanks", f"Scanning neighbouring tanks: {len(items)} scenes over {2 * cfg.neighbours_km:.0f} km", 0.78)
    wetc: dict = {}
    knc: dict = {}

    def load(it):
        try:
            return it, src.landsat_mndwi(it, W)
        except Exception:
            return it, None

    with cf.ThreadPoolExecutor(cfg.workers) as ex:
        for n, (it, mn) in enumerate(ex.map(load, items)):
            if n % 25 == 0:
                log("tanks", f"Neighbour scan: scene {n} of {len(items)}", 0.78 + 0.12 * n / max(1, len(items)))
            if mn is None:
                continue
            k = mn != an.NODATA
            if k.mean() < 0.3:
                continue
            hy = an.hydro_year(it.datetime.strftime("%Y-%m-%d"), cfg.year_start)
            wetc[hy] = wetc.get(hy, 0) + ((mn > 0) & k).astype(np.int16)
            knc[hy] = knc.get(hy, 0) + k.astype(np.int16)

    def era(yrs):
        k = np.zeros((W.h, W.w), int)
        n = np.zeros((W.h, W.w), int)
        for y in yrs:
            if y in knc:
                kn = knc[y] > 0
                k += kn
                n += kn & (wetc[y] > 0) & (wetc[y] >= 0.5 * knc[y])
        return (k >= max(2, math.ceil(len(yrs) / 2))) & (n >= 0.6 * k)

    E, N = era(ref), era(now)
    lab, cnt = ndi.label(E | N, structure=an.EIGHT)
    coreW = resample_nearest(R["core"], G, W, False)
    me_ids = set(np.unique(lab[coreW])) - {0}
    try:
        w_, s_, e_, n_ = W.bounds_ll()
        places = src.osm_places(w_, s_, e_, n_)
    except Exception:
        places = []
    rows = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        m = lab[sl] == i
        a, b = float((m & E[sl]).sum()) * W.px_ha, float((m & N[sl]).sum()) * W.px_ha
        me = i in me_ids
        if max(a, b) < 2.0 and not me:
            continue
        edge = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == W.h or sl[1].stop == W.w
        fill = m.sum() / m.size
        if (edge or fill < 0.10) and not me:      # cut by the scan edge, or a river / canal
            continue
        rr, cc = ndi.center_of_mass(m)
        lon, lat = W.px2ll(rr + sl[0].start, cc + sl[1].start)
        near = None
        if places:
            d, p = min(((km_between(lat, lon, p["lat"], p["lon"]), p) for p in places), key=lambda t: t[0])
            if d <= 3.0:
                near = p["name"]
        rows.append(dict(n=near, a=round(a, 1), b=round(b, 1), lat=round(float(lat), 5), lon=round(float(lon), 5),
                         me=bool(me)))
    rows.sort(key=lambda r: r["b"] - r["a"])
    return rows


def geojson_layers(R: dict) -> dict:
    G, S, M = R["G"], R["S"], R["masks"]
    return dict(full_ref=mask_to_geojson(M["full_ref"], G), still=mask_to_geojson(M["still"], G),
                never=mask_to_geojson(M["never"], G), wettest=mask_to_geojson(M["wettest"], G),
                farmed=mask_to_geojson(R["farmed"], S), weeds=mask_to_geojson(R["weeds"], S))
