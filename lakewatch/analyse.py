"""Pure-numpy analysis steps. No network access here, so these are easy to test."""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

EIGHT = np.ones((3, 3), bool)
NODATA = -128


def hydro_year(date: str, start_month: int) -> int:
    y, m = int(date[:4]), int(date[5:7])
    return y if m >= start_month else y - 1


def connected_to(mask: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Parts of mask that touch core (8-connected)."""
    lab, _ = ndi.label(mask, structure=EIGHT)
    ids = np.unique(lab[core])
    ids = ids[ids > 0]
    return np.isin(lab, ids) if len(ids) else np.zeros_like(mask, bool)


def fill_small_gaps(a: np.ndarray, max_px: int = 3) -> np.ndarray:
    """Fill no-data pixels (Landsat 7 scan-line gaps) from the nearest valid pixel within max_px."""
    bad = a == NODATA
    if not bad.any() or bad.all():
        return a
    dist, idx = ndi.distance_transform_edt(bad, return_indices=True)
    out = a.copy()
    sel = bad & (dist <= max_px)
    out[sel] = a[idx[0][sel], idx[1][sel]]
    return out


def find_lake(extent: np.ndarray, occurrence: np.ndarray, r: int, c: int, snap_px: int = 17):
    """Locate the water body under (or within ~500 m of) a picked pixel on a JRC max-extent raster.

    Returns (component mask, core mask, touches_edge) or None when there is no water nearby.
    """
    lab, _ = ndi.label(extent, structure=EIGHT)
    h, w = extent.shape
    r0, r1, c0, c1 = max(0, r - snap_px), min(h, r + snap_px + 1), max(0, c - snap_px), min(w, c + snap_px + 1)
    win = lab[r0:r1, c0:c1]
    ids = np.unique(win)
    ids = ids[ids > 0]
    if not len(ids):
        return None
    rr, cc = np.mgrid[r0:r1, c0:c1]
    d = np.hypot(rr - r, cc - c)
    best = min(ids, key=lambda i: (d[win == i].min(), -(lab == i).sum()))
    comp = lab == best
    occ = occurrence.astype(float)
    occ[occurrence > 100] = 0
    thr = max(1.0, np.percentile(occ[comp], 70), 0.5 * occ[comp].max())   # the part that is nearly always water
    core = comp & (occ >= thr)
    if not core.any():
        core = comp
    edge = bool(comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any())
    return comp, core, edge


def pick_peaks(scenes: list[dict], peak_months, min_valid: float = 0.85) -> dict:
    """For each hydrological year pick the fullest clear scene in the peak season.

    scenes: dicts with hy, month, area (px), valid (share of lake zone observed), gappy (bool).
    Returns {hy: scene}. Among scenes within 3% of the year's maximum the clearest wins.
    """
    by = {}
    for s in scenes:
        if s["month"] in peak_months and s["valid"] >= min_valid:
            by.setdefault(s["hy"], []).append(s)
    out = {}
    for hy, rows in by.items():
        top = max(r["area"] for r in rows)
        cands = [r for r in rows if r["area"] >= 0.97 * top]
        out[hy] = max(cands, key=lambda r: (not r.get("gappy", False), round(r["valid"], 2), r["area"]))
    return out


def choose_eras(years: list[int], current_hy: int, n: int = 6):
    """Reference era = first n peak years on record; 'now' = last n complete years."""
    done = [y for y in years if y < current_hy]
    ref = done[:n]
    now = [y for y in done[-n:] if y not in ref]
    return ref, now


def era_masks(wet: dict, known: dict, ref: list[int], now: list[int], core: np.ndarray):
    """Shoreline that filled in at least half of the reference years, and what became of it."""
    def count(yrs):
        return sum(wet[y] & known[y] for y in yrs), sum(known[y].astype(int) for y in yrs)
    wr, kr = count(ref)
    wn, kn = count(now)
    full_ref = connected_to((wr >= np.ceil(np.maximum(kr, 1) / 2)) & (kr >= max(2, len(ref) // 2)), core)
    seen_now = kn >= max(2, len(now) // 2)
    still_all = (wn >= np.ceil(np.maximum(kn, 1) / 2)) & seen_now
    still = connected_to(still_all, core) if (still_all & core).any() else still_all & full_ref
    never = full_ref & seen_now & (wn == 0)
    wettest = full_ref & seen_now & ~still & ~never
    gained = still & ~full_ref
    return dict(full_ref=full_ref, still=still, never=never, wettest=wettest, gained=gained)


def replay_alerts(wet: dict, known: dict, years: list[int], inside: np.ndarray, dry_years: int = 3):
    """Year in which each pixel would have raised an alert: dry at peak for dry_years observed
    years in a row after having been under water at least once. 0 = no alert."""
    shape = inside.shape
    alert = np.zeros(shape, np.int16)
    streak = np.zeros(shape, np.int16)
    ever = np.zeros(shape, bool)
    for y in years:
        k, w = known[y], wet[y] & known[y]
        streak = np.where(w, 0, np.where(k, streak + 1, streak))
        fire = inside & k & (streak == dry_years) & (alert == 0) & ever
        alert[fire] = y
        ever |= w
    return alert


def patches(mask: np.ndarray, min_px: int, join: int = 1):
    """Connected patches of mask (after bridging gaps of `join` px), largest first."""
    grow = ndi.binary_dilation(mask, iterations=join) if join else mask
    lab, n = ndi.label(grow, structure=EIGHT)
    out = []
    for i in range(1, n + 1):
        q = (lab == i) & mask
        if q.sum() >= min_px:
            out.append(q)
    return sorted(out, key=lambda q: -q.sum())


COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def compass(dy_north: float, dx_east: float) -> str:
    ang = (math.degrees(math.atan2(dx_east, dy_north)) + 360) % 360
    return COMPASS[int((ang + 22.5) // 45) % 8]


def classify_cover_year(mn: np.ndarray, nv: np.ndarray, months: np.ndarray, dry_months) -> np.ndarray:
    """Land cover for one hydrological year from a Sentinel-2 stack (int8 x100, -128 no data).

    1 water     : water in >= 60% of observations
    2 crop      : turns green (NDVI >= 0.55) and is later bare or harvested (NDVI <= 0.30 in the dry season)
    3 thicket   : turns green and never goes bare (weeds, reeds, shrubs, trees)
    4 other     : wet mud, sparse plants, mixed
    0 no data   : fewer than 4 clear observations
    """
    valid = mn != NODATA
    n = valid.sum(0)
    water = (mn > 0) & valid
    wf = water.sum(0) / np.maximum(n, 1)
    land = valid & (mn < 0)
    gmax = np.where(land, nv, -127).max(0)
    dry = np.isin(months, list(dry_months))
    if dry.any():
        dland = valid[dry] & (mn[dry] < -5)
        gmin = np.where(dland, nv[dry], 127).min(0)
    else:
        gmin = np.full(n.shape, 127)
    c = np.full(n.shape, 4, np.int8)
    c[(gmax >= 50) & (gmin > 30)] = 3
    c[(gmax >= 55) & (gmin <= 30)] = 2
    c[wf >= 0.6] = 1
    c[n < 4] = 0
    return c


def summarise_cover(cls: np.ndarray):
    """cls: (years, H, W). farmed = crop cycle in >= 3 years (or half, with a short record)."""
    ny = cls.shape[0]
    need = 3 if ny >= 5 else max(1, math.ceil(ny / 2))
    crop_n = (cls == 2).sum(0)
    thick_n = (cls == 3).sum(0)
    water_n = (cls == 1).sum(0)
    farmed = crop_n >= need
    weeds = ~farmed & (thick_n >= math.ceil(ny / 2)) & (water_n < math.ceil(ny / 2))
    return farmed, weeds, need


def otsu(values: np.ndarray, lo: float, hi: float) -> float:
    from skimage.filters import threshold_otsu
    v = values[np.isfinite(values)]
    if v.size < 50:
        return (lo + hi) / 2
    return float(np.clip(threshold_otsu(v), lo, hi))
