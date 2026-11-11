"""Turn pipeline results into a self-contained report folder (index.html + data.json + images)."""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import shutil

import numpy as np
from PIL import Image

from . import __version__
from . import sources as src
from .geo import mask_to_path, mercator_grid
from .pipeline import geojson_layers

HERE = os.path.dirname(os.path.abspath(__file__))


def _stretch(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros(rgb.shape, np.float32)
    for k in range(3):
        ch = rgb[..., k]
        v = ch[np.isfinite(ch) & (ch > -0.19)]
        if v.size < 10:
            continue
        lo, hi = np.percentile(v, 1.5), min(np.percentile(v, 98.5), 0.30)   # cap so clouds do not darken the land
        if hi <= lo:
            hi = np.percentile(v, 98.5)
        out[..., k] = np.clip((ch - lo) / (hi - lo + 1e-9), 0, 1)
    return (out ** 0.8 * 255).astype(np.uint8)


def build(R: dict, log=lambda *a, **k: None) -> str:
    cfg = R["cfg"]
    out = cfg.out
    os.makedirs(f"{out}/assets/frames", exist_ok=True)
    G, GF, S, M = R["G"], R["GF"], R["S"], R["masks"]
    fr0, fr1, fc0, fc1 = R["frame"]["win"]
    zoom, fw, fh = R["frame"]["zoom"], R["frame"]["fw"], R["frame"]["fh"]
    crop = lambda m: m[fr0:fr1, fc0:fc1]  # noqa: E731
    peaks, years = R["peaks"], R["years"]

    # ---- one true-colour frame per year, at the lake's fullest clear day
    log("report", f"Rendering {len(years)} yearly frames", 0.93)

    def frame(y):
        s = peaks[y]
        try:
            rgb = src.landsat_rgb(s["item"], GF)
            Image.fromarray(_stretch(rgb)).resize((fw, fh), Image.BICUBIC).save(
                f"{out}/assets/frames/f{y}.jpg", quality=80, optimize=True)
            ok = True
        except Exception:
            ok = False
        return dict(hy=y, date=s["date"], sensor=s["sensor"], ha=round(s["area"] * G.px_ha, 1), img=ok,
                    path=mask_to_path(crop(s["lake"]), zoom))

    with cf.ThreadPoolExecutor(cfg.workers) as ex:
        frames = list(ex.map(frame, years))

    # ---- base image for the map: latest clear Sentinel-2, warped to Web Mercator
    base = None
    if R.get("latest"):
        try:
            log("report", "Rendering the base image", 0.97)
            w, s, e, n = G.bounds_ll()              # whole search window, for context around the lake
            k = math.cos(math.radians((s + n) / 2))
            res = S.res / k / 2                     # 2x oversampled so the 10 m pixels look smooth
            while max((e - w) * 111320, (n - s) * 110540 / k) / res > 2400:
                res *= 1.25
            MG = mercator_grid(w, s, e, n, res)
            Image.fromarray(_stretch(src.s2_rgb(R["latest"]["item"], MG))).save(
                f"{out}/assets/base.jpg", quality=82, optimize=True, progressive=True)
            base = dict(img="assets/base.jpg", date=R["latest"]["date"], bounds=[[s, w], [n, e]])
        except Exception:
            base = None
    w, s, e, n = GF.bounds_ll()

    # ---- then / now comparison
    ref, now, rain = R["ref"], R["now"], R["rain"]
    area = {f["hy"]: f["ha"] for f in frames}
    then_y = max(ref, key=lambda y: area[y])
    rained = [y for y in now if y in rain]
    now_y = max(rained, key=lambda y: rain[y]) if rained else max(now, key=lambda y: area[y])
    compare = dict(then=dict(hy=then_y, ha=area[then_y], rain=rain.get(then_y)),
                   now=dict(hy=now_y, ha=area[now_y], rain=rain.get(now_y)))

    alerts = R["alerts"]
    cnt = {k: sum(a["st"] == k for a in alerts) for k in ("farm", "dry", "back")}
    data = dict(
        meta=dict(name=cfg.name, lat=cfg.lat, lon=cfg.lon, generated=R["today"], units=cfg.units,
                  ref=[ref[0], ref[-1]], ref_n=len(ref), now=[now[0], now[-1]], now_n=len(now),
                  first=years[0], last=years[-1], n_scenes=R["n_scenes"], peak_months=list(cfg.peak_months),
                  rain_months=list(cfg.rain_months), dry_months=list(cfg.dry_months), year_start=cfg.year_start,
                  s2_res=S.res, cover_years=R["cover_years"], cover_need=R["cover_need"],
                  neighbours_km=cfg.neighbours_km, notes=R["notes"], version=__version__),
        view=[[s, w], [n, e]], base=base, hires=R["hires"], tiles=src.WAYBACK_TILES,
        layers=geojson_layers(R), ha=R["ha"],
        frames=frames, fw=fw, fh=fh, ref_path=mask_to_path(crop(M["full_ref"]), zoom),
        rain={str(k): v for k, v in rain.items()}, compare=compare,
        alerts=alerts, alertsum=dict(n=len(alerts), ha=round(sum(a["ha"] for a in alerts), 1), cnt=cnt),
        hotspots=R["hotspots"], bulletin=R["bulletin"], radar=R["radar"], tanks=R["tanks"],
    )
    with open(f"{out}/data.json", "w") as f:
        json.dump(data, f, separators=(",", ":"))
    with open(f"{out}/data.js", "w") as f:   # lets index.html work when opened straight from disk
        f.write("window.LAKE_DATA=" + json.dumps(data, separators=(",", ":")) + ";")
    shutil.copy(f"{HERE}/template/index.html", f"{out}/index.html")
    shutil.copytree(f"{HERE}/template/vendor", f"{out}/vendor", dirs_exist_ok=True)
    log("report", f"Report written to {out}/index.html", 1.0)
    return out
