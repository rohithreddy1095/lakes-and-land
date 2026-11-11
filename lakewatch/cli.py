"""Command line: `lakewatch run` for one lake, `lakewatch serve` for the map picker."""
from __future__ import annotations

import argparse
import re
import sys

from .pipeline import Config, LakeNotFound, NotEnoughData


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "lake"


def _months(s: str):
    return tuple(int(x) for x in s.split(",") if x.strip())


def main(argv=None):
    p = argparse.ArgumentParser(prog="lakewatch", description="Satellite encroachment report for any lake or tank.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="analyse one lake and write a report folder")
    r.add_argument("--lat", type=float, required=True, help="latitude of a point on the lake's water")
    r.add_argument("--lon", type=float, required=True, help="longitude of that point")
    r.add_argument("--name", default="Lake", help="name shown on the report")
    r.add_argument("--out", help="output folder (default: reports/<name>)")
    r.add_argument("--units", choices=["acres", "ha"], default="acres")
    r.add_argument("--peak-months", type=_months, default=(9, 10, 11, 2, 3),
                   help="months when the lake is fullest and fields are not flooded (default 9,10,11,2,3)")
    r.add_argument("--dry-months", type=_months, default=(3, 4, 5, 6), help="months when harvested fields are bare")
    r.add_argument("--rain-months", type=_months, default=(6, 7, 8, 9, 10), help="rainy season for the rain line")
    r.add_argument("--year-start", type=int, default=7, help="first month of the hydrological year (default 7)")
    r.add_argument("--search-km", type=float, default=10, help="half-width of the window searched for the lake")
    r.add_argument("--neighbours-km", type=float, default=15, help="radius of the neighbour scan; 0 to skip")
    r.add_argument("--no-photos", action="store_true", help="skip the Esri Wayback photo-date lookup")
    r.add_argument("--workers", type=int, default=10)
    r.add_argument("--cache", help="cache folder (default ~/.cache/lakewatch)")

    s = sub.add_parser("serve", help="open the map picker in a local web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reports", default="reports", help="folder where reports are written")

    a = p.parse_args(argv)
    if a.cmd == "serve":
        from .server import create_app
        app = create_app(a.reports)
        print(f"Lake picker: http://{a.host}:{a.port}")
        app.run(host=a.host, port=a.port, threaded=True)
        return 0

    from . import pipeline, report
    cfg = Config(lat=a.lat, lon=a.lon, name=a.name, out=a.out or f"reports/{slug(a.name)}", units=a.units,
                 peak_months=a.peak_months, dry_months=a.dry_months, rain_months=a.rain_months,
                 year_start=a.year_start, search_km=a.search_km, neighbours_km=a.neighbours_km,
                 hires=not a.no_photos, workers=a.workers, cache=a.cache)
    try:
        report.build(pipeline.run(cfg))
    except (LakeNotFound, NotEnoughData) as e:
        print(f"lakewatch: {e}", file=sys.stderr)
        return 2
    print(f"Open {cfg.out}/index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
