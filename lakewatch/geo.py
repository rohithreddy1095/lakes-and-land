"""Grids, reprojection helpers and raster-to-vector conversion."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.vrt import WarpedVRT


def utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


@dataclass
class Grid:
    """A north-up raster grid in a projected CRS."""
    crs: str
    res: float
    xmin: float
    ymax: float
    w: int
    h: int

    @property
    def transform(self) -> Affine:
        return from_origin(self.xmin, self.ymax, self.res, self.res)

    @property
    def px_ha(self) -> float:
        return self.res * self.res / 1e4

    def _to(self):
        return Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

    def _from(self):
        return Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    def ll2px(self, lon, lat):
        """lon/lat -> fractional (row, col); int() of each gives the pixel index."""
        x, y = self._to().transform(lon, lat)
        return (self.ymax - y) / self.res, (x - self.xmin) / self.res

    def px2ll(self, r, c):
        """pixel index (centre) -> lon, lat."""
        x = self.xmin + (np.asarray(c) + 0.5) * self.res
        y = self.ymax - (np.asarray(r) + 0.5) * self.res
        return self._from().transform(x, y)

    def bounds_ll(self):
        """west, south, east, north of the grid's corner points."""
        xs = [self.xmin, self.xmin + self.w * self.res]
        ys = [self.ymax - self.h * self.res, self.ymax]
        lon, lat = self._from().transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
        return min(lon), min(lat), max(lon), max(lat)

    def crop(self, r0: int, r1: int, c0: int, c1: int) -> "Grid":
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(self.h, r1), min(self.w, c1)
        return Grid(self.crs, self.res, self.xmin + c0 * self.res, self.ymax - r0 * self.res, c1 - c0, r1 - r0)

    def offset_in(self, parent: "Grid"):
        """(row, col) of this grid's origin inside a parent grid of the same resolution."""
        return int(round((parent.ymax - self.ymax) / self.res)), int(round((self.xmin - parent.xmin) / self.res))


def grid_around(lon: float, lat: float, half_m: float, res: float, crs: str | None = None, snap: float = 30.0) -> Grid:
    crs = crs or utm_epsg(lon, lat)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    off = snap / 2  # Landsat pixel corners sit on multiples of 30 m offset by 15 m
    xmin = np.floor((x - half_m - off) / snap) * snap + off
    ymax = np.ceil((y + half_m - off) / snap) * snap + off
    n = int(np.ceil(2 * half_m / snap) * snap / res)
    return Grid(crs, res, float(xmin), float(ymax), n, n)


def refine(G: Grid, res: float) -> Grid:
    """Same footprint at another resolution."""
    k = G.res / res
    return Grid(G.crs, res, G.xmin, G.ymax, int(round(G.w * k)), int(round(G.h * k)))


def mercator_grid(west, south, east, north, res: float) -> Grid:
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x0, y0 = t.transform(west, south)
    x1, y1 = t.transform(east, north)
    return Grid("EPSG:3857", res, x0, y1, int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res)))


def read_to_grid(href: str, G: Grid, resampling=Resampling.nearest, band: int = 1, tries: int = 4):
    """Read one band of a (remote) raster onto grid G."""
    err = None
    for attempt in range(tries):
        try:
            with rasterio.open(href) as src:
                with WarpedVRT(src, crs=G.crs, transform=G.transform, width=G.w, height=G.h,
                               resampling=resampling) as vrt:
                    return vrt.read(band)
        except Exception as e:  # network hiccups are common with remote COGs
            err = e
            time.sleep(1.5 + 2.5 * attempt)
    raise err


def resample_nearest(arr: np.ndarray, src: Grid, dst: Grid, fill=0) -> np.ndarray:
    """Nearest-neighbour resample between two grids in the same CRS."""
    x = dst.xmin + (np.arange(dst.w) + 0.5) * dst.res
    y = dst.ymax - (np.arange(dst.h) + 0.5) * dst.res
    c = np.floor((x - src.xmin) / src.res).astype(int)
    r = np.floor((src.ymax - y) / src.res).astype(int)
    okc = (c >= 0) & (c < src.w)
    okr = (r >= 0) & (r < src.h)
    out = arr[np.clip(r, 0, src.h - 1)][:, np.clip(c, 0, src.w - 1)].copy()
    out[~okr, :] = fill
    out[:, ~okc] = fill
    return out


def block_fraction(mask: np.ndarray, src: Grid, dst: Grid) -> np.ndarray:
    """Share of each (coarser) dst pixel covered by a fine boolean mask on src."""
    x = src.xmin + (np.arange(src.w) + 0.5) * src.res
    y = src.ymax - (np.arange(src.h) + 0.5) * src.res
    c = np.floor((x - dst.xmin) / dst.res).astype(int)
    r = np.floor((dst.ymax - y) / dst.res).astype(int)
    ok = np.outer((r >= 0) & (r < dst.h), (c >= 0) & (c < dst.w))
    idx = (np.clip(r, 0, dst.h - 1)[:, None] * dst.w + np.clip(c, 0, dst.w - 1)[None, :])
    tot = np.bincount(idx[ok], minlength=dst.w * dst.h).astype(float)
    hit = np.bincount(idx[ok], weights=mask[ok].astype(float), minlength=dst.w * dst.h)
    return (hit / np.maximum(tot, 1)).reshape(dst.h, dst.w)


def mask_to_geojson(mask: np.ndarray, G: Grid, min_px: int = 1):
    """Boolean mask -> GeoJSON MultiPolygon in lon/lat (pixel-edge polygons with holes)."""
    polys = []
    if mask.any():
        tr = G._from()
        for geom, _ in features.shapes(mask.astype(np.uint8), mask=mask, transform=G.transform):
            rings = geom["coordinates"]
            if len(rings[0]) - 1 <= 4 and min_px > 1:
                continue
            out = []
            for ring in rings:
                xs, ys = zip(*ring)
                lon, lat = tr.transform(xs, ys)
                out.append([[round(a, 6), round(b, 6)] for a, b in zip(lon, lat)])
            polys.append(out)
    return {"type": "MultiPolygon", "coordinates": polys}


def mask_to_path(mask: np.ndarray, zoom: float = 1.0) -> str:
    """Boolean mask -> SVG path (pixel-edge outline, use fill-rule evenodd), scaled by zoom."""
    parts = []
    if mask.any():
        for geom, _ in features.shapes(mask.astype(np.uint8), mask=mask, transform=Affine.identity()):
            for ring in geom["coordinates"]:
                parts.append("M" + "L".join(f"{x * zoom:.1f},{y * zoom:.1f}" for x, y in ring[:-1]) + "Z")
    return "".join(parts)


def km_between(lat1, lon1, lat2, lon2) -> float:
    k = np.cos(np.radians((lat1 + lat2) / 2))
    return float(np.hypot(lat1 - lat2, (lon1 - lon2) * k) * 111.32)
