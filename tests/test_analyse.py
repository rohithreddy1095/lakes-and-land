import numpy as np

from lakewatch import analyse as an
from lakewatch.geo import Grid, block_fraction, mask_to_geojson, mask_to_path, resample_nearest, utm_epsg


def disc(n, r, cy=None, cx=None):
    cy = n // 2 if cy is None else cy
    cx = n // 2 if cx is None else cx
    y, x = np.mgrid[0:n, 0:n]
    return np.hypot(y - cy, x - cx) <= r


def test_hydro_year():
    assert an.hydro_year("2020-07-01", 7) == 2020
    assert an.hydro_year("2021-03-15", 7) == 2020
    assert an.hydro_year("2021-03-15", 1) == 2021


def test_find_lake_snaps_to_nearby_water():
    ext = disc(100, 15)
    occ = np.where(disc(100, 8), 90, 20).astype(np.uint8)
    comp, core, edge = an.find_lake(ext, occ, 50, 70)      # 5 px outside the shore
    assert comp.sum() == ext.sum() and core.sum() == disc(100, 8).sum() and not edge
    assert an.find_lake(ext, occ, 5, 5) is None


def test_fill_small_gaps_only_fills_narrow_gaps():
    a = np.full((20, 20), 50, np.int8)
    a[:, 5] = an.NODATA
    a[:, 10:19] = an.NODATA
    f = an.fill_small_gaps(a, 3)
    assert (f[:, 5] == 50).all() and (f[:, 14] == an.NODATA).all()


def test_pick_peaks_prefers_clear_scene_near_the_maximum():
    sc = [dict(hy=2000, month=10, area=100, valid=0.90, gappy=False),
          dict(hy=2000, month=11, area=98, valid=0.99, gappy=False),
          dict(hy=2000, month=1, area=300, valid=1.0, gappy=False),    # outside the peak months
          dict(hy=2000, month=2, area=120, valid=0.5, gappy=False)]    # too cloudy
    assert an.pick_peaks(sc, (9, 10, 11, 2, 3))[2000]["area"] == 98


def test_eras_and_alert_replay():
    n = 60
    big, small = disc(n, 20), disc(n, 14)
    years = list(range(1990, 2026))
    wet = {y: (big if y < 2000 else small) for y in years}
    wet[2010] = big                                           # one very wet year
    known = {y: np.ones((n, n), bool) for y in years}
    ref, now = an.choose_eras(years, 2026)
    assert ref == list(range(1990, 1996)) and now == list(range(2020, 2026))
    m = an.era_masks(wet, known, ref, now, core=disc(n, 3))
    assert m["full_ref"].sum() == big.sum() and m["still"].sum() == small.sum()
    assert m["never"].sum() == (big & ~small).sum() and m["wettest"].sum() == 0
    alert = an.replay_alerts(wet, known, years, m["full_ref"])
    ring = big & ~small
    assert set(np.unique(alert[ring])) == {2002}              # dry in 2000, 2001, 2002
    assert (alert[small] == 0).all()


def test_unobserved_years_do_not_count_as_dry():
    wet = {y: np.ones((4, 4), bool) for y in (1, 2)} | {y: np.zeros((4, 4), bool) for y in (3, 4, 5, 6)}
    known = {y: np.ones((4, 4), bool) for y in range(1, 7)}
    known[4] = np.zeros((4, 4), bool)                         # cloud
    alert = an.replay_alerts(wet, known, list(range(1, 7)), np.ones((4, 4), bool))
    assert (alert == 6).all()


def test_cover_classes():
    # five observations of a 1 x 4 strip: water, crop, thicket, bare
    mn = np.array([[[50, -30, -30, -30]]] * 5, np.int8)
    nv = np.array([[[0, 70, 70, 10]], [[0, 60, 65, 10]], [[0, 20, 60, 10]], [[0, 15, 55, 12]], [[0, 15, 58, 12]]], np.int8)
    months = np.array([10, 11, 3, 4, 5])
    c = an.classify_cover_year(mn, nv, months, (3, 4, 5, 6))
    assert c.tolist() == [[1, 2, 3, 4]]
    farmed, weeds, need = an.summarise_cover(np.stack([c] * 6))
    assert need == 3 and farmed.tolist() == [[False, True, False, False]] and weeds.tolist() == [[False, False, True, False]]


def test_compass():
    assert an.compass(1, 0) == "N" and an.compass(0, 1) == "E" and an.compass(-1, -1) == "SW"


def test_grid_helpers():
    assert utm_epsg(79.79, 16.83) == "EPSG:32644" and utm_epsg(-70.0, -33.0) == "EPSG:32719"
    g = Grid("EPSG:32644", 30.0, 500000.0, 1900000.0, 10, 10)
    s = Grid("EPSG:32644", 10.0, 500000.0, 1900000.0, 30, 30)
    m = np.zeros((10, 10), bool)
    m[2:4, 3:6] = True
    up = resample_nearest(m, g, s, False)
    assert up.sum() == m.sum() * 9
    assert np.allclose(block_fraction(up, s, g), m.astype(float))
    gj = mask_to_geojson(m, g)
    assert gj["type"] == "MultiPolygon" and len(gj["coordinates"]) == 1
    assert mask_to_path(m, 2.0).startswith("M")
    r, c = g.ll2px(*g.px2ll(4, 7))
    assert int(r) == 4 and int(c) == 7
