# Lakes & Land

Pick any lake, tank or reservoir. Get a satellite report that shows how much of it no longer fills,
what is on that ground now, and which alerts a monthly monitoring system would have raised.

Built as a proof of concept for satellite-based encroachment monitoring of lakes and tanks. It uses
only free, open data and needs no account or API key.

## What you get

One input: a point on the lake (click the map, or give latitude and longitude).

One output: a report folder with an interactive page (`index.html`). The page shows:

- **The headline number**: how much of the lake's earlier full spread no longer fills, and how much of that is farmed.
- **A map** with the old shoreline, today's shoreline, lost ground, farmed ground and weeds, over high-resolution photos from several dates.
- **This month's bulletin**: open water from the latest radar pass, ground under crop, and patches that are newly dry.
- **A year-by-year replay** since the 1980s or 1990s, with the alerts a "three dry years" rule would have raised and what became of each one.
- **Hotspots**: the five largest patches of lost lakebed, each with its own flood record.
- **Then and now**: the fullest early year against a recent wet year.
- **Neighbouring water bodies**: every tank within 15 km, so you can tell one shrinking lake from a regional dry spell.

The page works in English and Telugu, in acres or hectares, on a phone or a desktop.

## Quick start

Python 3.10 or newer.

```bash
git clone https://github.com/rohithreddy1095/lakes-and-land.git
cd lakes-and-land
pip install -e .
```

### Pick a lake on a map

```bash
lakewatch serve
```

Open http://127.0.0.1:8000, click on the open water of a lake, give it a name and press
**Analyse this lake**. The report opens when it is ready.

### Or use the command line

```bash
lakewatch run --lat 16.832 --lon 79.7885 --name "Yathavakilla Cheruvu"
```

The report is written to `reports/yathavakilla-cheruvu/`. Open `index.html` in a browser.

A run reads roughly 500 Landsat scenes, 400 Sentinel-2 scenes and 60 radar passes, a small window from
each. Expect 5 to 30 minutes for a first run, depending on the lake's size and your connection.
Downloads are cached in `~/.cache/lakewatch`, so a second run on the same lake takes a few minutes.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--units` | `acres` | `acres` or `ha` |
| `--peak-months` | `9,10,11,2,3` | Months when the lake is fullest and nearby fields are not flooded |
| `--dry-months` | `3,4,5,6` | Months when a harvested field is bare |
| `--rain-months` | `6,7,8,9,10` | Rainy season, for the rainfall line |
| `--year-start` | `7` | First month of the water year |
| `--search-km` | `10` | Half-width of the window searched for the lake. Raise it for large reservoirs |
| `--neighbours-km` | `15` | Radius of the neighbour scan. `0` skips it |
| `--no-photos` | off | Skip the high-resolution photo lookup |

The month defaults suit monsoon-fed tanks in peninsular India. December and January are left out of the
peak months because flooded paddy looks like lake water then. For other climates, change the months.

## How it works

1. **Find the lake.** The point is matched to a water body in the JRC Global Surface Water maximum-extent map.
2. **Measure water every year.** Landsat 4–9 (30 m), MNDWI > 0, joined to the lake's core. Each water year uses its fullest clear image in the peak months.
3. **Compare two periods.** The reference shoreline is the ground under water in at least half of the first six years on record. "Now" is the last six complete years. Ground inside the reference shoreline that no longer fills is split into "never flooded now" and "floods only in the wettest years".
4. **See what is there now.** Sentinel-2 (10 m): ground that turns green and is then harvested or bare, in at least three years, counts as farmed. Ground that stays green counts as weeds.
5. **Replay alerts.** Ground that stays dry at peak for three observed years in a row raises an alert. Each alert is then checked against today: farmed, still dry, or water came back.
6. **Check the present.** Sentinel-1 radar gives open water every 12 days through cloud. The latest clear Sentinel-2 image is compared with the same weeks in earlier years to flag newly dry patches.
7. **Scan the neighbours.** The same two periods are compared for every water body nearby.

## Limits

Read these before you act on a report.

- **It shows change, not illegality.** The legal boundary of a tank is its notified Full Tank Level and the survey record. A report tells you where to look.
- **"Farmed" is an upper bound.** Natural plants on drying mud can pass the crop test. Check against the photos and on the ground.
- **The record starts in the 1980s or 1990s.** A lake that shrank before then looks unchanged. The 1972–81 Landsat MSS archive is not used.
- **The alert rule over-alerts in a run of dry years.** In testing, roughly half of replayed alerts later refilled. A live system should weigh rainfall.
- **Small ponds are hard.** Below about 5 hectares the 30 m record is noisy.
- **Rainfall is coarse.** NASA POWER has a grid of about 50 km.
- **The Telugu text has not been reviewed by a native speaker.**

## Data sources

| Data | Source | Notes |
| --- | --- | --- |
| Landsat Collection 2 Level-2 | USGS / NASA, via Microsoft Planetary Computer | Public domain |
| Sentinel-1 RTC, Sentinel-2 L2A | ESA Copernicus, via Microsoft Planetary Computer | Free and open |
| Global Surface Water v1.4 | EC JRC / Google | Used only to find the lake |
| Monthly rainfall | NASA POWER | From 1981 |
| High-resolution photos | Esri World Imagery Wayback | The report loads tiles live from Esri and needs internet for this layer. Nothing is downloaded or redistributed. Esri's terms of use apply |
| Village names | OpenStreetMap, via Overpass | Optional. ODbL |

The report folder holds only Landsat and Sentinel images, so it can be shared freely. Without internet
the map falls back to the Sentinel-2 image.

## Project layout

```
lakewatch/
  cli.py        command line
  server.py     local web app with the map picker
  pipeline.py   the analysis, step by step
  analyse.py    pure-numpy logic: peaks, periods, alerts, land cover
  sources.py    data access
  geo.py        grids and raster-to-vector helpers
  report.py     writes the report folder
  template/     report page and picker page
tests/          unit tests for the analysis logic
```

Run the tests with `pytest`.

## Licence

MIT. See [LICENSE](LICENSE). Leaflet (BSD-2-Clause) is bundled in `lakewatch/template/vendor`.
