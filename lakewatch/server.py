"""Local web app: pick a lake on a map, watch the analysis run, open the report."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

from flask import Flask, abort, jsonify, request, send_from_directory

from . import pipeline, report
from .cli import slug
from .pipeline import Config, LakeNotFound, NotEnoughData

HERE = os.path.dirname(os.path.abspath(__file__))


def create_app(reports_dir: str = "reports") -> Flask:
    reports_dir = os.path.abspath(reports_dir)
    os.makedirs(reports_dir, exist_ok=True)
    app = Flask(__name__, static_folder=None)
    jobs: dict = {}
    one_at_a_time = threading.Lock()

    def worker(job, cfg):
        def log(stage, msg, frac=None):
            job["stage"], job["msg"] = stage, msg
            if frac is not None:
                job["progress"] = round(float(frac), 3)
            job["log"].append(msg)
        with one_at_a_time:
            job["state"] = "running"
            try:
                report.build(pipeline.run(cfg, log), log)
                job["state"] = "done"
            except (LakeNotFound, NotEnoughData) as e:
                job.update(state="failed", msg=str(e))
            except Exception as e:  # keep the server alive and show the reason
                job.update(state="failed", msg=f"{type(e).__name__}: {e}")
            job["finished"] = time.time()

    @app.get("/")
    def picker():
        return send_from_directory(f"{HERE}/template", "picker.html")

    @app.get("/vendor/<path:p>")
    def vendor(p):
        return send_from_directory(f"{HERE}/template/vendor", p)

    @app.post("/api/run")
    def run():
        b = request.get_json(force=True) or {}
        try:
            lat, lon = float(b["lat"]), float(b["lon"])
        except (KeyError, TypeError, ValueError):
            return jsonify(error="lat and lon are required"), 400
        if not (-60 <= lat <= 75 and -180 <= lon <= 180):
            return jsonify(error="That point is outside the area the satellites cover."), 400
        name = (str(b.get("name") or "").strip() or "Lake")[:80]
        sid = f"{slug(name)}-{lat:.4f}_{lon:.4f}".replace(".", "p")
        months = lambda k, d: tuple(int(x) for x in b.get(k, d))  # noqa: E731
        cfg = Config(lat=lat, lon=lon, name=name, out=os.path.join(reports_dir, sid),
                     units="ha" if b.get("units") == "ha" else "acres",
                     neighbours_km=float(b.get("neighbours_km", 15)),
                     peak_months=months("peak_months", (9, 10, 11, 2, 3)),
                     dry_months=months("dry_months", (3, 4, 5, 6)),
                     rain_months=months("rain_months", (6, 7, 8, 9, 10)),
                     year_start=int(b.get("year_start", 7)))
        jid = uuid.uuid4().hex[:12]
        jobs[jid] = dict(id=jid, state="queued", stage="queued", msg="Waiting for the previous lake to finish",
                         progress=0.0, log=[], report=f"/reports/{sid}/", name=name, started=time.time())
        threading.Thread(target=worker, args=(jobs[jid], cfg), daemon=True).start()
        return jsonify(id=jid)

    @app.get("/api/jobs/<jid>")
    def job(jid):
        j = jobs.get(jid)
        if not j:
            abort(404)
        return jsonify({k: v for k, v in j.items() if k != "log"} | {"log": j["log"][-6:]})

    @app.get("/api/reports")
    def listing():
        rows = []
        for d in sorted(os.listdir(reports_dir)):
            f = os.path.join(reports_dir, d, "data.json")
            if os.path.exists(f):
                try:
                    m = json.load(open(f))["meta"]
                    rows.append(dict(url=f"/reports/{d}/", name=m["name"], lat=m["lat"], lon=m["lon"],
                                     generated=m["generated"]))
                except Exception:
                    continue
        return jsonify(rows)

    @app.get("/reports/<sid>/")
    @app.get("/reports/<sid>/<path:p>")
    def served(sid, p="index.html"):
        return send_from_directory(os.path.join(reports_dir, sid), p)

    return app
