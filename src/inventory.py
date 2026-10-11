"""Step 1: list A2O sites with coordinates and their recording coverage.

Writes data/sites.csv (one row per sensor point).
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import a2o

DATA = Path(__file__).resolve().parent.parent / "data"


def coverage(site_id):
    def one(direction):
        body = {
            "filter": {"site_id": {"eq": site_id}},
            "projection": {"include": ["recorded_date"]},
            "sorting": {"order_by": "recorded_date", "direction": direction},
            "paging": {"items": 1},
        }
        return a2o._request("POST", f"{a2o.API}/audio_recordings/filter", json=body).json()

    first, last = one("asc"), one("desc")
    n = first["meta"]["paging"]["total"]
    if n == 0:
        return dict(site_id=site_id, n_recordings=0, first=None, last=None)
    return dict(site_id=site_id, n_recordings=n,
                first=first["data"][0]["recorded_date"],
                last=last["data"][0]["recorded_date"])


def main():
    DATA.mkdir(exist_ok=True)
    regions = {r["id"]: r["name"] for r in a2o.regions()}
    rows = []
    for s in a2o.sites():
        if s["latitude"] is None:
            continue
        notes = {}
        try:
            notes = json.loads(s["notes"] or "{}")
        except json.JSONDecodeError:
            pass
        rows.append(dict(site_id=s["id"], name=s["name"], region_id=s["region_id"],
                         region=regions.get(s["region_id"]),
                         condition=notes.get("Condition"),
                         lat=s["latitude"], lon=s["longitude"]))
    sites = pd.DataFrame(rows)
    with ThreadPoolExecutor(4) as ex:
        cov = pd.DataFrame(list(ex.map(coverage, sites.site_id)))
    sites = sites.merge(cov, on="site_id")
    sites["first"] = pd.to_datetime(sites["first"], utc=True)
    sites["last"] = pd.to_datetime(sites["last"], utc=True)
    sites["span_days"] = (sites["last"] - sites["first"]).dt.days
    sites.sort_values(["region", "name"]).to_csv(DATA / "sites.csv", index=False)
    print(sites.describe(include="all").T[["count", "mean", "min", "max"]])
    print((sites.n_recordings > 0).sum(), "sites with audio")


if __name__ == "__main__":
    main()
