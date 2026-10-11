"""Step 4: hourly ERA5 2 m temperature for each planned clip (Open-Meteo archive).

ERA5 is hourly on the hour; clips start 5 min past, so we interpolate linearly.
Writes data/temperature.csv (clip_id, temp_c, plus site context).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA = Path(__file__).resolve().parent.parent / "data"
URL = "https://archive-api.open-meteo.com/v1/archive"


def era5(lat, lon, start, end):
    for attempt in range(6):
        r = requests.get(URL, params=dict(latitude=lat, longitude=lon, start_date=start,
                                          end_date=end, hourly="temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
                                          models="era5", timezone="GMT"), timeout=120)
        if r.status_code == 200:
            h = r.json()["hourly"]
            df = pd.DataFrame(h)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            return df.set_index("time")
        time.sleep(5 * 2 ** attempt)
    r.raise_for_status()


def main():
    plan = pd.read_csv(DATA / "plan.csv", parse_dates=["t_utc"])
    sites = pd.read_csv(DATA / "sites.csv").set_index("site_id")
    cache = DATA / "era5"
    cache.mkdir(exist_ok=True)
    out = []
    for sid, grp in plan.groupby("site_id"):
        path = cache / f"{sid}.parquet"
        if path.exists():
            w = pd.read_parquet(path)
        else:
            lat, lon = sites.loc[sid, ["lat", "lon"]]
            start = (grp.t_utc.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            end = (grp.t_utc.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            w = era5(lat, lon, start, end)
            w.to_parquet(path)
            time.sleep(1)
        t = grp.t_utc.astype("int64").values
        ti = w.index.astype("int64").values
        g = grp[["clip_id"]].copy()
        for col, name in [("temperature_2m", "temp_c"), ("relative_humidity_2m", "rh"),
                          ("wind_speed_10m", "wind"), ("precipitation", "precip")]:
            g[name] = np.interp(t, ti, w[col].values.astype(float))
        out.append(g)
        print(sid, len(g), f"{g.temp_c.min():.1f}..{g.temp_c.max():.1f} C")
    pd.concat(out).to_csv(DATA / "temperature.csv", index=False)


if __name__ == "__main__":
    main()
