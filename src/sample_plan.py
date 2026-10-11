"""Step 2: choose sensors and the night-time clips to analyse.

One sensor per A2O region (the one with the most audio), at least a year of
coverage. For each sensor we draw nights stratified by calendar month and take
a 40 s clip every two hours from 19:00 to 05:00 local time, so each night gives
a within-night temperature gradient as well as seasonal contrast.

Usage: python sample_plan.py [n_sites] [nights_per_month]
Writes data/recordings/<site>.parquet and data/plan.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import a2o

DATA = Path(__file__).resolve().parent.parent / "data"
LOCAL_HOURS = [19, 21, 23, 1, 3, 5]
CLIP_S = 40
CLIP_OFFSET_MIN = 5   # start 5 min past the hour, clear of file boundaries
SEED = 7


def choose_sites(n_sites):
    s = pd.read_csv(DATA / "sites.csv")
    s = s[(s.n_recordings >= 3000) & (s.span_days >= 365)]
    s = s.sort_values("n_recordings", ascending=False).groupby("region_id").head(1)
    if n_sites >= len(s):
        return s
    # spread across latitude: take evenly spaced ranks
    s = s.sort_values("lat").reset_index(drop=True)
    idx = np.unique(np.linspace(0, len(s) - 1, n_sites).round().astype(int))
    return s.iloc[idx]


def recording_table(site_id):
    path = DATA / "recordings" / f"{site_id}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    r = pd.DataFrame(a2o.recordings(site_id))
    r["start"] = pd.to_datetime(r["recorded_date"], utc=True)
    r["end"] = r["start"] + pd.to_timedelta(r["duration_seconds"], unit="s")
    r = r.drop(columns="recorded_date").sort_values("start").reset_index(drop=True)
    path.parent.mkdir(exist_ok=True)
    r.to_parquet(path)
    return r


def utc_offset_hours(rec):
    off = rec["recorded_utc_offset"].dropna()
    if off.empty:
        return None
    s = off.mode().iloc[0].replace(":", "")   # "+10:00" or "+1000"
    sign = -1 if s.startswith("-") else 1
    return sign * (int(s[1:3]) + int(s[3:5]) / 60)


def plan_site(site, nights_per_month, rng):
    rec = recording_table(int(site.site_id))
    off = utc_offset_hours(rec)
    if off is None:
        off = round(site.lon / 15 * 2) / 2
    starts, ends = rec["start"].values, rec["end"].values

    def find(t_utc):
        t0 = np.datetime64(t_utc.tz_convert(None))
        t1 = t0 + np.timedelta64(CLIP_S, "s")
        i = np.searchsorted(starts, t0, side="right") - 1
        if i >= 0 and ends[i] >= t1:
            return int(rec.at[i, "id"]), float((t0 - starts[i]) / np.timedelta64(1, "s"))
        return None

    # candidate nights: local evening dates spanning the coverage
    local = rec["start"] + pd.Timedelta(hours=off)
    nights = pd.Series(pd.date_range(local.min().normalize(), local.max().normalize(), freq="D"))
    rows = []
    for month, grp in nights.groupby(nights.dt.month):
        picked = 0
        for d in rng.permutation(grp.values):
            if picked >= nights_per_month:
                break
            d = pd.Timestamp(d)
            clips = []
            for h in LOCAL_HOURS:
                day = d if h >= 12 else d + pd.Timedelta(days=1)
                t_local = day + pd.Timedelta(hours=h, minutes=CLIP_OFFSET_MIN)
                t_utc = (t_local - pd.Timedelta(hours=off)).tz_localize("UTC")
                hit = find(t_utc)
                if hit:
                    clips.append((h, t_utc, *hit))
            if len(clips) >= 4:   # need most of the night to see a gradient
                picked += 1
                for h, t_utc, rid, offset_s in clips:
                    rows.append(dict(site_id=int(site.site_id), night=d.date(), local_hour=h,
                                     t_utc=t_utc, recording_id=rid,
                                     start_offset=round(offset_s, 1),
                                     utc_offset=off))
    return rows


def main():
    n_sites = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    npm = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    rng = np.random.default_rng(SEED)
    sites = choose_sites(n_sites)
    rows = []
    for _, site in sites.iterrows():
        r = plan_site(site, npm, rng)
        print(f"{site['name']:<40} lat {site.lat:7.2f}  clips {len(r)}")
        rows += r
    plan = pd.DataFrame(rows)
    plan["clip_id"] = (plan.site_id.astype(str) + "_" + plan.recording_id.astype(str) + "_"
                       + plan.start_offset.round().astype(int).astype(str))
    plan.to_csv(DATA / "plan.csv", index=False)
    print(len(plan), "clips across", plan.site_id.nunique(), "sites")


if __name__ == "__main__":
    main()
