"""Load features + plan + ERA5 into tidy tables."""
from pathlib import Path

import numpy as np
import pandas as pd

from features import BAND_EDGES

DATA = Path(__file__).resolve().parent.parent / "data"
BAND_LO = BAND_EDGES[:-1]


def clips():
    """One row per processed clip with weather and site context, plus arrays."""
    chunks = [np.load(p) for p in sorted((DATA / "features").glob("chunk_*.npz"))]
    arr = {k: np.concatenate([c[k] for c in chunks])
           for k in ("clip_id", "modspec", "band_db", "fund_rate", "fund_prom")}
    plan = pd.read_csv(DATA / "plan.csv", parse_dates=["t_utc"])
    temp = pd.read_csv(DATA / "temperature.csv")
    sites = pd.read_csv(DATA / "sites.csv")[["site_id", "name", "region", "lat", "lon"]]
    meta = pd.DataFrame({"clip_id": arr["clip_id"], "row": np.arange(len(arr["clip_id"]))})
    meta = (meta.merge(plan, on="clip_id").merge(temp, on="clip_id").merge(sites, on="site_id")
            .drop_duplicates("clip_id"))
    meta["t_local"] = meta.t_utc.dt.tz_localize(None) + pd.to_timedelta(meta.utc_offset, unit="h")
    meta["month"] = meta.t_local.dt.month
    meta["doy"] = meta.t_local.dt.dayofyear
    meta["night"] = meta.site_id.astype(str) + "_" + meta.night.astype(str)
    # hours since 19:00 local, so the night is a continuous axis
    meta["hours_in"] = (meta.local_hour - 19) % 24 + 5 / 60
    return meta.reset_index(drop=True), arr


def detections(meta, arr, min_prom=15.0, rate_range=(2.0, 120.0)):
    """Long table: one row per (clip, band) with a clear pulse train."""
    rate = arr["fund_rate"][meta.row.values]
    prom = arr["fund_prom"][meta.row.values]
    level = arr["band_db"][meta.row.values]
    ok = (prom >= min_prom) & (rate >= rate_range[0]) & (rate <= rate_range[1])
    i, b = np.nonzero(ok)
    d = meta.iloc[i][["clip_id", "site_id", "name", "night", "month", "local_hour", "hours_in",
                      "temp_c", "rh", "wind", "precip", "lat"]].reset_index(drop=True)
    d["band"] = b
    d["carrier_hz"] = BAND_LO[b] + 125
    d["rate"] = rate[i, b]
    d["log_rate"] = np.log(d.rate)
    d["prom"] = prom[i, b]
    d["level_db"] = level[i, b]
    return drop_clockwork(d)


def drop_clockwork(d, tol=0.002, min_nights=3):
    """Remove machine pulses: a rate repeated to within tol Hz at one site on
    min_nights or more nights. Insect rates drift with temperature and cannot
    do that; electric-fence energisers and similar devices do (Wambiana
    Cattle Station has a broadband 2.000 Hz tick on 10 nights)."""
    bad = np.zeros(len(d), bool)
    for shift in (0.0, 0.5):   # two staggered binnings catch values on a bin edge
        key = np.floor(d.rate / tol + shift).astype(int)
        nights = d.assign(key=key).groupby(["site_id", "key"]).night.transform("nunique")
        bad |= (nights >= min_nights).values
    bad = pd.Series(bad, index=d.index)
    if bad.any():
        print(f"clockwork filter: dropped {bad.sum()} of {len(d)} detections "
              f"({d[bad].groupby('name').size().to_dict()})")
    return d[~bad].reset_index(drop=True)
