"""Step 5: does insect pulse rate track temperature, and can audio read it?

A. Song units. Within each site and carrier band, pulse-rate detections are
   grouped into "units" (one acoustic species, roughly) by splitting the sorted
   log-rates at gaps wider than GAP. A unit must be heard on >= MIN_NIGHTS nights.

B. Within-night test. Temperature falls through the night, and so (for
   ectotherms) should pulse rate. Comparing a unit with itself across hours of
   the same night removes species turnover and seasonal effects. Time of night
   is a confound (a circadian slowdown would look the same), so we regress
   d ln(rate) on d T *and* d hours: nights cool at different speeds, which
   separates the two.

C. Across-season fit. ln(rate) = unit intercept + beta * T (+ hour effects).
   beta converts to Q10 = exp(10 beta); insect muscle physiology predicts
   Q10 of roughly 1.5 to 3.

D. Thermometer. Per-unit inverse calibration (T on ln rate), cross-validated
   by night, combined across units heard in a clip. Baseline: site x month x
   hour climatology from training nights. The test that matters is whether
   audio beats or improves on climatology.

Writes results/*.csv, results/summary.json and figures/*.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from dataset import clips, detections

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
MIN_PROM = 15.0
GAP = 0.12          # ln units, about a 13% jump in rate
CARRIER_GAP = 4     # bands (1 kHz)
MIN_NIGHTS = 4
MIN_DET = 12
SEED = 0


def merge_bands(d, max_band_gap=2, rate_tol=0.03):
    """A broadband song shows the same pulse rate in several adjacent carrier
    bands. Within a clip, chain detections whose bands are within
    max_band_gap and whose rates agree to rate_tol; keep the most prominent."""
    d = d.sort_values(["clip_id", "band"]).reset_index(drop=True)
    song = np.zeros(len(d), int)
    sid = 0
    for _, idx in d.groupby("clip_id").indices.items():
        heads = []                      # (last band, log rate, song id)
        for i in idx:
            b, lr = d.band.iat[i], d.log_rate.iat[i]
            match = next((h for h in heads
                          if b - h[0] <= max_band_gap and abs(lr - h[1]) <= rate_tol), None)
            if match:
                heads.remove(match)
                heads.append((b, match[1], match[2]))
                song[i] = match[2]
            else:
                sid += 1
                heads.append((b, lr, sid))
                song[i] = sid
    d["song"] = song
    d["n_bands"] = d.groupby("song").band.transform("size")
    d["carrier_band"] = d.groupby("song").band.transform("mean")
    return d.sort_values("prom", ascending=False).drop_duplicates("song")


def assign_units(d, mode="band"):
    """Group detections into song units. Three definitions:

    band            (main, fixed before seeing results) same site and 250 Hz
                    band, split at log-rate gaps. A broadband song appears as
                    several units, one per band; inference bootstraps whole
                    nights, so the duplication does not shrink the CIs.
    merged_band     as band, after merging one song's adjacent-band copies
                    into its most prominent band.
    carrier_region  merged songs, split by log-rate gap first, then by
                    carrier gaps > CARRIER_GAP bands.
    """
    if mode == "band":
        d = d.sort_values(["site_id", "band", "log_rate"]).copy()
        d["unit"] = (d.site_id.ne(d.site_id.shift()) | d.band.ne(d.band.shift())
                     | (d.log_rate.diff() > GAP)).cumsum()
    elif mode == "merged_band":
        d = merge_bands(d).sort_values(["site_id", "band", "log_rate"])
        d["unit"] = (d.site_id.ne(d.site_id.shift()) | d.band.ne(d.band.shift())
                     | (d.log_rate.diff() > GAP)).cumsum()
    elif mode == "carrier_region":
        d = merge_bands(d).sort_values(["site_id", "log_rate"])
        d["rate_grp"] = (d.site_id.ne(d.site_id.shift()) | (d.log_rate.diff() > GAP)).cumsum()
        d = d.sort_values(["rate_grp", "carrier_band"])
        d["unit"] = (d.rate_grp.ne(d.rate_grp.shift())
                     | (d.carrier_band.diff() > CARRIER_GAP)).cumsum()
        d["carrier_hz"] = d.groupby("unit").carrier_hz.transform("median")
    else:
        raise ValueError(mode)
    stats = d.groupby("unit").agg(n=("rate", "size"), nights=("night", "nunique"),
                                  t_range=("temp_c", lambda t: t.max() - t.min()))
    keep = stats[(stats.n >= MIN_DET) & (stats.nights >= MIN_NIGHTS)].index
    d = d[d.unit.isin(keep)]
    # one detection per unit per clip
    d = d.sort_values("prom", ascending=False).drop_duplicates(["unit", "clip_id"])
    return d, stats.loc[keep]


def bootstrap_slope(df, formula, coef, group, n=500, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = df[group].unique()
    idx = {g: i for g, i in df.groupby(group).indices.items()}
    out = []
    for _ in range(n):
        pick = rng.choice(groups, len(groups))
        sample = df.iloc[np.concatenate([idx[g] for g in pick])]
        out.append(smf.ols(formula, sample).fit().params[coef])
    return np.percentile(out, [2.5, 97.5])


def within_night(d):
    pairs = []
    for (unit, night), g in d.groupby(["unit", "night"]):
        if len(g) < 2:
            continue
        g = g.sort_values("hours_in")
        a, b = np.triu_indices(len(g), 1)
        pairs.append(pd.DataFrame({
            "unit": unit, "night": night, "site_id": g.site_id.iloc[0],
            "d_log_rate": g.log_rate.values[b] - g.log_rate.values[a],
            "d_temp": g.temp_c.values[b] - g.temp_c.values[a],
            "d_hours": g.hours_in.values[b] - g.hours_in.values[a]}))
    p = pd.concat(pairs, ignore_index=True)
    # pairs that jump more than 25% are probably a different species in the band
    p = p[p.d_log_rate.abs() < np.log(1.25)]
    m_t = smf.ols("d_log_rate ~ 0 + d_temp", p).fit()
    m_th = smf.ols("d_log_rate ~ 0 + d_temp + d_hours", p).fit()
    ci_t = bootstrap_slope(p, "d_log_rate ~ 0 + d_temp", "d_temp", "night")
    ci_th = bootstrap_slope(p, "d_log_rate ~ 0 + d_temp + d_hours", "d_temp", "night")
    return p, dict(
        n_pairs=len(p), n_nights=p.night.nunique(), n_units=p.unit.nunique(),
        beta_temp_only=m_t.params["d_temp"], ci_temp_only=list(ci_t),
        beta_temp=m_th.params["d_temp"], ci_temp=list(ci_th),
        beta_hours=m_th.params["d_hours"],
        corr_dtemp_dhours=float(np.corrcoef(p.d_temp, p.d_hours)[0, 1]))


def across_season(d):
    m = smf.ols("log_rate ~ C(unit) + temp_c + C(local_hour)", d).fit()
    ci = bootstrap_slope(d, "log_rate ~ C(unit) + temp_c + C(local_hour)", "temp_c",
                         "night", n=200)
    per_unit = []
    for u, g in d.groupby("unit"):
        if g.temp_c.max() - g.temp_c.min() < 4:
            continue
        f = smf.ols("log_rate ~ temp_c", g).fit()
        per_unit.append(dict(unit=u, site=g.name.iloc[0], carrier_hz=g.carrier_hz.median(),
                             median_rate=g.rate.median(), n=len(g), nights=g.night.nunique(),
                             t_range=g.temp_c.max() - g.temp_c.min(),
                             beta=f.params["temp_c"], se=f.bse["temp_c"], r=np.sqrt(f.rsquared)
                             * np.sign(f.params["temp_c"])))
    pu = pd.DataFrame(per_unit)
    return pu, dict(beta_pooled=m.params["temp_c"], ci_pooled=list(ci),
                    q10_pooled=float(np.exp(10 * m.params["temp_c"])),
                    n_units_fit=len(pu), frac_units_positive=float((pu.beta > 0).mean()),
                    median_unit_beta=float(pu.beta.median()))


def thermometer(meta, d, n_folds=5):
    """Night-grouped CV. Returns per-clip predictions for clips with >=1 unit."""
    rng = np.random.default_rng(SEED)
    nights = meta.night.unique()
    fold_of = dict(zip(nights, rng.integers(0, n_folds, len(nights))))
    meta = meta.assign(fold=meta.night.map(fold_of))
    d = d.assign(fold=d.night.map(fold_of))
    meta = meta.assign(s1=np.sin(2 * np.pi * meta.doy / 365.25), c1=np.cos(2 * np.pi * meta.doy / 365.25),
                       s2=np.sin(4 * np.pi * meta.doy / 365.25), c2=np.cos(4 * np.pi * meta.doy / 365.25))
    clim_formula = "temp_c ~ s1 + c1 + s2 + c2 + C(local_hour)"
    preds = []
    for k in range(n_folds):
        tr_m, te_m = meta[meta.fold != k], meta[meta.fold == k]
        tr, te = d[d.fold != k], d[d.fold == k]
        # climatology: smooth seasonal cycle + hour-of-night effect, per site
        for sid, g in te_m.groupby("site_id"):
            f = smf.ols(clim_formula, tr_m[tr_m.site_id == sid]).fit()
            preds += [dict(clip_id=c, clim_pred=v, fold=k)
                      for c, v in zip(g.clip_id, f.predict(g))]
        cal = {}
        for u, g in tr.groupby("unit"):
            if g.night.nunique() < MIN_NIGHTS or g.temp_c.max() - g.temp_c.min() < 4:
                continue
            f = smf.ols("temp_c ~ log_rate", g).fit()
            if f.params["log_rate"] <= 0 or f.f_pvalue > 0.05:   # only units that behave
                continue
            cal[u] = (f.params["Intercept"], f.params["log_rate"], f.mse_resid,
                      g.temp_c.mean(), g.temp_c.var())
        for cid, g in te.groupby("clip_id"):
            est = [(a + b * lr, 1 / v, mu, 1 / s2) for u, lr in zip(g.unit, g.log_rate)
                   if u in cal for a, b, v, mu, s2 in [cal[u]]]
            if not est:
                continue
            t, w, mu, w0 = np.array(est).T
            preds.append(dict(clip_id=cid, audio_pred=float(np.average(t, weights=w)),
                              # control: which songs are present, ignoring their rate
                              presence_pred=float(np.average(mu, weights=w0)),
                              n_units=len(est), fold=k))
    p = pd.DataFrame(preds).groupby("clip_id").first().reset_index()
    p = p.merge(meta[["clip_id", "site_id", "name", "temp_c", "local_hour", "month"]], on="clip_id")
    a = p.dropna(subset=["audio_pred", "clim_pred"]).copy()
    # blend: learn the audio weight on the out-of-fold predictions (one parameter)
    resid_c = a.temp_c - a.clim_pred
    delta = a.audio_pred - a.clim_pred
    w = float(np.clip((resid_c * delta).sum() / (delta ** 2).sum(), 0, 1))
    a["blend_pred"] = a.clim_pred + w * delta
    mae = lambda x: float(np.mean(np.abs(x)))
    summary = dict(
        n_clips_with_audio_estimate=len(a),
        frac_clips_with_estimate=len(a) / len(meta),
        mae_climatology=mae(a.temp_c - a.clim_pred),
        mae_presence_only=mae(a.temp_c - a.presence_pred),
        mae_audio=mae(a.temp_c - a.audio_pred),
        mae_blend=mae(a.temp_c - a.blend_pred),
        blend_weight_audio=w,
        frac_audio_within_1c=float((np.abs(a.temp_c - a.audio_pred) <= 1).mean()),
        corr_audio_anomaly=float(np.corrcoef(a.audio_pred - a.clim_pred, resid_c)[0, 1]),
    )
    return a, summary


def figures(d, p, pu, a):
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

    # 1. best units: rate vs temperature
    # best-correlated units, skipping band copies of a song already shown
    chosen = []
    for _, u in pu.sort_values("r", ascending=False).iterrows():
        if not any(u.site == c.site and abs(np.log(u.median_rate / c.median_rate)) < 0.15
                   for c in chosen):
            chosen.append(u)
        if len(chosen) == 6:
            break
    top = pd.DataFrame(chosen)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for ax, (_, u) in zip(axes.flat, top.iterrows()):
        g = d[d.unit == u.unit]
        sc = ax.scatter(g.temp_c, g.rate, c=g.hours_in, cmap="viridis", s=14)
        tt = np.linspace(g.temp_c.min(), g.temp_c.max(), 50)
        f = np.polyfit(g.temp_c, g.log_rate, 1)
        ax.plot(tt, np.exp(np.polyval(f, tt)), color="k", lw=1)
        ax.set_title(f"{u.site[:24]}\n{u.carrier_hz/1000:.2f} kHz, Q10 {np.exp(10*u.beta):.2f}",
                     fontsize=9)
        ax.set_xlabel("ERA5 air temperature (°C)")
        ax.set_ylabel("pulse rate (Hz)")
    fig.colorbar(sc, ax=axes, label="hours after 19:00", shrink=0.6)
    fig.savefig(FIG / "units_rate_vs_temp.png", bbox_inches="tight")
    plt.close(fig)

    # 2. within-night pairs
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(p.d_temp, p.d_log_rate * 100, s=5, alpha=0.3)
    bins = pd.cut(p.d_temp, np.arange(-12, 6, 1))
    m = p.groupby(bins, observed=True).agg(x=("d_temp", "mean"), y=("d_log_rate", "mean"),
                                           n=("d_temp", "size"))
    m = m[m.n >= 10]
    ax.plot(m.x, m.y * 100, "o-", color="C3", label="binned mean")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("temperature change within night (°C)")
    ax.set_ylabel("pulse-rate change, same song (%)")
    ax.legend(frameon=False)
    fig.savefig(FIG / "within_night_pairs.png", bbox_inches="tight")
    plt.close(fig)

    # 3. per-unit Q10 distribution
    fig, ax = plt.subplots(figsize=(5.5, 4))
    q = np.exp(10 * pu.beta.clip(-0.2, 0.2))
    ax.hist(q, bins=np.arange(0.2, 4.01, 0.1), color="C0")
    ax.axvspan(1.5, 3, color="C2", alpha=0.15, label="typical insect Q10")
    ax.set_xlabel("Q10 of pulse rate, per song unit")
    ax.set_ylabel("song units")
    ax.legend(frameon=False)
    fig.savefig(FIG / "q10_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 4. thermometer
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    lim = [a.temp_c.min() - 1, a.temp_c.max() + 1]
    for ax, col, title in [(axes[0], "clim_pred", "climatology (season + hour, per site)"),
                           (axes[1], "audio_pred", "audio only")]:
        ax.scatter(a.temp_c, a[col], s=6, alpha=0.4)
        ax.plot(lim, lim, color="k", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"{title}\nMAE {np.mean(np.abs(a.temp_c - a[col])):.2f} °C", fontsize=10)
        ax.set_xlabel("ERA5 temperature (°C)")
    axes[0].set_ylabel("predicted (°C)")
    fig.savefig(FIG / "thermometer.png", bbox_inches="tight")
    plt.close(fig)


def main():
    RES.mkdir(exist_ok=True)
    meta, arr = clips()
    det = detections(meta, arr, min_prom=MIN_PROM)
    d, ustats = assign_units(det)
    print(f"{len(meta)} clips, {len(det)} detections, {d.unit.nunique()} units "
          f"at {d.site_id.nunique()} sites")
    p, s_night = within_night(d)
    pu, s_season = across_season(d)
    a, s_therm = thermometer(meta, d)
    summary = dict(n_clips=len(meta), n_sites=int(meta.site_id.nunique()),
                   n_detections=len(det), n_units=int(d.unit.nunique()),
                   within_night=s_night, across_season=s_season, thermometer=s_therm)
    summary = json.loads(json.dumps(summary, default=float))
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    pu.to_csv(RES / "units.csv", index=False)
    a.to_csv(RES / "thermometer_predictions.csv", index=False)
    site_err = a.assign(e_audio=(a.temp_c - a.audio_pred).abs(),
                        e_clim=(a.temp_c - a.clim_pred).abs()).groupby("name").agg(
        clips=("clip_id", "size"), mae_audio=("e_audio", "mean"), mae_clim=("e_clim", "mean"))
    site_err.to_csv(RES / "thermometer_by_site.csv")
    by_rate = pu.assign(q10=np.exp(10 * pu.beta)).groupby(
        pd.cut(pu.median_rate, [0, 5, 15, 40, 120]), observed=True).agg(
        units=("q10", "size"), median_q10=("q10", "median"), median_r=("r", "median"),
        frac_positive=("beta", lambda b: (b > 0).mean()))
    by_rate.to_csv(RES / "units_by_rate_class.csv")
    figures(d, p, pu, a)
    print(json.dumps(summary, indent=2))
    print(site_err.round(2).to_string())
    print(by_rate.round(2).to_string())

    # sensitivity to the song-unit definition
    rows = []
    for mode in ("band", "merged_band", "carrier_region"):
        dm, _ = assign_units(det, mode)
        _, sn = within_night(dm)
        pm, ss = across_season(dm)
        _, st = thermometer(meta, dm)
        rows.append(dict(unit_definition=mode, units=dm.unit.nunique(),
                         within_night_beta=sn["beta_temp"], within_night_ci=sn["ci_temp"],
                         season_beta=ss["beta_pooled"], frac_units_positive=ss["frac_units_positive"],
                         clips_with_estimate=st["n_clips_with_audio_estimate"],
                         mae_climatology=st["mae_climatology"], mae_presence=st["mae_presence_only"],
                         mae_audio=st["mae_audio"], mae_blend=st["mae_blend"]))
    sens = pd.DataFrame(rows)
    sens.to_csv(RES / "sensitivity_unit_definition.csv", index=False)
    print(sens.round(3).to_string())


if __name__ == "__main__":
    main()
