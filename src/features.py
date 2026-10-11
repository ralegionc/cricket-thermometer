"""Step 3: download each planned clip and measure pulse rates per carrier band.

For every 250 Hz carrier band between 1 and 11 kHz we take the amplitude
envelope and its modulation spectrum (Welch). Insect songs are trains of
pulses at a fixed carrier frequency, so a pulsing insect shows up as a sharp
modulation peak in its band. We keep:

  modspec   log-spaced modulation spectrum per band (for ML / plots)
  band_db   band level
  fund_rate fundamental pulse rate per band (sub-bin interpolated, Hz)
  fund_prom local peak prominence (peak power / median power at 0.75-1.33x)

Audio is never written to disk. Usage: python features.py [workers]
"""
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import ndimage, signal

import a2o

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "features"
SR = 22050
N_FFT, HOP = 512, 64
ENV_FS = SR / HOP
BAND_EDGES = np.arange(1000, 11001, 250)            # 40 bands
RATE_MIN, RATE_MAX = 1.0, 150.0
MOD_BINS = np.geomspace(RATE_MIN, RATE_MAX, 49)     # 48 log-spaced bins
WELCH_SEG = 4096                                    # 11.9 s -> 0.084 Hz bins
CHUNK = 100


def load(flac_bytes):
    y, sr = sf.read(io.BytesIO(flac_bytes), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        y = signal.resample_poly(y, SR // g, int(sr) // g).astype("float32")
    return y


def band_envelopes(y):
    f, _, Z = signal.stft(y, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP,
                          boundary=None, padded=False)
    P = np.abs(Z) ** 2
    env = np.stack([P[(f >= lo) & (f < hi)].sum(axis=0)
                    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])])
    return np.sqrt(env)


LOG_GRID = np.geomspace(RATE_MIN, RATE_MAX, 701)    # 0.72% steps


def local_prominence(rates, S):
    """Power over a local baseline: running median across 0.75x to 1.33x of
    each rate (on a log-rate grid). A global median would favour slow rates,
    because envelope noise is roughly 1/f."""
    logS = np.log(S)
    on_grid = np.stack([np.interp(LOG_GRID, rates, row) for row in logS])
    base = ndimage.median_filter(on_grid, size=(1, 81), mode="nearest")
    base = np.stack([np.interp(rates, LOG_GRID, row) for row in base])
    return np.exp(logS - base)


def fundamental(rates, spec, prom):
    """Pick the pulse fundamental, guarding against harmonics.

    Take the most prominent peak; if a peak at 1/2 or 1/3 of its rate carries
    at least 30% of its prominence, that lower one is the fundamental.
    """
    peaks, _ = signal.find_peaks(spec)
    if len(peaks) == 0:
        return np.nan, 0.0
    top = peaks[np.argmax(prom[peaks])]
    best = top
    for k in (3, 2):
        target = rates[top] / k
        if target < RATE_MIN:
            continue
        near = peaks[np.abs(rates[peaks] - target) <= max(0.06 * target, 0.12)]
        if len(near) and prom[near].max() >= 0.3 * prom[top]:
            best = near[np.argmax(prom[near])]
            break
    # parabolic interpolation on log power for sub-bin rate
    i = best
    if 0 < i < len(spec) - 1:
        a, b, c = np.log(spec[i - 1:i + 2])
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        shift = 0.0
    step = rates[1] - rates[0]
    return rates[i] + shift * step, float(prom[i])


def clip_features(y):
    env = band_envelopes(y)
    band_db = 10 * np.log10(np.median(env ** 2, axis=1) + 1e-12)
    norm = env / (env.mean(axis=1, keepdims=True) + 1e-12) - 1
    rates, S = signal.welch(norm, fs=ENV_FS, nperseg=WELCH_SEG, noverlap=WELCH_SEG // 2,
                            nfft=WELCH_SEG * 2, detrend="linear", axis=1)
    keep = (rates >= RATE_MIN) & (rates <= RATE_MAX)
    rates, S = rates[keep], S[:, keep] + 1e-20
    which = np.digitize(rates, MOD_BINS) - 1
    modspec = np.stack([np.log10(S[:, which == b].mean(axis=1)) for b in range(len(MOD_BINS) - 1)],
                       axis=1)
    P = local_prominence(rates, S)
    fr, fp = zip(*(fundamental(rates, s, p) for s, p in zip(S, P)))
    return dict(modspec=modspec.astype("float16"), band_db=band_db.astype("float32"),
                fund_rate=np.array(fr, "float32"), fund_prom=np.array(fp, "float32"))


def done_ids():
    ids = set()
    for p in OUT.glob("chunk_*.npz"):
        ids.update(np.load(p)["clip_id"].tolist())
    return ids


def save_chunk(batch):
    n = len(list(OUT.glob("chunk_*.npz")))
    np.savez_compressed(OUT / f"chunk_{n:05d}.npz",
                        clip_id=np.array([b["clip_id"] for b in batch]),
                        **{k: np.stack([b[k] for b in batch])
                           for k in ("modspec", "band_db", "fund_rate", "fund_prom")})


def fetch(row):
    try:
        raw = a2o.segment(row.recording_id, row.start_offset, row.start_offset + 40)
        return row.clip_id, load(raw)
    except Exception as e:   # one bad clip should not stop the run
        print("skip", row.clip_id, e, flush=True)
        return row.clip_id, None


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    OUT.mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv(DATA / "plan.csv")
    todo = plan[~plan.clip_id.isin(done_ids())]
    print(len(todo), "clips to process", flush=True)
    batch = []
    with ThreadPoolExecutor(workers) as ex:
        for i, (cid, y) in enumerate(ex.map(fetch, todo.itertuples()), 1):
            if y is not None and len(y) >= SR * 30:
                batch.append(dict(clip_id=cid, **clip_features(y)))
            if len(batch) >= CHUNK:
                save_chunk(batch)
                batch = []
            if i % 100 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    if batch:
        save_chunk(batch)


if __name__ == "__main__":
    main()
