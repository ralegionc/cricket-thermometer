"""Minimal client for the Australian Acoustic Observatory (A2O) API.

Anonymous access is enough for site metadata, recording listings and short
audio segments (media.flac with start/end offsets). Be polite: low concurrency.
"""
import time

import requests

API = "https://api.acousticobservatory.org"
HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}

_session = requests.Session()
_session.headers.update(HEADERS)


def _request(method, url, retries=5, **kw):
    for attempt in range(retries):
        try:
            r = _session.request(method, url, timeout=120, **kw)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {retries} attempts: {url}")


def _paged_get(path):
    out, page = [], 1
    while True:
        d = _request("GET", f"{API}/{path}", params={"items": 100, "page": page}).json()
        out += d["data"]
        if d["meta"]["paging"]["next"] is None:
            return out
        page += 1


def sites():
    return _paged_get("sites")


def regions():
    return _paged_get("regions")


def recordings(site_id, fields=("id", "recorded_date", "duration_seconds",
                                "sample_rate_hertz", "recorded_utc_offset")):
    """All recording metadata for one site, oldest first."""
    out, page = [], 1
    while True:
        body = {
            "filter": {"site_id": {"eq": site_id}},
            "projection": {"include": list(fields)},
            "sorting": {"order_by": "recorded_date", "direction": "asc"},
            "paging": {"items": 500, "page": page},
        }
        d = _request("POST", f"{API}/audio_recordings/filter", json=body).json()
        out += d["data"]
        if d["meta"]["paging"]["next"] is None:
            return out
        page += 1


def segment(recording_id, start, end):
    """FLAC bytes for [start, end) seconds of a recording.

    The server cuts segments on demand and occasionally answers 200 with a
    non-audio body while it does, so check the magic bytes and retry."""
    for attempt in range(4):
        r = _request("GET", f"{API}/audio_recordings/{recording_id}/media.flac",
                     params={"start_offset": start, "end_offset": end},
                     headers={"Accept": "audio/flac"})
        if r.content[:4] == b"fLaC":
            return r.content
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"no FLAC for recording {recording_id} @ {start}")
