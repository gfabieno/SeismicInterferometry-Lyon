#!/usr/bin/env python3
"""
Download RESIF (ws.resif.fr) miniSEED data for network 1F stations A001-A099 and B001-B099,
for all available time, split into 1 UTC day per file, 3 channels per file.

Uses:
- Availability extent endpoint to discover station time range:
  https://ws.resif.fr/fdsnws/availability/1/extent?...
- DataSelect endpoint to download waveform data:
  https://ws.resif.fr/fdsnws/dataselect/1/query?...

Requirements:
  pip install requests

Example:
  python download_resif_1f.py --out data_1F
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import requests


AVAIL_EXTENT_URL = "https://ws.resif.fr/fdsnws/availability/1/extent"
DATASELECT_URL = "https://ws.resif.fr/fdsnws/dataselect/1/query"


@dataclass(frozen=True)
class Config:
    network: str = "1F"
    location: str = "00"
    quality: str = "M"
    channels_csv: str = "DPE,DPN,DPZ"
    out_dir: str = "resif_1F"
    timeout_s: int = 120
    retries: int = 5
    backoff_s: float = 2.0
    user_agent: str = "resif-downloader/1.0 (contact: you@example.com)"


def station_codes(prefixes=("A", "B")) -> list[str]:
    codes = []
    for prefix in prefixes:
        for i in range(1, 100):
            codes.append(f"{prefix}{i:03d}")
    return codes


def isoformat_utc(t: dt.datetime) -> str:
    # RESIF accepts ISO strings without 'Z' fine; keep explicit UTC by using naive UTC datetime here.
    return t.strftime("%Y-%m-%dT%H:%M:%S")


def utc_day_floor(t: dt.datetime) -> dt.datetime:
    return dt.datetime(t.year, t.month, t.day)


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: dict,
    timeout_s: int,
    retries: int,
    backoff_s: float,
    stream: bool = False,
) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = session.request(method, url, params=params, timeout=timeout_s, stream=stream)
            return resp
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            sleep_s = backoff_s * (2 ** attempt)
            time.sleep(sleep_s)
    raise RuntimeError(f"Unreachable: {last_exc}")


def parse_availability_extent_text(text: str) -> Optional[Tuple[dt.datetime, dt.datetime]]:
    """
    Parse RESIF availability extent output in TEXT format.

    Typical rows are whitespace-delimited and include start/end timestamps.
    We'll extract the earliest start and latest end from all data rows.
    """
    min_start: Optional[dt.datetime] = None
    max_end: Optional[dt.datetime] = None

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        # Heuristic: look for two ISO-like timestamps in the row.
        # Availability extent commonly includes starttime and endtime as columns.
        iso_candidates = [p for p in parts if ("T" in p and "-" in p and ":" in p)]
        if len(iso_candidates) < 2:
            continue

        start_s, end_s = iso_candidates[0], iso_candidates[1]
        for s in (start_s, end_s):
            if s.endswith("Z"):
                # strip Z; we treat as UTC-naive
                pass
        try:
            start_dt = dt.datetime.fromisoformat(start_s.replace("Z", ""))
            end_dt = dt.datetime.fromisoformat(end_s.replace("Z", ""))
        except ValueError:
            continue

        if min_start is None or start_dt < min_start:
            min_start = start_dt
        if max_end is None or end_dt > max_end:
            max_end = end_dt

    if min_start is None or max_end is None:
        return None
    return (min_start, max_end)


def get_station_timerange(cfg: Config, session: requests.Session, station: str) -> Optional[Tuple[dt.datetime, dt.datetime]]:
    """
    Use availability /extent to discover full available time range for the given station.
    Per RESIF docs, if start/end are omitted, it defaults to the fully available time range.
    """
    params = {
        "net": cfg.network,
        "sta": station,
        "loc": cfg.location,
        "cha": cfg.channels_csv,
        "quality": cfg.quality,
        "format": "text",  # easier to parse robustly
        "nodata": 204,
    }

    resp = request_with_retries(
        session,
        "GET",
        AVAIL_EXTENT_URL,
        params=params,
        timeout_s=cfg.timeout_s,
        retries=cfg.retries,
        backoff_s=cfg.backoff_s,
        stream=False,
    )

    if resp.status_code == 204:
        return None
    resp.raise_for_status()

    rng = parse_availability_extent_text(resp.text)
    return rng


def iter_utc_days(start: dt.datetime, end: dt.datetime) -> Iterable[Tuple[dt.datetime, dt.datetime]]:
    """
    Yield [day_start, day_end) windows in UTC, clipped to [start, end).
    """
    if end <= start:
        return
    day = utc_day_floor(start)
    # Move to the next day boundary if start is not at midnight; we still want that partial day.
    # We'll clip each yielded day window to [start, end).
    while day < end:
        next_day = day + dt.timedelta(days=1)
        win_start = max(start, day)
        win_end = min(end, next_day)
        if win_end > win_start:
            yield (win_start, win_end)
        day = next_day


def build_filename(cfg: Config, station: str, day_start: dt.datetime) -> str:
    # One file per UTC day, containing all 3 channels (DPE,DPN,DPZ)
    datestr = day_start.strftime("%Y%m%d")
    chans = cfg.channels_csv.replace(",", "-")
    return f"{cfg.network}.{station}.{cfg.location}.{chans}.{cfg.quality}.{datestr}.mseed"

def estimate_station_size(
    cfg: Config,
    session: requests.Session,
    station: str,
    min_sample_secs: int = 3600,
) -> float:
    """
    Estimate total bytes for this station by downloading a small sample chunk
    (min_sample_secs long) and extrapolating to the station's full duration.
    Returns an estimate in bytes (float).
    """
    rng = get_station_timerange(cfg, session, station)
    if rng is None:
        return 0.0

    start, end = rng
    duration = (end - start).total_seconds()
    if duration <= 0:
        return 0.0

    # Choose sample window (start of available range)
    sample_end = start + dt.timedelta(seconds=min_sample_secs)
    sample_end = min(sample_end, end)

    params = {
        "network": cfg.network,
        "station": station,
        "location": cfg.location,
        "quality": cfg.quality,
        "channel": cfg.channels_csv,
        "starttime": isoformat_utc(start),
        "endtime": isoformat_utc(sample_end),
        "nodata": 204,
    }

    # Try a HEAD first (faster) — if Content-Length available, great
    try:
        resp = session.head(DATASELECT_URL, params=params, timeout=30)
        length = resp.headers.get("Content-Length")
        if length is not None:
            sample_bytes = float(length)
        else:
            # fallback: GET and read full sample
            g = session.get(DATASELECT_URL, params=params, timeout=60)
            if g.status_code == 204:
                return 0.0
            g.raise_for_status()
            sample_bytes = float(len(g.content))
    except Exception as exc:
        print(f"  ! Sample error for {station}: {exc}")
        return 0.0

    # Extrapolate to full duration
    sample_secs_actual = (sample_end - start).total_seconds()
    if sample_secs_actual <= 0:
        return 0.0

    bytes_per_sec = sample_bytes / sample_secs_actual
    estimated_total = bytes_per_sec * duration
    return estimated_total

def download_day(
    cfg: Config,
    session: requests.Session,
    station: str,
    win_start: dt.datetime,
    win_end: dt.datetime,
) -> bool:
    """
    Download one day window. Returns True if file was downloaded, False if no data / skipped.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    fname = build_filename(cfg, station, utc_day_floor(win_start))
    fpath = os.path.join(cfg.out_dir, fname)

    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        return False  # already have it

    params = {
        "network": cfg.network,
        "station": station,
        "location": cfg.location,
        "quality": cfg.quality,
        "channel": cfg.channels_csv,
        "starttime": isoformat_utc(win_start),
        "endtime": isoformat_utc(win_end),
        "nodata": 204,
    }

    resp = request_with_retries(
        session,
        "GET",
        DATASELECT_URL,
        params=params,
        timeout_s=cfg.timeout_s,
        retries=cfg.retries,
        backoff_s=cfg.backoff_s,
        stream=True,
    )

    if resp.status_code == 204:
        return False

    resp.raise_for_status()

    # Stream to disk
    tmp_path = fpath + ".part"
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    os.replace(tmp_path, fpath)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", dest="out_dir", default="resif_1F", help="Output directory")
    ap.add_argument("--timeout", type=int, default=120, help="Request timeout (seconds)")
    ap.add_argument("--retries", type=int, default=5, help="Retries on network errors")
    ap.add_argument("--backoff", type=float, default=2.0, help="Backoff base (seconds)")
    ap.add_argument("--estimate_size", type=bool, default=False,
                    help="Estimate total size of dataset before downloading")
    ap.add_argument("--station_prefixes", type=str, nargs="+", default=["A"],
                    help="Station prefixes to include in the download")
    args = ap.parse_args()

    cfg = Config(out_dir=args.out_dir, timeout_s=args.timeout, retries=args.retries, backoff_s=args.backoff)

    stations = station_codes(prefixes=tuple(args.station_prefixes))

    with requests.Session() as session:
        session.headers.update({"User-Agent": cfg.user_agent})

        if args.estimate_size:
            print("\n=== ESTIMATING TOTAL DOWNLOAD SIZE ===")
            total_bytes = 0.0
            for sta in stations:
                est = estimate_station_size(cfg, session, sta)
                if est > 0:
                    print(f"  {sta}: ~{est / 1e6:.1f} MB")
                    total_bytes += est
                else:
                    print(f"  {sta}: no data")
            print("=== TOTAL ESTIMATED DATASET SIZE ===")
            print(
                f"Total: ~{total_bytes / 1e9:.2f} GB ({total_bytes / 1e6:.1f} MB)")
            print("=" * 40)

        print("\n=== STARTING DOWNLOAD ===")
        for idx, sta in enumerate(stations, 1):
            print(f"[{idx:03d}/{len(stations)}] Station {sta}: discovering available time range...")
            rng = get_station_timerange(cfg, session, sta)
            if rng is None:
                print(f"  - No availability info / no data (extent returned 204). Skipping.")
                continue

            start, end = rng
            print(f"  - Available: {start.isoformat()} to {end.isoformat()} (UTC)")

            downloaded = 0
            checked = 0
            for win_start, win_end in iter_utc_days(start, end):
                checked += 1
                try:
                    got = download_day(cfg, session, sta, win_start, win_end)
                except requests.HTTPError as e:
                    # If RESIF rejects too-large requests (e.g., 413), you can further split.
                    print(f"  ! HTTP error for {sta} {win_start}–{win_end}: {e}")
                    continue

                if got:
                    downloaded += 1
                    print(f"    + {build_filename(cfg, sta, utc_day_floor(win_start))}")
            print(f"  - Done {sta}: downloaded {downloaded} files (checked {checked} day-windows).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
