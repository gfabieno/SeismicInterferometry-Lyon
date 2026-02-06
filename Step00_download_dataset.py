#!/usr/bin/env python3
"""
Simple RESIF ObsPy downloader (parallel + resume + metadata-only size estimate)
+ End summary table per station: start/end, gaps?, size, position, file count.

Install:
  pip install obspy

Run:
  python resif_simple_obspy_downloader.py --out data --workers 8
  python resif_simple_obspy_downloader.py --out data --estimate-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional, Tuple, List, Dict, Set

from obspy import UTCDateTime, read_inventory
from obspy.clients.fdsn import Client
from utils import mseed_path


# --- Size model (metadata-only) ---
DEFAULT_BYTES_PER_SAMPLE = 4.0   # rough (e.g., 32-bit/sample)
DEFAULT_OVERHEAD_FACTOR = 1.2    # headers/record padding


def station_codes() -> List[str]:
    return [f"A{i:03d}" for i in range(1, 100)] #+ [f"B{i:03d}" for i in range(1, 100)]


def utc_midnight(day: dt.date) -> UTCDateTime:
    return UTCDateTime(dt.datetime(day.year, day.month, day.day))


def iter_utc_days(start: UTCDateTime, end: UTCDateTime) -> Iterable[Tuple[UTCDateTime, UTCDateTime]]:
    """Yield day windows clipped to [start, end)."""
    if end <= start:
        return
    cur = utc_midnight(start.date)
    while cur < end:
        nxt = cur + 86400
        win_start = max(start, cur)
        win_end = min(end, nxt)
        if win_end > win_start:
            yield win_start, win_end
        cur = nxt


def iter_dates_inclusive(d0: dt.date, d1: dt.date) -> Iterable[dt.date]:
    """Inclusive date range: d0..d1."""
    cur = d0
    while cur <= d1:
        yield cur
        cur += dt.timedelta(days=1)


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stationxml_path(out_root: str, network: str, station: str) -> str:
    return os.path.join(out_root, "metadata", f"{network}.{station}.station.xml")


def station_waveform_dir(out_root: str, network: str, station: str) -> str:
    return os.path.join(out_root, "waveforms", network, station)


def retry(fn, *, retries: int, backoff: float, jitter: float = 0.25):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            sleep_s = backoff * (2 ** attempt) + random.random() * jitter
            time.sleep(sleep_s)
    raise last  # type: ignore


def get_inventory_channel_level(client: Client, network: str, station: str,
                                location: str, channels: str, retries: int):
    def _call():
        return client.get_stations(
            network=network,
            station=station,
            location=location,
            channel=channels,
            level="channel",
        )
    return retry(_call, retries=retries, backoff=1.5)


def get_time_range_from_inv(inv) -> Optional[Tuple[UTCDateTime, UTCDateTime]]:
    starts = []
    ends = []
    for net in inv:
        for sta in net:
            for cha in sta:
                if cha.start_date:
                    starts.append(cha.start_date)
                if cha.end_date:
                    ends.append(cha.end_date)
    if not starts:
        return None
    start = min(starts)
    end = max(ends) if ends else UTCDateTime()  # ongoing -> now
    if end <= start:
        return None
    return start, end


def get_sample_rates_from_inv(inv) -> Dict[str, float]:
    """MAX sample_rate per channel code across epochs (conservative)."""
    rates: Dict[str, float] = {}
    for net in inv:
        for sta in net:
            for cha in sta:
                code = cha.code
                sr = float(cha.sample_rate or 0.0)
                if code not in rates or sr > rates[code]:
                    rates[code] = sr
    return rates


def estimate_station_bytes_metadata_only(
    inv,
    channel_list: List[str],
    start: UTCDateTime,
    end: UTCDateTime,
    bytes_per_sample: float,
    overhead_factor: float,
) -> Tuple[float, Dict[str, float]]:
    rates = get_sample_rates_from_inv(inv)
    sum_sr = sum(rates.get(ch, 0.0) for ch in channel_list)
    duration_s = float(end - start)
    est = duration_s * sum_sr * bytes_per_sample * overhead_factor
    return est, rates


def ensure_stationxml(client: Client, out_root: str, network: str, station: str,
                      location: str, channels: str, retries: int) -> None:
    """Download StationXML once per station (skip if exists)."""
    path = stationxml_path(out_root, network, station)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    safe_makedirs(os.path.dirname(path))

    def _call():
        return client.get_stations(
            network=network,
            station=station,
            location=location,
            channel=channels,
            level="response",
        )

    inv = retry(_call, retries=retries, backoff=1.5)
    tmp = path + ".part"
    inv.write(tmp, format="STATIONXML")
    os.replace(tmp, path)


def download_one_day(client: Client, out_root: str, network: str, station: str,
                     location: str, channels: str,
                     win_start: UTCDateTime, win_end: UTCDateTime, retries: int) -> str:
    """Download one day window (skip if exists). Returns: done|nodata|skip|error"""
    day = win_start.date
    out = mseed_path(out_root, network, location, station, channels, day)
    safe_makedirs(os.path.dirname(out))

    if os.path.exists(out) and os.path.getsize(out) > 0:
        return "skip"

    def _call():
        return client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channels,
            starttime=win_start,
            endtime=win_end,
        )

    try:
        st = retry(_call, retries=retries, backoff=1.5)
        if len(st) == 0:
            return "nodata"
        tmp = out + ".part"
        st.write(tmp, format="MSEED")
        os.replace(tmp, out)
        return "done"
    except Exception:
        try:
            if os.path.exists(out + ".part"):
                os.remove(out + ".part")
        except Exception:
            pass
        return "error"


def format_gb(nbytes: float) -> str:
    return f"{nbytes/1e9:.3f} GB"


def list_station_mseed_files(out_root: str, network: str, station: str) -> List[str]:
    root = station_waveform_dir(out_root, network, station)
    files = []
    if not os.path.isdir(root):
        return files
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".mseed"):
                files.append(os.path.join(dirpath, fn))
    return files


def parse_date_from_filename(path: str) -> Optional[dt.date]:
    # expected ...YYYYMMDD.mseed
    base = os.path.basename(path)
    if len(base) < 8:
        return None
    try:
        yyyymmdd = base.split(".")[-2]  # last token before mseed is YYYYMMDD
        if len(yyyymmdd) != 8 or not yyyymmdd.isdigit():
            return None
        y = int(yyyymmdd[0:4]); m = int(yyyymmdd[4:6]); d = int(yyyymmdd[6:8])
        return dt.date(y, m, d)
    except Exception:
        return None


def get_position_from_stationxml(xml_path: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        inv = read_inventory(xml_path)
        net = inv[0]
        sta = net[0]
        return float(sta.latitude), float(sta.longitude), float(sta.elevation)
    except Exception:
        return None, None, None


def print_summary_table(rows: List[Dict[str, object]]) -> None:
    # plain-text table (no extra deps)
    headers = ["Station", "Start", "End", "Gaps?", "MissingDays", "Files", "SizeActual", "SizeEst", "Lat", "Lon", "Elev(m)"]
    # compute column widths
    def s(x): return "" if x is None else str(x)
    table = [headers] + [[
        s(r.get("station")), s(r.get("start")), s(r.get("end")), s(r.get("gaps")),
        s(r.get("missing_days")), s(r.get("files")), s(r.get("size_actual")),
        s(r.get("size_est")), s(r.get("lat")), s(r.get("lon")), s(r.get("elev"))
    ] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]

    def fmt_row(row):
        return " | ".join(row[i].ljust(widths[i]) for i in range(len(headers)))

    print("\n=== Dataset summary by station ===")
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for r in table[1:]:
        print(fmt_row(r))
    print("=================================\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="Output folder root (default: data)")
    ap.add_argument("--provider", default="RESIF", help="FDSN provider (default: RESIF)")
    ap.add_argument("--network", default="1F", help=f"Network code (default: 1F)")
    ap.add_argument("--location", default="00", help=f"Location code (default: 00)")
    ap.add_argument("--channels", default="DPZ,DPN,DPE", help=f"Channel codes comma-separated (default: DPE,DPN,DPZ)")
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    ap.add_argument("--retries", type=int, default=5, help="Retries per request (default: 5)")
    ap.add_argument("--estimate-only", action="store_true", help="Only estimate disk usage and exit")
    ap.add_argument("--bytes-per-sample", type=float, default=DEFAULT_BYTES_PER_SAMPLE,
                    help=f"Estimation model bytes/sample (default: {DEFAULT_BYTES_PER_SAMPLE})")
    ap.add_argument("--overhead", type=float, default=DEFAULT_OVERHEAD_FACTOR,
                    help=f"Estimation overhead factor (default: {DEFAULT_OVERHEAD_FACTOR})")
    args = ap.parse_args()

    out_root = args.out
    safe_makedirs(out_root)

    stations = station_codes()
    client = Client(args.provider)
    network = args.network
    location = args.location
    channels = args.channels
    channel_list = channels.split(",")

    print(f"Output root: {out_root}")
    print(f"FDSN Provider: {args.provider}")
    print(f"Network: {network}, Location: {location}, Channels: {channels}")
    print(f"Stations to process: {len(stations)}")

    # 1) Discover ranges + metadata-only estimate
    print("=== Discovering availability & estimating total size (metadata-only) ===")
    print(f"Model: bytes_per_sample={args.bytes_per_sample}, overhead_factor={args.overhead}\n")

    ranges: dict[str, Tuple[UTCDateTime, UTCDateTime]] = {}
    est_bytes_by_station: dict[str, float] = {}

    total_est = 0.0

    for i, sta in enumerate(stations, 1):
        print(f"[{i:03d}/{len(stations)}] {sta}: ", end="", flush=True)
        try:
            inv = get_inventory_channel_level(client, network, sta, location,
                                              channels, retries=args.retries)
            rng = get_time_range_from_inv(inv)
            if rng is None:
                print("no data")
                continue
            start, end = rng
            ranges[sta] = (start, end)

            est_bytes, rates = estimate_station_bytes_metadata_only(
                inv, channel_list, start, end,
                bytes_per_sample=args.bytes_per_sample,
                overhead_factor=args.overhead
            )
            est_bytes_by_station[sta] = est_bytes
            total_est += est_bytes

            rates_str = ", ".join(f"{ch}:{rates.get(ch, 0.0):g}Hz" for ch in channel_list)
            days = float(end - start) / 86400.0
            print(f"{rates_str} | ~{days:.1f} days | ~{format_gb(est_bytes)}")

        except Exception as e:
            print(f"failed ({e})")

    print(f"\nEstimated TOTAL: ~{format_gb(total_est)}  (~{total_est/1e12:.3f} TB)")
    print("=============================================\n")

    if args.estimate_only:
        return 0

    # 2) StationXML (sequential, simple)
    print("=== Writing StationXML ===")
    for i, sta in enumerate(stations, 1):
        try:
            ensure_stationxml(client, out_root, network, sta, location, channels,
                              retries=args.retries)
            print(f"[{i:03d}/{len(stations)}] {sta}: ok")
        except Exception as e:
            print(f"[{i:03d}/{len(stations)}] {sta}: failed ({e})")

    # 3) Build day tasks
    tasks: List[Tuple[str, UTCDateTime, UTCDateTime]] = []
    for sta, (start, end) in ranges.items():
        for ws, we in iter_utc_days(start, end):
            tasks.append((sta, ws, we))

    print(f"\n=== Downloading daily MiniSEED (parallel={args.workers}) ===")
    print(f"Total day tasks: {len(tasks)}")
    if not tasks:
        print("Nothing to download.")
        return 0

    # 4) Parallel download (one Client per worker call)
    def worker(task):
        sta, ws, we = task
        local_client = Client(args.provider)
        status = download_one_day(local_client, out_root, network, sta, location,
                                  channels, ws, we, retries=args.retries)
        return sta, ws.date, status

    done = skip = nodata = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futures):
            sta, day, status = fut.result()
            if status == "done":
                done += 1
                print(f"+ {sta} {day.isoformat()}")
            elif status == "skip":
                skip += 1
            elif status == "nodata":
                nodata += 1
            else:
                err += 1
                print(f"! {sta} {day.isoformat()} (error)")

    print("\n=== Download Summary ===")
    print(f"done:   {done}")
    print(f"skip:   {skip} (already existed)")
    print(f"nodata: {nodata}")
    print(f"errors: {err}")
    print(f"Output: {out_root}")

    # 5) End summary table per station (based on files on disk)
    rows: List[Dict[str, object]] = []
    for sta, (start, end) in ranges.items():
        files = list_station_mseed_files(out_root, network, sta)
        size_actual = sum(os.path.getsize(f) for f in files) if files else 0

        # infer gaps from missing days between start and end
        # expected days: start.date .. end.date (inclusive) BUT if end is exactly at midnight,
        # the last day might have no window. We'll use end_minus_epsilon for robustness.
        end_for_days = end - 1e-6
        expected_dates: Set[dt.date] = set(iter_dates_inclusive(start.date, end_for_days.date))

        have_dates: Set[dt.date] = set()
        for f in files:
            d = parse_date_from_filename(f)
            if d:
                have_dates.add(d)

        missing = sorted(expected_dates - have_dates)
        gaps = "YES" if len(missing) > 0 else "NO"

        xmlp = stationxml_path(out_root, network, sta)
        lat, lon, elev = get_position_from_stationxml(xmlp) if os.path.exists(xmlp) else (None, None, None)

        rows.append({
            "station": sta,
            "start": str(start.date),
            "end": str(end.date),
            "gaps": gaps,
            "missing_days": len(missing),
            "files": len(files),
            "size_actual": format_gb(float(size_actual)),
            "size_est": format_gb(float(est_bytes_by_station.get(sta, 0.0))),
            "lat": f"{lat:.5f}" if lat is not None else "",
            "lon": f"{lon:.5f}" if lon is not None else "",
            "elev": f"{elev:.1f}" if elev is not None else "",
        })

    # sort for readability
    rows.sort(key=lambda r: r["station"])  # type: ignore
    print_summary_table(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
