#!/usr/bin/env python3
"""
STEP 0 — Inventory & metadata check + station-day index + coverage report
(Updated for folder layout with YEAR subfolders)

Folder layout:
  waveforms/1F/A001/2018/*.mseed
  waveforms/1F/A001/2019/*.mseed
  ...

Filename convention example:
  1F.A001.00.DPZ-DPN-DPE.M.20180914.mseed

Outputs (in --outdir):
- station_day_index.csv
- station_day_coverage.csv
- station_day_coverage_heatmap.png
- stations_missing.csv
- files_inventory.csv
- stations_metadata_validated.csv
- sampling_rate_hist.png
- sensors_latlon.png / sensors_projected.png (if StationXML provides coords)
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from obspy import read, UTCDateTime, read_inventory
from obspy.core.stream import Stream

from utils import inv_get_coords, discover_mseed_files, stations_from_folders

# -----------------------------
# Parsing / expectations
# -----------------------------
FNAME_RE = re.compile(
    r"""
    ^
    (?P<net>[A-Z0-9]{1,2})\.
    (?P<sta>[A-Z0-9]{3,5})\.
    (?P<loc>[A-Z0-9]{2})\.
    (?P<comps>[A-Z0-9\-]+)\.
    (?P<date>\d{8})
    \.mseed
    $
    """,
    re.VERBOSE
)

EXPECTED_STATIONS = [f"A{i:03d}" for i in range(1, 100)]  # A001..A099


@dataclass
class FileKey:
    net: str
    sta: str
    loc: str
    comps: str
    date: str

    @property
    def station_id(self) -> str:
        return f"{self.net}.{self.sta}.{self.loc}"


def parse_filename(p: Path) -> Optional[FileKey]:
    m = FNAME_RE.match(p.name)
    if not m:
        return None
    return FileKey(
        net=m.group("net"),
        sta=m.group("sta"),
        loc=m.group("loc"),
        comps=m.group("comps"),
        date=m.group("date"),
    )


def report_missing_stations_from_folders(station_folders: List[str],
                                         expected: List[str]) -> List[str]:
    """
    Folder-driven missing station list (robust for partial downloads).
    """
    return sorted(set(expected) - set(station_folders))


# -----------------------------
# QC helpers
# -----------------------------
def stream_header_qc(st: Stream) -> Dict[str, object]:
    starts, ends, srs, npts, chans = [], [], [], [], []
    for tr in st:
        starts.append(tr.stats.starttime)
        ends.append(tr.stats.endtime)
        srs.append(float(tr.stats.sampling_rate))
        npts.append(int(tr.stats.npts))
        chans.append(tr.stats.channel)

    out = {
        "n_traces": len(st),
        "start_time_min": min(starts).datetime.isoformat() if starts else None,
        "end_time_max": max(ends).datetime.isoformat() if ends else None,
        "sampling_rate_median": float(np.median(srs)) if srs else None,
        "sampling_rate_min": float(np.min(srs)) if srs else None,
        "sampling_rate_max": float(np.max(srs)) if srs else None,
        "channels": ",".join(sorted(set(chans))) if chans else "",
        "npts_total": int(np.sum(npts)) if npts else 0,
    }

    try:
        st2 = st.copy()
        st2.sort()
        st2.merge(method=0, fill_value=None)  # no interpolation
        gaps = st2.get_gaps()
        out["num_gaps"] = len(gaps)
        if gaps:
            gap_secs = [float(g[6]) for g in gaps]
            out["max_gap_s"] = float(np.max(gap_secs))
            out["gap_s_total"] = float(np.sum(gap_secs))
        else:
            out["max_gap_s"] = 0.0
            out["gap_s_total"] = 0.0
    except Exception:
        out["num_gaps"] = None
        out["max_gap_s"] = None
        out["gap_s_total"] = None

    return out


def make_maps(df_stations: pd.DataFrame, outdir: Path) -> None:
    dfc = df_stations.dropna(subset=["latitude", "longitude"]).copy()
    if dfc.empty:
        print("[WARN] No coordinates available; skipping maps.")
        return

    # Lat/Lon
    plt.figure()
    plt.scatter(dfc["longitude"], dfc["latitude"], s=12)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Sensor locations (lat/lon)")
    plt.tight_layout()
    plt.savefig(outdir / "sensors_latlon.png", dpi=200)
    plt.close()

    # Projected (UTM) if pyproj available
    try:
        from pyproj import CRS, Transformer

        lon0 = float(dfc["longitude"].mean())
        utm_zone = int(np.floor((lon0 + 180) / 6) + 1)
        lat0 = float(dfc["latitude"].mean())
        epsg = 32600 + utm_zone if lat0 >= 0 else 32700 + utm_zone

        transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
        xs, ys = transformer.transform(dfc["longitude"].to_numpy(), dfc["latitude"].to_numpy())

        plt.figure()
        plt.scatter(xs, ys, s=12)
        plt.xlabel(f"Easting (m) — UTM zone {utm_zone} (EPSG:{epsg})")
        plt.ylabel("Northing (m)")
        plt.title("Sensor locations (projected UTM)")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(outdir / "sensors_projected.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"[WARN] Projected map skipped (pyproj missing or error): {e}")


# -----------------------------
# Index + coverage products
# -----------------------------
def build_station_day_index(files: List[Path],
                            station_folders: Optional[List[str]] = None
                            ) -> Tuple[Dict[Tuple[str, str], Path], pd.DataFrame]:
    """
    Returns:
      - dict: (sid, yyyymmdd) -> path  (if duplicates exist, last wins; flagged)
      - dataframe with one row per file: station_id, date, path, duplicate_flag, station_folder_present
    """
    index: Dict[Tuple[str, str], Path] = {}
    seen: Dict[Tuple[str, str], int] = {}
    rows = []
    station_folder_set = set(station_folders) if station_folders else None

    for p in files:
        key = parse_filename(p)
        if key is None:
            continue

        sid = key.station_id
        day = key.date
        k = (sid, day)
        seen[k] = seen.get(k, 0) + 1
        duplicate_flag = seen[k] > 1

        # keep latest encountered path (files sorted -> deterministic)
        index[k] = p

        rows.append({
            "station_id": sid,
            "network": key.net,
            "station": key.sta,
            "location": key.loc,
            "date_yyyymmdd": day,
            "path": str(p),
            "duplicate_flag": bool(duplicate_flag),
            "station_folder_present": (
                (key.sta in station_folder_set) if station_folder_set is not None else None
            ),
        })

    df_index = pd.DataFrame(rows).sort_values(["station_id", "date_yyyymmdd"])
    return index, df_index


def make_coverage_table(df_index: pd.DataFrame) -> pd.DataFrame:
    """
    station x day coverage table with 1 if present, 0 if missing,
    using global min/max day found.
    """
    if df_index.empty:
        return pd.DataFrame()

    days = sorted(df_index["date_yyyymmdd"].unique())
    d0, d1 = days[0], days[-1]
    all_days = pd.date_range(pd.to_datetime(d0), pd.to_datetime(d1), freq="D")
    all_days_str = [d.strftime("%Y%m%d") for d in all_days]

    stations = sorted(df_index["station_id"].unique())
    present = set(zip(df_index["station_id"], df_index["date_yyyymmdd"]))

    mat = np.zeros((len(stations), len(all_days_str)), dtype=np.int8)
    for i, sid in enumerate(stations):
        for j, day in enumerate(all_days_str):
            mat[i, j] = 1 if (sid, day) in present else 0

    cov = pd.DataFrame(mat, index=stations, columns=all_days_str)
    cov.index.name = "station_id"
    return cov.reset_index()


def plot_coverage_heatmap(cov_df: pd.DataFrame, outdir: Path, max_days: int = 180) -> None:
    """
    Quicklook heatmap. If span is huge, limit to first max_days columns for readability.
    """
    if cov_df.empty:
        return

    station_ids = cov_df["station_id"].to_list()
    day_cols = [c for c in cov_df.columns if c != "station_id"]
    if len(day_cols) > max_days:
        day_cols = day_cols[:max_days]

    Z = cov_df[day_cols].to_numpy()

    plt.figure(figsize=(max(8, len(day_cols) / 8), max(6, len(station_ids) / 12)))
    plt.imshow(Z, aspect="auto", interpolation="nearest")
    plt.yticks(np.arange(len(station_ids)), station_ids, fontsize=6)
    xt = np.linspace(0, len(day_cols) - 1, num=min(10, len(day_cols))).astype(int)
    plt.xticks(xt, [day_cols[i] for i in xt], rotation=45, ha="right", fontsize=7)
    plt.xlabel("Day (YYYYMMDD)")
    plt.ylabel("Station")
    plt.title("Station-day coverage (1=present, 0=missing) — quicklook")
    plt.tight_layout()
    plt.savefig(outdir / "station_day_coverage_heatmap.png", dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", type=Path,
                    help="Root folder, defautlt: data/")
    ap.add_argument("--network", default="1F", type=str,
                    help="Network folder under data-root (default: 1F)")
    ap.add_argument("--outdir", default="outputs/Step01_check_inventory", type=Path)
    ap.add_argument("--fs-expected", default=250.0, type=float)
    ap.add_argument("--fs-tol", default=0.1, type=float)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    inv_dir = args.data_root / "metadata"
    waveforms_dir = args.data_root / "waveforms"
    inv = read_inventory(str(inv_dir)+"/*", format="STATIONXML")

    # Folder-driven station discovery + missing station reporting
    station_folders = stations_from_folders(args.data_root, args.network)
    missing = report_missing_stations_from_folders(station_folders, EXPECTED_STATIONS)
    pd.DataFrame({"missing_station": missing}).to_csv(args.outdir / "stations_missing.csv", index=False)
    print(f"[INFO] Station folders found under {waveforms_dir}/{args.network}: {len(station_folders)}")
    print(f"[INFO] Missing stations (expected A001..A099): {len(missing)}")
    if missing:
        print("       " + ", ".join(missing))

    # File discovery (fast glob with YEAR subfolders)
    files = discover_mseed_files(args.data_root, args.network)
    print(f"[INFO] Found {len(files)} *.mseed files under {waveforms_dir}/{args.network}/<STA>/<YEAR>/")

    # Build station-day index & coverage products
    station_day_index, df_index = build_station_day_index(files, station_folders=station_folders)
    df_index.to_csv(args.outdir / "station_day_index.csv", index=False)
    print(f"[INFO] Wrote {args.outdir/'station_day_index.csv'} ({len(df_index)} rows)")

    cov = make_coverage_table(df_index)
    cov.to_csv(args.outdir / "station_day_coverage.csv", index=False)
    print(f"[INFO] Wrote {args.outdir/'station_day_coverage.csv'}")
    plot_coverage_heatmap(cov, args.outdir)

    # File-level inventory with ObsPy header QC + station summary
    rows_files = []
    station_summary: Dict[str, Dict[str, object]] = {}

    for p in files:

        key = parse_filename(p)
        if key is None:
            continue

        qc = {}
        read_error = None
        try:
            st = read(str(p), headonly=True)
            qc = stream_header_qc(st)
        except Exception as e:
            read_error = str(e)

        rows_files.append({
            "path": str(p),
            "net": key.net,
            "station": key.sta,
            "loc": key.loc,
            "components": key.comps,
            "date": key.date,
            "read_error": read_error,
            **qc
        })

        sid = key.station_id
        if sid not in station_summary:
            lat = lon = elev = None
            if inv is not None:
                lat, lon, elev = inv_get_coords(inv, key.net, key.sta)
            station_summary[sid] = {
                "network": key.net,
                "station": key.sta,
                "location": key.loc,
                "latitude": lat,
                "longitude": lon,
                "elevation_m": elev,
                "files_count": 0,
                "start_time": None,
                "end_time": None,
                "channels_union": set(),
                "fs_medians": [],
                "num_gaps_total": 0,
                "gap_s_total": 0.0,
                "max_gap_s": 0.0,
                "read_errors": 0,
            }

        ss = station_summary[sid]
        ss["files_count"] += 1

        if read_error:
            ss["read_errors"] += 1
            continue

        try:
            if qc.get("start_time_min"):
                t0 = UTCDateTime(qc["start_time_min"])
                ss["start_time"] = t0 if ss["start_time"] is None else min(ss["start_time"], t0)
            if qc.get("end_time_max"):
                t1 = UTCDateTime(qc["end_time_max"])
                ss["end_time"] = t1 if ss["end_time"] is None else max(ss["end_time"], t1)
        except Exception:
            pass

        if qc.get("channels"):
            ss["channels_union"].update([c.strip() for c in str(qc["channels"]).split(",") if c.strip()])
        if qc.get("sampling_rate_median") is not None:
            ss["fs_medians"].append(float(qc["sampling_rate_median"]))

        if qc.get("num_gaps") is not None:
            ss["num_gaps_total"] += int(qc["num_gaps"])
        if qc.get("gap_s_total") is not None:
            ss["gap_s_total"] += float(qc["gap_s_total"])
        if qc.get("max_gap_s") is not None:
            ss["max_gap_s"] = max(ss["max_gap_s"], float(qc["max_gap_s"]))

    df_files = pd.DataFrame(rows_files)
    df_files.to_csv(args.outdir / "files_inventory.csv", index=False)
    print(f"[INFO] Wrote {args.outdir/'files_inventory.csv'}")

    # Station metadata table + flags
    station_rows = []
    for sid, ss in sorted(station_summary.items()):
        fs_med = float(np.median(ss["fs_medians"])) if ss["fs_medians"] else None
        fs_min = float(np.min(ss["fs_medians"])) if ss["fs_medians"] else None
        fs_max = float(np.max(ss["fs_medians"])) if ss["fs_medians"] else None

        # rough 3C check: tighten later if needed
        flag_missing_3c = (len(ss["channels_union"]) < 3)
        flag_bad_fs = (fs_med is not None) and (abs(fs_med - args.fs_expected) > args.fs_tol)

        station_rows.append({
            "station_id": sid,
            "network": ss["network"],
            "station": ss["station"],
            "location": ss["location"],
            "latitude": ss["latitude"],
            "longitude": ss["longitude"],
            "elevation_m": ss["elevation_m"],
            "files_count": ss["files_count"],
            "start_time": ss["start_time"].datetime.isoformat() if ss["start_time"] else None,
            "end_time": ss["end_time"].datetime.isoformat() if ss["end_time"] else None,
            "channels": ",".join(sorted(ss["channels_union"])),
            "fs_median": fs_med,
            "fs_min": fs_min,
            "fs_max": fs_max,
            "num_gaps_total": ss["num_gaps_total"],
            "gap_s_total": ss["gap_s_total"],
            "max_gap_s": ss["max_gap_s"],
            "read_errors": ss["read_errors"],
            "flag_missing_3c": bool(flag_missing_3c),
            "flag_bad_fs": bool(flag_bad_fs),
        })

    df_sta = pd.DataFrame(station_rows)
    df_sta.to_csv(args.outdir / "stations_metadata_validated.csv", index=False)
    print(f"[INFO] Wrote {args.outdir/'stations_metadata_validated.csv'}")

    # Sampling rate histogram
    if df_sta["fs_median"].notna().any():
        plt.figure()
        plt.hist(df_sta["fs_median"].dropna().values, bins=30)
        plt.xlabel("Median sampling rate per station (Hz)")
        plt.ylabel("Count")
        plt.title("Sampling rate distribution (Step 0)")
        plt.tight_layout()
        plt.savefig(args.outdir / "sampling_rate_hist.png", dpi=200)
        plt.close()

    # Maps if coords exist
    make_maps(df_sta, args.outdir)

    # Duplicate station-day check
    n_dups = int(df_index["duplicate_flag"].sum()) if not df_index.empty else 0
    if n_dups:
        print(f"[WARN] Found {n_dups} duplicate station-day files (see station_day_index.csv duplicate_flag)")

    print("[DONE] Step 0 complete.")
    print("       Key outputs:")
    print("       - stations_missing.csv (folder-based)")
    print("       - station_day_index.csv, station_day_coverage.csv (+ heatmap)")
    print("       - files_inventory.csv, stations_metadata_validated.csv")


if __name__ == "__main__":
    main()
