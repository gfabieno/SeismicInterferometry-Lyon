#!/usr/bin/env python3
"""
STEP 2 - Build a map of the seismic stations

This script reads seismic station metadata from an inventory file and maps their locations
relative to a reference point using local Cartesian coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from obspy import read_inventory
import argparse
from utils import inv_station_xy


def _iter_station_coords_from_inventory(inv) -> list[Tuple[str, str, str, float, float]]:
    """
    Return list of (net, sta, loc, lat, lon) for all stations in an ObsPy Inventory.
    loc is not always meaningful at station-level; kept as '-' for labeling simplicity.
    """
    rows: list[Tuple[str, str, str, float, float]] = []
    for net in inv:
        for sta in net.stations:
            lat = getattr(sta, "latitude", None)
            lon = getattr(sta, "longitude", None)
            if lat is None or lon is None:
                continue
            try:
                rows.append((net.code, sta.code, "-", float(lat), float(lon)))
            except Exception:
                continue
    return rows


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="data/metadata", type=Path,
                    help="Optional StationXML for coordinates / response")
    ap.add_argument("--outdir", default="outputs/Step02_map_stations",
                    type=Path)
    args = ap.parse_args()
    inv = read_inventory(str(args.inventory)+"/*", format="STATIONXML")
    outfile_xy = args.outdir / "station_map_local_xy.png"

    inv_station_xy(inv, outfile_xy, annotate=True)


if __name__ == "__main__":
    main()