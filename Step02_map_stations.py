#!/usr/bin/env python3
"""Step02_map_stations.py

Goal
----
Build a map of seismic stations from a StationXML inventory and display
their locations in local Cartesian coordinates relative to a reference
point.

Features
--------
- Read an ObsPy StationXML inventory (or a metadata folder) and compute
  local XY coordinates for stations.
- Produce a simple annotated map for quick inspection.

Output
------
Files are written to the directory specified by `--outdir` (default:
`outputs/Step02_map_stations`). The script currently writes:

- station_map_local_xy.png : annotated station map in local Cartesian
  coordinates (X, Y) centered on the network.

If additional map types are produced in the future (lat/lon scatter,
projected coordinates), they will also be written under `--outdir`.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple



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
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--inventory", default="data/metadata", type=Path,
                    help="Optional StationXML for coordinates / response")
    ap.add_argument("--outdir", default="outputs/Step02_map_stations",
                    type=Path)
    ap.add_argument("--show-doc", action="store_true",
                    help="Print the module documentation and exit")
    args = ap.parse_args()

    if args.show_doc:
        print(__doc__)
        return 0

    # Defer heavy imports until after handling --show-doc
    from obspy import read_inventory
    from utils import inv_station_xy

    args.outdir.mkdir(parents=True, exist_ok=True)
    inv = read_inventory(str(args.inventory) + "/*", format="STATIONXML")
    outfile_xy = args.outdir / "station_map_local_xy.png"

    inv_station_xy(inv, outfile_xy, annotate=True)


if __name__ == "__main__":
    main()