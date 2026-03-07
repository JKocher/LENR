#!/usr/bin/env python3
"""Post-processing for LENR frequency sweep results.

Reads VTU files produced by frequency_sweep.py, computes pressure
amplitude diagnostics, and generates frequency-response plots, axial
profiles, and an enriched CSV.

The Elmer Helmholtz solver outputs a complex-valued pressure field as
two DOFs per node: "pressure wave 1" (real part) and "pressure wave 2"
(imaginary part).  The complex amplitude is sqrt(re^2 + im^2).

Note: the upstream model uses mm for mesh coordinates but SI (m-based)
material properties, so output values are in model units — not Pa.

Requires: numpy, matplotlib, meshio
"""

import argparse
import csv
from glob import glob as globfn
import os
import re
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")
try:
    import meshio
except ImportError:
    sys.exit("meshio is required: pip install meshio")

# Field names written by Elmer's Helmholtz solver (2-DOF complex pressure)
FIELD_REAL = "pressure wave 1"
FIELD_IMAG = "pressure wave 2"

# Input CSV written by frequency_sweep.py — the coupling contract between scripts
SWEEP_CSV = "sweep_summary.csv"

# Axial profile: include mesh nodes within this distance (mm) of the Z-axis.
# 2 mm is ~2% of the mesh diameter — tight enough to approximate the axis
# while capturing enough nodes for a smooth profile.
AXIAL_TOL_MM = 2.0

# Plot resolution (dots per inch)
PLOT_DPI = 150


def find_vtu_files(results_dir):
    """Return sorted list of (freq_khz, vtu_path) from CSV or glob scan."""
    csv_path = os.path.join(results_dir, SWEEP_CSV)
    pairs = []

    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                freq = float(row["freq_khz"])
                vtu = os.path.join(results_dir, row["vtu_file"])
                pairs.append((freq, vtu))
        source = "sweep_summary.csv"
    else:
        vtu_re = re.compile(r"^case_(\d+)kHz\.vtu$")
        pattern = os.path.join(results_dir, "case_*kHz.vtu")
        for path in sorted(globfn(pattern)):
            m = vtu_re.match(os.path.basename(path))
            if not m:
                continue
            pairs.append((float(m.group(1)), path))
        source = "glob scan"

    pairs.sort(key=lambda x: x[0])
    return pairs, source


def analyze_vtu(vtu_path):
    """Read a VTU file and compute pressure amplitude diagnostics.

    The Helmholtz solver stores the complex pressure as two real-valued
    fields (real and imaginary parts).  We compute the complex amplitude
    magnitude: |p| = sqrt(re^2 + im^2).

    Returns dict with keys: max_pressure, mean_pressure, center_pressure,
    max_location (x,y,z), max_distance_from_center, axial_z, axial_mag.
    """
    mesh = meshio.read(vtu_path)
    points = mesh.points

    available = list(mesh.point_data.keys())
    if FIELD_REAL not in mesh.point_data or FIELD_IMAG not in mesh.point_data:
        raise KeyError(
            f"Expected fields '{FIELD_REAL}' and '{FIELD_IMAG}', "
            f"found: {available}"
        )

    # Real and imaginary parts of the complex pressure amplitude
    p_re = mesh.point_data[FIELD_REAL].ravel()
    p_im = mesh.point_data[FIELD_IMAG].ravel()
    amplitude = np.sqrt(p_re**2 + p_im**2)

    max_idx = np.argmax(amplitude)
    center_idx = np.argmin(np.linalg.norm(points, axis=1))

    axial_z, axial_mag = extract_axial_profile(points, amplitude)

    return {
        "max_pressure": amplitude[max_idx],
        "mean_pressure": np.mean(amplitude),
        "center_pressure": amplitude[center_idx],
        "max_location_x": points[max_idx, 0],
        "max_location_y": points[max_idx, 1],
        "max_location_z": points[max_idx, 2],
        "max_distance_from_center": np.linalg.norm(points[max_idx]),
        "axial_z": axial_z,
        "axial_mag": axial_mag,
    }


def extract_axial_profile(points, mag, tol=AXIAL_TOL_MM):
    """Filter points within tol mm of the Z-axis, return (z, mag) sorted by z."""
    r_xy = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    mask = r_xy < tol
    z = points[mask, 2]
    m = mag[mask]
    order = np.argsort(z)
    return z[order], m[order]


def plot_frequency_response(results, output_path):
    """2-panel plot: max/mean pressure (log) and center pressure (linear)."""
    freqs = [r["freq_khz"] for r in results]
    maxp = [r["max_pressure"] for r in results]
    meanp = [r["mean_pressure"] for r in results]
    centerp = [r["center_pressure"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.semilogy(freqs, maxp, "o-", label="Max amplitude", color="#2196F3")
    ax1.semilogy(freqs, meanp, "s-", label="Mean amplitude", color="#FF9800")
    ax1.set_ylabel("Pressure amplitude (model units)")
    ax1.set_title("Frequency Response — LENR Reactor Sweep")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(freqs, centerp, "D-", color="#4CAF50")
    ax2.set_xlabel("Frequency (kHz)")
    ax2.set_ylabel("Center amplitude (model units)")
    ax2.set_title("Pressure Amplitude at Cavity Center (r=0)")
    ax2.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_axial_profiles(results, output_path):
    """Overlay pressure along Z-axis for all frequencies."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.viridis
    n = len(results)
    for i, r in enumerate(results):
        color = cmap(i / max(n - 1, 1))
        ax.plot(r["axial_z"], r["axial_mag"],
                label=f'{r["freq_khz"]:.0f} kHz', color=color, alpha=0.8)

    ax.set_xlabel("Z position (mm)")
    ax.set_ylabel("Pressure amplitude (model units)")
    ax.set_title("Axial Pressure Amplitude (horn-to-horn)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def write_csv(results, input_csv, output_path):
    """Merge original CSV columns with computed diagnostics."""
    # Read original rows if available
    original_rows = {}
    if input_csv and os.path.isfile(input_csv):
        with open(input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            original_fields = list(reader.fieldnames)
            for row in reader:
                original_rows[float(row["freq_khz"])] = row
    else:
        original_fields = ["freq_khz"]

    extra_fields = [
        "max_pressure", "mean_pressure", "center_pressure",
        "max_location_x", "max_location_y", "max_location_z",
        "max_distance_from_center",
    ]
    all_fields = original_fields + extra_fields

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for r in results:
            freq = r["freq_khz"]
            if freq in original_rows:
                row = dict(original_rows[freq])
            else:
                row = {"freq_khz": freq}
            row["max_pressure"] = f'{r["max_pressure"]:.6g}'
            row["mean_pressure"] = f'{r["mean_pressure"]:.6g}'
            row["center_pressure"] = f'{r["center_pressure"]:.4g}'
            row["max_location_x"] = f'{r["max_location_x"]:.5g}'
            row["max_location_y"] = f'{r["max_location_y"]:.5g}'
            row["max_location_z"] = f'{r["max_location_z"]:.5g}'
            row["max_distance_from_center"] = f'{r["max_distance_from_center"]:.5g}'
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze frequency sweep VTU results from ElmerSolver."
    )
    parser.add_argument(
        "results_dir", nargs="?", default="sweep_results",
        help="directory containing VTU files (default: sweep_results/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir

    if not os.path.isdir(results_dir):
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Sweep Analysis: {results_dir}/")

    pairs, source = find_vtu_files(results_dir)
    if not pairs:
        print(f"  No VTU files found in {results_dir}/")
        sys.exit(1)

    print(f"  Found {len(pairs)} VTU files (from {source})\n")

    results = []
    for i, (freq, vtu_path) in enumerate(pairs, 1):
        if not os.path.isfile(vtu_path):
            print(f"  [{i}/{len(pairs)}]  {freq:.0f} kHz  WARNING: {vtu_path} not found, skipping")
            continue
        try:
            data = analyze_vtu(vtu_path)
        except Exception as e:
            print(f"  [{i}/{len(pairs)}]  {freq:.0f} kHz  WARNING: {e}, skipping")
            continue

        data["freq_khz"] = freq
        results.append(data)
        print(f"  [{i}/{len(pairs)}]  {freq:.0f} kHz  "
              f"max={data['max_pressure']:.1f}  "
              f"mean={data['mean_pressure']:.3f}  "
              f"center={data['center_pressure']:.1e}")

    if not results:
        print("\nError: all VTU files failed to process")
        sys.exit(1)

    # Generate outputs
    analysis_png = os.path.join(results_dir, "sweep_analysis.png")
    axial_png = os.path.join(results_dir, "axial_profiles.png")
    analysis_csv = os.path.join(results_dir, "sweep_analysis.csv")
    input_csv = os.path.join(results_dir, SWEEP_CSV)

    plot_frequency_response(results, analysis_png)
    plot_axial_profiles(results, axial_png)
    write_csv(results, input_csv, analysis_csv)

    print(f"\n{'=' * 60}")
    print(f"  Analysis complete: {len(results)} frequencies processed")
    print(f"{'=' * 60}")
    print(f"  Frequency response: {analysis_png}")
    print(f"  Axial profiles:     {axial_png}")
    print(f"  Enriched CSV:       {analysis_csv}")


if __name__ == "__main__":
    main()
