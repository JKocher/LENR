"""
Elmer FEM frequency sweep for the LENR sonoluminescence reactor.

Automates what was previously a manual process:
  1. Edit angular frequency in case.sif
  2. Run ElmerSolver
  3. Rename the output .vtu file
  4. Repeat for each frequency

Usage:
  python frequency_sweep.py                     # default 20-110 kHz in 10kHz steps
  python frequency_sweep.py 20 200 10           # custom: start_kHz end_kHz step_kHz
  python frequency_sweep.py 40 40               # single frequency

Output:
  sweep_results/case_{freq}kHz.vtu    Result file per frequency
  sweep_results/sweep_summary.csv     Tabular summary for downstream analysis

Environment:
  ELMER_SOLVER    Path to ElmerSolver executable (auto-detected if on PATH)
"""

import sys
import os
import re
import subprocess
import math
import shutil
import csv

SIF_FILE = "case.sif"
OUTPUT_DIR = "sweep_results"
VTU_OUTPUT = "case_t0001.vtu"


def find_elmer_solver():
    """Find ElmerSolver executable, checking env var then PATH."""
    env_path = os.environ.get("ELMER_SOLVER")
    if env_path:
        if os.path.isfile(env_path):
            return env_path
        print(f"Warning: ELMER_SOLVER set to '{env_path}' but file not found.")

    found = shutil.which("ElmerSolver")
    if found:
        return found

    print("Error: ElmerSolver not found.")
    print("  Install Elmer FEM (https://www.elmerfem.org/blog/binaries/)")
    print("  and ensure ElmerSolver is on your PATH, or set the ELMER_SOLVER")
    print("  environment variable to the full path of the executable.")
    sys.exit(1)


def set_angular_frequency(sif_path, angular_freq):
    """Replace the Angular Frequency value in the SIF file."""
    with open(sif_path, "r") as f:
        content = f.read()

    content = re.sub(
        r"(Angular Frequency\s*=\s*)[\d.]+",
        rf"\g<1>{angular_freq:.1f}",
        content,
    )

    with open(sif_path, "w") as f:
        f.write(content)


def run_sweep(start_khz, end_khz, step_khz, solver_path):
    """Run Elmer at each frequency and collect results."""
    with open(SIF_FILE, "r") as f:
        original_sif = f.read()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_steps = (end_khz - start_khz) // step_khz + 1
    results = []

    try:
        for i in range(num_steps):
            freq = start_khz + i * step_khz
            angular_freq = 2 * math.pi * freq * 1000
            print(f"\n{'='*60}")
            print(f"  [{i+1}/{num_steps}] {freq} kHz  (omega = {angular_freq:.1f} rad/s)")
            print(f"{'='*60}")

            set_angular_frequency(SIF_FILE, angular_freq)

            # Remove stale output so we don't copy old results on failure
            if os.path.exists(VTU_OUTPUT):
                os.remove(VTU_OUTPUT)

            result = subprocess.run(
                [solver_path, SIF_FILE],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                results.append((freq, "FAILED", ""))
                print(f"  FAILED - ElmerSolver exited with code {result.returncode}")
                if result.stderr:
                    for line in result.stderr.strip().splitlines()[-5:]:
                        print(f"    {line}")
                continue

            # Extract convergence info from output
            norm_match = re.search(r"SS.*NRM,RELC\):\s*\(\s*([\d.E+-]+)", result.stdout)
            norm = norm_match.group(1) if norm_match else "N/A"

            if os.path.exists(VTU_OUTPUT):
                dest = os.path.join(OUTPUT_DIR, f"case_{freq}kHz.vtu")
                shutil.copy2(VTU_OUTPUT, dest)
                results.append((freq, norm, dest))
                print(f"  Converged. Pressure norm: {norm}")
                print(f"  Saved: {dest}")
            else:
                results.append((freq, "FAILED", ""))
                print(f"  FAILED - solver returned 0 but no output produced")

    finally:
        # Always restore original SIF, even on Ctrl+C or crash
        with open(SIF_FILE, "w") as f:
            f.write(original_sif)
        print(f"\n  Original case.sif restored.")

        # Write CSV summary (including partial results on interruption)
        csv_path = os.path.join(OUTPUT_DIR, "sweep_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["freq_khz", "angular_freq_rad_s", "pressure_norm", "status", "vtu_file"])
            for freq_khz, norm, path in results:
                angular = 2 * math.pi * freq_khz * 1000
                status = "converged" if path else "failed"
                writer.writerow([freq_khz, f"{angular:.1f}", norm, status, os.path.basename(path) if path else ""])

    # Summary
    print(f"\n{'='*60}")
    print(f"  Sweep complete: {len(results)} frequencies")
    print(f"{'='*60}")
    print(f"  {'Freq (kHz)':<12} {'Pressure Norm':<18} {'Output'}")
    print(f"  {'-'*55}")
    for freq_khz, norm, path in results:
        print(f"  {freq_khz:<12} {norm:<18} {path}")
    print(f"\n  Results in: {OUTPUT_DIR}/")
    print(f"  Summary CSV: {csv_path}")


if __name__ == "__main__":
    if not os.path.exists(SIF_FILE):
        print(f"Error: {SIF_FILE} not found. Run from the Elmer-3a directory.")
        sys.exit(1)

    solver = find_elmer_solver()
    print(f"Using ElmerSolver: {solver}")

    if len(sys.argv) == 4:
        start, end, step = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    elif len(sys.argv) == 3:
        start, end, step = int(sys.argv[1]), int(sys.argv[2]), 10
    else:
        start, end, step = 20, 110, 10  # match original result screenshots

    if step <= 0:
        print(f"Error: step must be positive (got {step})")
        sys.exit(1)
    if end < start:
        print(f"Error: end frequency ({end} kHz) must be >= start ({start} kHz)")
        sys.exit(1)

    run_sweep(start, end, step, solver)
