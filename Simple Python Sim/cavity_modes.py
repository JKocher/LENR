#!/usr/bin/env python3
"""Analytical acoustic modes of a spherical water-filled cavity.

Computes resonant frequencies and pressure mode shapes using spherical
Bessel functions.  Replaces the two-point-source approximation in the
original script with an exact eigenmode solution.

The cavity wall boundary condition lies between two idealized limits:
  - Pressure-release (p=0): zeros of j_n(ka)
  - Rigid wall (dp/dr=0):   zeros of j_n'(ka)
The real aluminum-water interface (impedance ratio ~9) is closer to rigid.

Usage:
  python cavity_modes.py                      # defaults: a=18.517mm, c=1497 m/s
  python cavity_modes.py --radius 18.517      # explicit radius in mm
  python cavity_modes.py --freq 40            # find optimal radius for 40 kHz
  python cavity_modes.py --sweep 20 120 1     # frequency sweep (start, end, step kHz)
"""

import argparse
import sys

import numpy as np
from scipy.special import spherical_jn
from scipy.optimize import brentq

# Geometry and material defaults (matching Elmer case.sif)
DEFAULT_RADIUS_MM = 18.517
DEFAULT_SOUND_SPEED = 1497.0  # m/s in water at room temperature

# Bessel zero for the (0,1) breathing mode under each BC type
KA_PRESSURE_RELEASE = np.pi     # j_0(ka) = 0
KA_RIGID = 4.49341              # j_0'(ka) = 0


def find_bessel_zeros(func, x_max=50, num_zeros=10):
    """Find positive zeros of func(x) using sign-change detection + Brent."""
    x = np.linspace(0.01, x_max, 50000)
    y = func(x)
    sign_changes = np.where(np.diff(np.sign(y)))[0]
    zeros = []
    for idx in sign_changes:
        try:
            z = brentq(func, x[idx], x[idx + 1])
            if z > 0.1:
                zeros.append(z)
        except ValueError:
            pass
        if len(zeros) >= num_zeros:
            break
    return zeros


def resonant_frequencies(a_mm, c=DEFAULT_SOUND_SPEED, n_max=3, m_max=4):
    """Compute resonant frequencies for angular orders 0..n_max.

    Returns list of dicts with keys: n, m, ka, freq_khz, bc_type, wavelength_mm.
    """
    a = a_mm * 1e-3
    modes = []

    for n in range(n_max + 1):
        # Pressure-release: j_n(ka) = 0
        jn = lambda x, _n=n: spherical_jn(_n, x)
        for m, ka in enumerate(find_bessel_zeros(jn, num_zeros=m_max), 1):
            f = ka * c / (2 * np.pi * a)
            modes.append({
                "n": n, "m": m, "ka": ka,
                "freq_khz": f / 1000,
                "wavelength_mm": c / f * 1000,
                "bc_type": "P-R",
            })

        # Rigid: j_n'(ka) = 0
        if n == 0:
            djn = lambda x: -spherical_jn(1, x)
        else:
            djn = lambda x, _n=n: (spherical_jn(_n - 1, x)
                                    - (_n + 1) / x * spherical_jn(_n, x))
        for m, ka in enumerate(find_bessel_zeros(djn, num_zeros=m_max), 1):
            f = ka * c / (2 * np.pi * a)
            modes.append({
                "n": n, "m": m, "ka": ka,
                "freq_khz": f / 1000,
                "wavelength_mm": c / f * 1000,
                "bc_type": "Rigid",
            })

    modes.sort(key=lambda x: x["freq_khz"])
    return modes


def optimal_radius(target_khz, c=DEFAULT_SOUND_SPEED):
    """Return (radius_pr_mm, radius_rigid_mm) for the (0,1) breathing mode."""
    f = target_khz * 1000
    a_pr = KA_PRESSURE_RELEASE * c / (2 * np.pi * f) * 1000
    a_rg = KA_RIGID * c / (2 * np.pi * f) * 1000
    return a_pr, a_rg


def mode_shape_radial(a_mm, n=0, ka=KA_RIGID, num_points=200):
    """Return (r_mm, amplitude) for a spherical Bessel mode."""
    a = a_mm * 1e-3
    r = np.linspace(0, a, num_points)
    k = ka / a
    shape = np.array([spherical_jn(n, k * ri) if ri > 0 else
                       (1.0 if n == 0 else 0.0) for ri in r])
    return r * 1000, shape


def frequency_sweep_analytical(a_mm, f_start, f_end, f_step,
                               c=DEFAULT_SOUND_SPEED):
    """Compute center pressure response vs frequency for (0,1) mode.

    Uses a simple driven-oscillator model: response ~ 1/|k^2 - k_res^2 + i*gamma|
    to approximate the frequency response shape near resonance.
    """
    a = a_mm * 1e-3
    # Resonance for rigid (0,1)
    k_res = KA_RIGID / a
    gamma = 0.01 * k_res  # damping (matches case.sif Sound damping = 0.01)

    freqs = np.arange(f_start, f_end + f_step / 2, f_step)
    response = []
    for f_khz in freqs:
        k = 2 * np.pi * f_khz * 1000 / c
        # Driven response near resonance
        denom = (k**2 - k_res**2)**2 + (gamma * k)**2
        response.append(1.0 / np.sqrt(denom))

    return freqs, np.array(response)


def plot_modes(a_mm, c=DEFAULT_SOUND_SPEED, output_dir="."):
    """Generate comprehensive mode analysis plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    modes = resonant_frequencies(a_mm, c)

    # --- Plot 1: Mode table + breathing mode shapes ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Breathing modes (n=0) only
    breathing = [m for m in modes if m["n"] == 0]
    pr_modes = [m for m in breathing if m["bc_type"] == "P-R"]
    rg_modes = [m for m in breathing if m["bc_type"] == "Rigid"]

    for m in pr_modes[:3]:
        r_mm, shape = mode_shape_radial(a_mm, n=0, ka=m["ka"])
        ax1.plot(r_mm, np.abs(shape), "--", alpha=0.6,
                 label=f'P-R (0,{m["m"]}) {m["freq_khz"]:.1f} kHz')
    for m in rg_modes[:3]:
        r_mm, shape = mode_shape_radial(a_mm, n=0, ka=m["ka"])
        ax1.plot(r_mm, np.abs(shape), "-", linewidth=2,
                 label=f'Rigid (0,{m["m"]}) {m["freq_khz"]:.1f} kHz')

    ax1.axvline(a_mm, color="grey", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Radial distance from center (mm)")
    ax1.set_ylabel("|j_0(kr)| — normalized pressure amplitude")
    ax1.set_title(f"Breathing Mode Shapes (a = {a_mm} mm)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Frequency response
    freqs, resp = frequency_sweep_analytical(a_mm, 20, 120, 0.5, c)
    ax2.semilogy(freqs, resp / resp.max(), color="#1E88E5", linewidth=2)

    # Mark resonances
    for m in rg_modes[:3]:
        ax2.axvline(m["freq_khz"], color="#FF9800", linestyle="--", alpha=0.5)
        ax2.text(m["freq_khz"] + 0.5, 0.7, f'{m["freq_khz"]:.1f} kHz',
                 fontsize=8, rotation=90, color="#FF9800")

    ax2.set_xlabel("Frequency (kHz)")
    ax2.set_ylabel("Relative response (log)")
    ax2.set_title(f"Analytical Frequency Response (a = {a_mm} mm)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Spherical Cavity Acoustic Modes — Analytical Solution",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "cavity_modes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Plot 2: Design chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    a_range = np.linspace(5, 40, 200)
    f_pr = KA_PRESSURE_RELEASE * c / (2 * np.pi * a_range * 1e-3) / 1000
    f_rg = KA_RIGID * c / (2 * np.pi * a_range * 1e-3) / 1000

    ax.fill_between(a_range, f_pr, f_rg, alpha=0.15, color="#1E88E5")
    ax.plot(a_range, f_pr, "--", color="#1E88E5", alpha=0.6,
            label="Pressure-release limit")
    ax.plot(a_range, f_rg, "--", color="#1565C0", alpha=0.6,
            label="Rigid wall limit")
    ax.axvline(a_mm, color="#E53935", linestyle=":", linewidth=2,
               label=f"Current cavity ({a_mm} mm)")

    # Mark standard transducer frequencies
    for ft in [20, 28, 40, 60]:
        ax.axhline(ft, color="#FF9800", linestyle=":", alpha=0.3)
        ax.text(39, ft + 1, f"{ft} kHz", fontsize=8, color="#FF9800", ha="right")

    ax.set_xlabel("Cavity radius (mm)")
    ax.set_ylabel("Fundamental breathing mode frequency (kHz)")
    ax.set_title("Cavity Design Chart — Mode (0,1)")
    ax.legend(fontsize=9)
    ax.set_xlim(5, 40)
    ax.set_ylim(0, 200)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "cavity_design_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analytical acoustic modes of a spherical cavity."
    )
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_MM,
                        help="cavity radius in mm (default: %(default)s)")
    parser.add_argument("--speed", type=float, default=DEFAULT_SOUND_SPEED,
                        help="sound speed in m/s (default: %(default)s)")
    parser.add_argument("--freq", type=float, default=None,
                        help="target frequency in kHz — compute optimal radius")
    parser.add_argument("--sweep", nargs=3, type=float, metavar=("START", "END", "STEP"),
                        help="frequency sweep range in kHz")
    parser.add_argument("--plot", action="store_true",
                        help="generate mode shape and design chart plots")
    parser.add_argument("--output-dir", default=".",
                        help="directory for plot output (default: current dir)")
    args = parser.parse_args()

    a = args.radius
    c = args.speed

    # Mode table
    print(f"Cavity radius: {a} mm")
    print(f"Sound speed:   {c} m/s")
    print()

    if args.freq is not None:
        r_pr, r_rg = optimal_radius(args.freq, c)
        print(f"Target frequency: {args.freq} kHz")
        print(f"  Optimal radius (P-R):   {r_pr:.1f} mm")
        print(f"  Optimal radius (Rigid): {r_rg:.1f} mm")
        print(f"  Bracket: {r_pr:.1f} – {r_rg:.1f} mm")
        return

    modes = resonant_frequencies(a, c)
    breathing = [m for m in modes if m["n"] == 0]

    print("Breathing modes (n=0) — these focus energy at center:")
    print(f"{'Mode':>8s}  {'BC':>6s}  {'ka':>8s}  {'Freq (kHz)':>12s}  {'Wavelength':>12s}")
    print("-" * 52)
    for m in breathing:
        label = f'(0,{m["m"]})'
        print(f'{label:>8s}  {m["bc_type"]:>6s}  {m["ka"]:>8.3f}'
              f'  {m["freq_khz"]:>12.1f}  {m["wavelength_mm"]:>10.1f} mm')

    if args.sweep:
        start, end, step = args.sweep
        freqs, resp = frequency_sweep_analytical(a, start, end, step, c)
        peak_idx = np.argmax(resp)
        print(f"\nAnalytical sweep {start}–{end} kHz:")
        print(f"  Peak response at: {freqs[peak_idx]:.1f} kHz")

    if args.plot:
        plot_modes(a, c, args.output_dir)

    # Standard transducer frequency matching
    print(f"\nTransducer matching for a = {a} mm cavity:")
    print(f"{'Transducer':>12s}  {'Radius for P-R':>14s}  {'Radius for Rigid':>16s}  {'Match?':>8s}")
    print("-" * 56)
    for ft in [20, 28, 40, 60, 80, 100]:
        rp, rr = optimal_radius(ft, c)
        match = "YES" if rp <= a <= rr else ""
        print(f"  {ft:>6d} kHz  {rp:>12.1f} mm  {rr:>14.1f} mm  {match:>8s}")


if __name__ == "__main__":
    main()
