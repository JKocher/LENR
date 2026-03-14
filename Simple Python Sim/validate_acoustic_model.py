#!/usr/bin/env python3
"""Validation suite for acoustic_model.py physics correlations.

Checks implemented correlations against published reference data from
Marczak (1997), Kell (1975), Herrig (2018), Chen-Millero, and IAPWS-95.
Also validates mode frequency predictions, physics sanity, and (optionally)
cross-checks against Elmer FEM results.

Usage:
    python validate_acoustic_model.py              # Levels 1-4
    python validate_acoustic_model.py --fem        # Include Level 5 (FEM)
    python validate_acoustic_model.py --level 2    # Only up to Level 2

Exit code: 0 if all pass, 1 if any fail.

Requires: numpy only (no scipy, no matplotlib)
"""

import argparse
import os
import sys

import numpy as np

from acoustic_model import (
    h2o_properties, d2o_properties, nacl_correction,
    fluid_properties, breathing_modes, band_position,
    port_band_position, wall_transmission, impedance_ratio,
    estimated_frequency, freq_response, optimal_radius,
    pressure_field_j0,
    WALL_PRESETS, KA_PR, KA_RG,
    DEFAULT_RADIUS_MM,
)

# -- Test infrastructure -----------------------------------------------------

_pass_count = 0
_fail_count = 0


def check(name, actual, expected, tol, unit=""):
    """Assert actual is within tol of expected. Print result."""
    global _pass_count, _fail_count
    ok = abs(actual - expected) <= tol
    status = "PASS" if ok else "FAIL"
    unit_str = f" {unit}" if unit else ""
    tol_str = f"+/-{tol}"
    print(f"  {name:<28s} {actual:>10.2f}  ref={expected:<10.2g}"
          f"  tol={tol_str:<8s} {status}")
    if ok:
        _pass_count += 1
    else:
        _fail_count += 1
        print(f"    *** DELTA = {abs(actual - expected):.4f}{unit_str}")


def check_bool(name, condition, detail=""):
    """Assert a boolean condition."""
    global _pass_count, _fail_count
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {name:<50s} {status}{suffix}")
    if condition:
        _pass_count += 1
    else:
        _fail_count += 1


# -- Level 1: Correlation accuracy -------------------------------------------

def level_1():
    """Published reference data comparisons."""
    print("\nLevel 1: Correlation Accuracy")

    # H2O sound speed — Marczak (1997)
    c, rho = h2o_properties(25)
    check("H2O c @ 25C", c, 1497.0, 0.5, "m/s")
    check("H2O rho @ 25C", rho, 997.05, 0.1, "kg/m3")

    c, rho = h2o_properties(50)
    check("H2O c @ 50C", c, 1543.0, 0.5, "m/s")
    check("H2O rho @ 50C", rho, 988.04, 0.1, "kg/m3")

    c, rho = h2o_properties(75)
    check("H2O c @ 75C", c, 1555.1, 0.5, "m/s")
    check("H2O rho @ 75C", rho, 974.84, 0.1, "kg/m3")

    c, rho = h2o_properties(95)
    check("H2O c @ 95C", c, 1547.0, 1.0, "m/s")
    check("H2O rho @ 95C", rho, 961.87, 0.2, "kg/m3")

    # IAPWS-95 high-T
    c, rho = h2o_properties(200)
    check("H2O c @ 200C", c, 1354, 2, "m/s")
    check("H2O rho @ 200C", rho, 865, 2, "kg/m3")

    c, rho = h2o_properties(300)
    check("H2O c @ 300C", c, 1022, 2, "m/s")
    check("H2O rho @ 300C", rho, 712, 2, "kg/m3")

    # D2O — Herrig (2018)
    c, rho = d2o_properties(25)
    check("D2O c @ 25C", c, 1399.6, 2, "m/s")
    check("D2O rho @ 25C", rho, 1104.4, 1, "kg/m3")

    # NaCl — Chen-Millero (simplified). At 25C, 35 ppt (3.5%):
    # Full Chen-Millero gives ~38-40 m/s for this salinity.
    dc, drho = nacl_correction(25, 3.5)
    check("NaCl dc @ 25C, 3.5%", dc, 39, 5, "m/s")


# -- Level 2: Combined parameter bounds --------------------------------------

def level_2():
    """Verify all fluid scenarios produce physically reasonable values."""
    print("\nLevel 2: Combined Parameter Bounds")

    scenarios = [
        ("H2O 25C",     dict(T=25, nacl_pct=0, d2o_pct=0)),
        ("H2O 95C",     dict(T=95, nacl_pct=0, d2o_pct=0)),
        ("Reactor 160C D2O", dict(T=160, nacl_pct=0, d2o_pct=10)),
        ("H2O 200C",    dict(T=200, nacl_pct=0, d2o_pct=0)),
        ("H2O 300C",    dict(T=300, nacl_pct=0, d2o_pct=0)),
        ("5% NaCl 25C", dict(T=25, nacl_pct=5, d2o_pct=0)),
        ("100% D2O 25C", dict(T=25, nacl_pct=0, d2o_pct=100)),
        ("H2O 5C",      dict(T=5, nacl_pct=0, d2o_pct=0)),
        ("D2O+200C",    dict(T=200, nacl_pct=0, d2o_pct=100)),
    ]

    for name, kw in scenarios:
        c, rho = fluid_properties(**kw)
        c_ok = 900 <= c <= 1600
        rho_ok = 700 <= rho <= 1200
        finite = np.isfinite(c) and np.isfinite(rho) and c > 0 and rho > 0
        check_bool(f"{name}: c={c:.0f} in [900,1600]", c_ok)
        check_bool(f"{name}: rho={rho:.0f} in [700,1200]", rho_ok)
        check_bool(f"{name}: finite positive", finite)

    # All wall presets give impedance_ratio > 1 with any fluid
    for wall_name, (wc, wrho) in WALL_PRESETS.items():
        for fluid_name, kw in scenarios[:4]:  # Test with H2O at various temps
            c, rho = fluid_properties(**kw)
            zr = impedance_ratio(wc, wrho, c, rho)
            check_bool(f"{wall_name}/{fluid_name}: Zr={zr:.1f} > 1", zr > 1)


# -- Level 3: Mode frequency predictions -------------------------------------

def level_3():
    """Verify breathing mode frequencies match known values."""
    print("\nLevel 3: Mode Frequency Predictions")

    modes = breathing_modes(18.517, 1497, num_modes=5)

    # (0,1) mode
    m1 = modes[0]
    check("(0,1) f_pr", m1['f_pr'], 40.4, 0.5, "kHz")
    check("(0,1) f_rg", m1['f_rg'], 57.8, 0.5, "kHz")

    # (0,2) mode
    m2 = modes[1]
    check("(0,2) f_pr", m2['f_pr'], 80.8, 0.5, "kHz")
    check("(0,2) f_rg", m2['f_rg'], 99.4, 0.5, "kHz")

    # Reactor design point: 160C, 10% D2O (c ~ 1440 m/s)
    c_rx, _ = fluid_properties(160, 0, 10)
    modes_rx = breathing_modes(18.517, c_rx, 2)
    m1_rx = modes_rx[0]
    check("(0,1) f_pr @ 160C 10%D2O", m1_rx['f_pr'], 38.9, 1.0, "kHz")
    check("(0,1) f_rg @ 160C 10%D2O", m1_rx['f_rg'], 55.6, 1.0, "kHz")
    m2_rx = modes_rx[1]
    check("(0,2) f_pr @ 160C 10%D2O", m2_rx['f_pr'], 77.8, 1.0, "kHz")
    check("(0,2) f_rg @ 160C 10%D2O", m2_rx['f_rg'], 95.6, 1.0, "kHz")

    # Optimal radius for 40 kHz: r_pr close to 18.517 (P-R limit),
    # r_rg > r_pr (rigid needs bigger cavity since ka_rg > ka_pr)
    r_pr, r_rg = optimal_radius(40, 1497)
    check_bool(f"optimal_radius(40): r_pr={r_pr:.1f} near 18.5",
               abs(r_pr - 18.7) < 1.0)
    check_bool(f"optimal_radius(40): r_rg={r_rg:.1f} > r_pr={r_pr:.1f}",
               r_rg > r_pr)


# -- Level 4: Physics sanity -------------------------------------------------

def level_4():
    """Verify physics edge cases and mathematical identities."""
    print("\nLevel 4: Physics Sanity")

    # wall_transmission(d=0) = 1.0
    T0 = wall_transmission(40, 0, 5000, 2700, 1497, 997)
    check_bool(f"wall_transmission(d=0) = {T0:.4f} == 1.0",
               abs(T0 - 1.0) < 1e-10)

    # impedance_ratio with equal materials = 1.0
    zr_eq = impedance_ratio(1497, 997, 1497, 997)
    check_bool(f"impedance_ratio(equal) = {zr_eq:.4f} == 1.0",
               abs(zr_eq - 1.0) < 1e-10)

    # band_position(1.0) = 0.0
    bp1 = band_position(1.0)
    check_bool(f"band_position(1.0) = {bp1:.4f} == 0.0",
               abs(bp1) < 1e-10)

    # port_band_position at low freq -> near P-R (0)
    pbp_low = port_band_position(1.0, 1497, 9.5, 9.0)
    check_bool(f"port_pos @ 1kHz = {pbp_low:.4f} near 0",
               pbp_low < 0.1)

    # port_band_position at high freq -> approaches wall pos
    # At 200 kHz with 9.5mm port, lambda/D ~ 0.4, so ~13% port influence
    pos_wall = band_position(9.0)
    pbp_high = port_band_position(200.0, 1497, 9.5, 9.0)
    check_bool(f"port_pos @ 200kHz = {pbp_high:.4f} > 0.5*wall",
               pbp_high > 0.5 * pos_wall)

    # port_band_position with port_r=0 -> exactly pos_wall
    pbp_no_port = port_band_position(40.0, 1497, 0, 9.0)
    check_bool(f"port_pos(r=0) = {pbp_no_port:.4f} == {pos_wall:.4f}",
               abs(pbp_no_port - pos_wall) < 1e-10)

    # freq_response peak near estimated_frequency at low damping
    zr = impedance_ratio(5000, 2700, 1497, 997)
    pos = band_position(zr)
    modes = breathing_modes(18.517, 1497, 1)
    f_est = estimated_frequency(modes[0], pos)
    freqs, resp = freq_response(18.517, 1497, 0.005, zr,
                                f_max=100, num_points=5000, num_modes=1)
    f_peak = freqs[np.argmax(resp)]
    check_bool(f"resp peak {f_peak:.1f} near est {f_est:.1f} kHz",
               abs(f_peak - f_est) < 2.0)

    # Bessel zero constants
    check_bool(f"KA_PR[0] = {KA_PR[0]:.6f} == pi",
               abs(KA_PR[0] - np.pi) < 1e-10)
    check_bool(f"KA_RG[0] = {KA_RG[0]:.5f} ~ 4.49341",
               abs(KA_RG[0] - 4.49341) < 1e-4)

    # pressure_field_j0: center value = 1.0 for any ka
    X, Z, P = pressure_field_j0(KA_PR[0], 18.517, n=101)
    mid = 50  # center index for n=101
    check_bool(f"field j0: P(center) = {P[mid, mid]:.4f} == 1.0",
               abs(P[mid, mid] - 1.0) < 1e-10)

    # pressure_field_j0: P-R mode has P~0 at cavity wall (edge)
    edge_val = P[mid, -1]  # midline z=0, rightmost point at r=a
    check_bool(f"field j0: P(edge, P-R) = {edge_val:.6f} ~ 0",
               abs(edge_val) < 0.02)

    # pressure_field_j0: mode (0,2) has 1 internal node ring
    X2, Z2, P2 = pressure_field_j0(KA_PR[1], 18.517, n=201)
    mid2 = 100
    radial = P2[mid2, mid2:]  # center to right edge (positive x)
    radial = radial[~np.isnan(radial)]
    sign_changes = int(np.sum(np.diff(np.sign(radial)) != 0))
    check_bool(f"field j0: mode (0,2) has {sign_changes} node(s)",
               sign_changes == 1)

    # pressure_field_j0: output shape and NaN masking
    check_bool(f"field j0: shape = {X.shape} == (101, 101)",
               X.shape == (101, 101) and Z.shape == (101, 101)
               and P.shape == (101, 101))

    # Corners should be NaN (outside cavity circle)
    check_bool(f"field j0: corner is NaN (outside cavity)",
               np.isnan(P[0, 0]))


# -- Level 5: FEM cross-validation -------------------------------------------

def level_5():
    """Cross-validate against Elmer FEM sweep results (conditional)."""
    print("\nLevel 5: FEM Cross-Validation")

    base = os.path.join(os.path.dirname(__file__), '..', 'Elmer-3a',
                        'sweep_results')
    high_freq = os.path.join(base, 'high_freq_fixed.csv')
    fine_sweep = os.path.join(base, 'fine_sweep_fixed.csv')

    if not os.path.exists(high_freq):
        print(f"  SKIP: {high_freq} not found")
        return
    if not os.path.exists(fine_sweep):
        print(f"  SKIP: {fine_sweep} not found")
        return

    # Load high_freq_fixed.csv
    data = np.genfromtxt(high_freq, delimiter=',', names=True)

    # 95 kHz: center_pressure > 100 (confirms breathing mode excitation)
    mask_95 = np.abs(data['freq_khz'] - 95) < 0.5
    if mask_95.any():
        cp_95 = float(data['center_pressure'][mask_95][0])
        check_bool(f"FEM 95kHz center_pressure={cp_95:.0f} > 100",
                   cp_95 > 100)
    else:
        print("  SKIP: 95 kHz not found in high_freq_fixed.csv")

    # Mode (0,2) band [80.8, 99.4] contains 95 kHz
    modes = breathing_modes(18.517, 1497, 2)
    m2 = modes[1]
    check_bool(f"(0,2) band [{m2['f_pr']:.1f}, {m2['f_rg']:.1f}] "
               f"contains 95 kHz",
               m2['f_pr'] <= 95 <= m2['f_rg'])


# -- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate acoustic_model.py against reference data.")
    parser.add_argument("--level", type=int, default=4,
                        help="max level to run (1-5, default: 4)")
    parser.add_argument("--fem", action="store_true",
                        help="include Level 5 FEM cross-validation")
    args = parser.parse_args()

    max_level = 5 if args.fem else args.level

    print("=== LENR Acoustic Model Validation ===")

    if max_level >= 1:
        level_1()
    if max_level >= 2:
        level_2()
    if max_level >= 3:
        level_3()
    if max_level >= 4:
        level_4()
    if max_level >= 5:
        level_5()

    print(f"\nSummary: {_pass_count}/{_pass_count + _fail_count} PASS, "
          f"{_fail_count} FAIL")

    sys.exit(1 if _fail_count > 0 else 0)


if __name__ == '__main__':
    main()
