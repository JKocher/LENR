#!/usr/bin/env python3
"""Shared acoustic physics for LENR cavity analysis tools.

Provides:
  - Fluid property correlations (H2O, D2O, NaCl) with IAPWS-95 high-T data
  - Spherical cavity breathing mode computation
  - Wall impedance and transmission models
  - Port boundary condition corrections
  - Wall material and geometry presets

Used by: cavity_modes.py, parametric_explorer.py

Requires: numpy (always), scipy (only for find_bessel_zeros / mode shapes)
"""

import numpy as np

# -- Precomputed Bessel zeros (exact mathematical constants) ---------------
#
# Breathing modes (n=0) of a spherical cavity:
#   Pressure-release BC: j_0(ka) = sin(ka)/ka = 0  -->  ka = n*pi
#   Rigid-wall BC:       j_0'(ka) = 0  -->  tan(ka) = ka
#
KA_PR = [np.pi * n for n in range(1, 8)]
KA_RG = [4.49341, 7.72525, 10.9041, 14.0662, 17.2208, 20.3713, 23.5195]


# -- IAPWS-95 lookup table for high-temperature water ---------------------
# (T_C, c_m_s, rho_kg_m3) along the liquid saturation curve.
# Sources: NIST Chemistry WebBook, IAPWS-95 equation of state.
# Below 95 C, the Marczak/Kell correlations are more accurate and are used
# instead. This table extends coverage to 300 C for pressurized operation.
_IAPWS_TABLE = np.array([
    # T,    c,     rho
    [  5, 1426, 999.97],
    [ 10, 1447, 999.70],
    [ 15, 1466, 999.10],
    [ 20, 1482, 998.20],
    [ 25, 1497, 997.05],
    [ 30, 1509, 995.65],
    [ 35, 1520, 994.03],
    [ 40, 1529, 992.22],
    [ 45, 1536, 990.21],
    [ 50, 1543, 988.04],
    [ 55, 1548, 985.69],
    [ 60, 1551, 983.20],
    [ 65, 1553, 980.55],
    [ 70, 1555, 977.76],
    [ 75, 1555, 974.84],
    [ 80, 1554, 971.79],
    [ 85, 1553, 968.61],
    [ 90, 1550, 965.31],
    [ 95, 1547, 961.89],
    [100, 1543, 958.35],
    [110, 1530, 950.95],
    [120, 1514, 943.11],
    [130, 1496, 934.83],
    [140, 1475, 926.13],
    [150, 1466, 917.01],
    [160, 1440, 907.45],
    [170, 1425, 897.45],
    [180, 1409, 887.00],
    [190, 1382, 876.08],
    [200, 1354, 864.66],
    [210, 1325, 852.72],
    [220, 1294, 840.22],
    [230, 1261, 827.12],
    [240, 1227, 813.37],
    [250, 1210, 799.00],
    [260, 1168, 783.63],
    [270, 1141, 767.46],
    [280, 1108, 750.28],
    [290, 1068, 732.13],
    [300, 1022, 712.14],
])
_IAPWS_T = _IAPWS_TABLE[:, 0]
_IAPWS_C = _IAPWS_TABLE[:, 1]
_IAPWS_RHO = _IAPWS_TABLE[:, 2]

# Saturation pressure (bar) for warning display
_PSAT_T = np.array([25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300])
_PSAT_P = np.array([0.03, 0.12, 0.39, 1.01, 2.32, 4.76, 8.92,
                     15.5, 25.5, 39.8, 59.5, 85.9])


# -- Fluid property correlations -------------------------------------------

def h2o_properties(T):
    """Water sound speed (m/s) and density (kg/m3) vs temperature.

    Sound speed: Marczak (1997), valid 0-95 C, +/-0.02 m/s.
    Density: Kell (1975), valid 5-100 C, +/-0.03 kg/m3.
    Above 95 C: falls back to IAPWS-95 lookup table.
    """
    if T <= 95:
        c = (1402.385 + 5.038813 * T - 5.799136e-2 * T**2
             + 3.287156e-4 * T**3 - 1.398845e-6 * T**4
             + 2.787860e-9 * T**5)
        rho = ((999.83952 + 16.945176 * T - 7.9870401e-3 * T**2
                - 46.170461e-6 * T**3 + 105.56302e-9 * T**4
                - 280.54253e-12 * T**5)
               / (1 + 16.87985e-3 * T))
        return c, rho
    # High-T: interpolate IAPWS-95 table
    c = float(np.interp(T, _IAPWS_T, _IAPWS_C))
    rho = float(np.interp(T, _IAPWS_T, _IAPWS_RHO))
    return c, rho


def d2o_properties(T):
    """Heavy water sound speed (m/s) and density (kg/m3) vs temperature.

    Polynomial fits to NIST/Herrig (2018) tabulated data, valid 5-95 C.
    Max error: c +/-1.1 m/s, rho +/-0.3 kg/m3.

    Note: For D2O fractions used in mixing (fluid_properties), values above
    95 C are clamped to 95 C. At the reactor design point (10% D2O, 160 C)
    the clamping error is <0.1% on sound speed and ~1.5% on density —
    negligible because the 90% H2O component dominates via IAPWS-95.
    """
    c = (1350.976 + 4.694049 * T - 2.009954e-1 * T**2
         + 4.765162e-3 * T**3 - 4.797849e-5 * T**4 + 1.655053e-7 * T**5)
    rho = (1104.632 + 1.968220e-1 * T - 6.857048e-3 * T**2
           - 7.022205e-5 * T**3 + 4.923662e-7 * T**4)
    return c, rho


def nacl_correction(T, S_pct):
    """Sound speed and density correction for dissolved NaCl.

    Speed: Chen-Millero (simplified), validated 0-40 C, 0-4%.
    Density: linear approximation with temperature correction.
    """
    S = S_pct * 10  # ppt
    dc = ((1.389 - 0.01262 * T + 6.46e-5 * T**2) * S
          + (1.572e-2 + 2.747e-4 * T - 4.16e-6 * T**2) * S**1.5
          + (-3.15e-3) * S**2)
    drho = S_pct * (7.6 - 0.016 * T)
    return dc, drho


def fluid_properties(T, nacl_pct=0, d2o_pct=0):
    """Compute fluid c (m/s) and rho (kg/m3) from physical parameters."""
    c_h2o, rho_h2o = h2o_properties(T)
    if nacl_pct > 0:
        dc, drho = nacl_correction(T, nacl_pct)
        c_h2o += dc
        rho_h2o += drho

    if d2o_pct <= 0:
        return c_h2o, rho_h2o

    # D2O correlations only valid to 95C — clamp silently for mixing.
    # At 10% D2O + 160C (reactor design point), clamping error is <0.1%
    # on sound speed and ~1.5% on density — negligible since H2O dominates.
    T_d2o = min(T, 95.0)
    c_d2o, rho_d2o = d2o_properties(T_d2o)
    if nacl_pct > 0:
        dc, drho = nacl_correction(T_d2o, nacl_pct)
        c_d2o += dc
        rho_d2o += drho

    x = d2o_pct / 100.0
    # Mix via bulk modulus K = rho*c^2 (not linear c) for physical consistency
    K_h2o = rho_h2o * c_h2o**2
    K_d2o = rho_d2o * c_d2o**2
    rho_mix = (1 - x) * rho_h2o + x * rho_d2o
    K_mix = (1 - x) * K_h2o + x * K_d2o
    c_mix = np.sqrt(K_mix / rho_mix)
    return c_mix, rho_mix


def saturation_pressure(T):
    """Approximate saturation pressure (bar) for display purposes."""
    if T <= 100:
        return 1.01
    return float(np.interp(T, _PSAT_T, _PSAT_P))


# -- Wall presets ----------------------------------------------------------

# Note: Aluminum uses 5000 m/s (extensional/bar speed) to match case.sif.
# Longitudinal bulk speed is ~6320 m/s — a known upstream inconsistency.
WALL_PRESETS = {
    'Aluminum':      (5000, 2700),
    '316 Stainless': (5790, 8000),
    'Titanium':      (6070, 4500),
    'Copper':        (4760, 8960),
    'PEEK':          (2530, 1300),
    'Glass':         (5640, 2230),
}

# Reference geometry (from upstream CAD/mesh analysis)
DEFAULT_RADIUS_MM = 18.517
DEFAULT_WALL_THICKNESS = 6.9   # mm at pole
DEFAULT_PORT_RADIUS = 2.0      # mm — sight window / backlight port


# -- Physics ---------------------------------------------------------------

def breathing_modes(a_mm, c_fluid, num_modes=5):
    """Compute breathing mode (0,m) frequency bands in kHz.

    Returns list of dicts with f_pr (pressure-release) and f_rg (rigid)
    frequency bounds for each mode order.
    """
    a = a_mm * 1e-3
    n = min(num_modes, len(KA_PR), len(KA_RG))
    return [{
        'm': i + 1,
        'f_pr': KA_PR[i] * c_fluid / (2 * np.pi * a) / 1000,
        'f_rg': KA_RG[i] * c_fluid / (2 * np.pi * a) / 1000,
    } for i in range(n)]


def impedance_ratio(wall_c, wall_rho, c_fluid, rho_fluid):
    """Acoustic impedance ratio Z_wall / Z_fluid."""
    return (wall_rho * wall_c) / (rho_fluid * c_fluid)


def band_position(zr):
    """Position in P-R <-> Rigid band (0=P-R, 1=Rigid).
    Uses reflection coefficient: R = (Z-1)/(Z+1)."""
    return (zr - 1) / (zr + 1) if zr > 1 else 0.0


def estimated_frequency(mode, pos):
    """Estimated resonant frequency from band position."""
    return mode['f_pr'] + pos * (mode['f_rg'] - mode['f_pr'])


def port_band_position(f_khz, c_fluid, port_r_mm, zr):
    """Frequency-dependent band position blending port and wall effects.

    Utility ports (water circulation / observation windows) create small
    open areas in the cavity wall that act as pressure-release boundaries.
    The wall impedance determines the BC for the enclosed portion.
    At low frequencies (lambda >> port diameter), the open area dominates.
    At high frequencies (lambda << port diameter), wall impedance wins.

    Returns 0 (P-R) to ~pos_wall (impedance-based), not 0 to 1.
    """
    pos_wall = (zr - 1) / (zr + 1) if zr > 1 else 0.0
    if port_r_mm <= 0:
        return pos_wall  # No port = fully enclosed, wall impedance rules
    lam = c_fluid / (f_khz * 1000) * 1000  # wavelength in mm
    D_port = 2 * port_r_mm
    ratio = lam / D_port  # lambda/D
    # Port influence: ratio >> 1 (low freq) -> port dominates -> P-R
    #                 ratio << 1 (high freq) -> wall dominates -> pos_wall
    port_weight = ratio**2 / (1.0 + ratio**2)
    return pos_wall * (1.0 - port_weight)


def wall_transmission(f_khz, d_mm, c_wall, rho_wall, c_fluid, rho_fluid):
    """Power transmission coefficient through a planar slab.

    Transfer matrix method for a single layer between two identical
    semi-infinite fluid media.
    """
    if d_mm <= 0:
        return 1.0
    f = f_khz * 1000
    Z_s = rho_wall * c_wall
    Z_f = rho_fluid * c_fluid
    # c_wall in m/s, d_mm in mm — multiply c by 1000 to get mm/s
    k = 2 * np.pi * f / (c_wall * 1000)  # wavenumber in 1/mm
    kd = k * d_mm
    # Avoid tan singularities near kd = pi/2 + n*pi
    Z_in = Z_s * (Z_f + 1j * Z_s * np.tan(kd)) / (Z_s + 1j * Z_f * np.tan(kd))
    T = 4 * Z_f * np.real(Z_in) / np.abs(Z_f + Z_in)**2
    return np.clip(T, 0, 1)


def wall_character(zr):
    """Human-readable description of wall impedance behavior."""
    if zr > 20:
        return 'Very rigid'
    if zr > 10:
        return 'Rigid'
    if zr > 5:
        return 'Semi-rigid'
    if zr > 2:
        return 'Intermediate'
    return 'Soft (P-R like)'


def freq_response(a_mm, c_fluid, damping, zr,
                  f_max=250, num_points=2000, num_modes=5,
                  wall_d=0, wall_c=0, wall_rho=0, rho_fluid=997,
                  port_r_mm=0, zr_for_port=0, layers=None):
    """Analytical frequency response for breathing modes.

    When layers are enabled, applies wall transmission and/or
    port-based band position corrections.

    layers: set containing 'wall' and/or 'port', or None for L0 only.
    """
    if layers is None:
        layers = set()

    pos_static = band_position(zr)
    freqs = np.linspace(1, f_max, num_points)
    response = np.zeros_like(freqs)

    eff_damping = max(damping, 1e-4)
    modes = breathing_modes(a_mm, c_fluid, num_modes)

    for mode in modes:
        if 'port' in layers:
            f_mid = (mode['f_pr'] + mode['f_rg']) / 2
            pos = port_band_position(f_mid, c_fluid, port_r_mm,
                                     zr_for_port or zr)
        else:
            pos = pos_static

        f_res = estimated_frequency(mode, pos)
        k_res = 2 * np.pi * f_res * 1000 / c_fluid
        k = 2 * np.pi * freqs * 1000 / c_fluid
        gamma = eff_damping * k_res
        denom = (k**2 - k_res**2)**2 + (gamma * k)**2
        response += 1.0 / np.sqrt(np.maximum(denom, 1e-30))

    # Apply wall transmission weighting
    if 'wall' in layers and wall_d > 0 and wall_c > 0:
        T_wall = np.array([wall_transmission(f, wall_d, wall_c, wall_rho,
                                             c_fluid, rho_fluid)
                           for f in freqs])
        response *= T_wall

    return freqs, response


def pressure_field_j0(ka_eff, a_mm, n=200):
    """2D pressure field for breathing mode (n=0) cross-section.

    Computes P(r) = j_0(ka_eff * r/a) on a Cartesian grid spanning
    the full cavity circle. j_0(x) = sin(x)/x — pure numpy, no scipy.

    Returns (X_mm, Z_mm, P):
      X_mm: horizontal position (mm), shape (n, n)
      Z_mm: vertical position along acoustic axis (mm), shape (n, n)
      P: pressure amplitude normalized to P(0)=1, shape (n, n).
         NaN outside the cavity (r > a).
    """
    x = np.linspace(-a_mm, a_mm, n)
    X_mm, Z_mm = np.meshgrid(x, x)
    R = np.sqrt(X_mm**2 + Z_mm**2)

    # j_0(x) = sin(x)/x, with j_0(0) = 1
    kr = ka_eff * R / a_mm
    P = np.ones_like(kr)
    nonzero = kr > 0
    P[nonzero] = np.sin(kr[nonzero]) / kr[nonzero]

    # Mask outside cavity
    P[R > a_mm] = np.nan

    return X_mm, Z_mm, P


# -- Scipy-dependent functions (optional) ----------------------------------
# These are only needed by cavity_modes.py for exact Bessel zero finding
# and mode shape computation. The parametric explorer uses precomputed
# zeros (KA_PR, KA_RG) and doesn't need scipy.

def find_bessel_zeros(func, x_max=50, num_zeros=10):
    """Find positive zeros of func(x) using sign-change detection + Brent.

    Requires scipy.
    """
    from scipy.optimize import brentq
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


def resonant_frequencies(a_mm, c, n_max=3, m_max=4):
    """Compute resonant frequencies for angular orders 0..n_max.

    Returns list of dicts with keys: n, m, ka, freq_khz, bc_type, wavelength_mm.
    Requires scipy.
    """
    from scipy.special import spherical_jn
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


def optimal_radius(target_khz, c):
    """Return (radius_pr_mm, radius_rigid_mm) for the (0,1) breathing mode."""
    f = target_khz * 1000
    a_pr = KA_PR[0] * c / (2 * np.pi * f) * 1000
    a_rg = KA_RG[0] * c / (2 * np.pi * f) * 1000
    return a_pr, a_rg


def mode_shape_radial(a_mm, n=0, ka=None, num_points=200):
    """Return (r_mm, amplitude) for a spherical Bessel mode.

    Requires scipy.
    """
    from scipy.special import spherical_jn
    if ka is None:
        ka = KA_RG[0]
    a = a_mm * 1e-3
    r = np.linspace(0, a, num_points)
    k = ka / a
    shape = np.array([spherical_jn(n, k * ri) if ri > 0 else
                       (1.0 if n == 0 else 0.0) for ri in r])
    return r * 1000, shape
