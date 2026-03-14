#!/usr/bin/env python3
"""Interactive parametric explorer for LENR reactor cavity acoustics.

Visualizes how cavity resonant modes shift with changes to geometry,
fluid properties, and wall material. Uses thermophysical correlations
to ensure physically consistent fluid properties — temperature,
salinity, and D2O fraction drive sound speed and density together
along their known curves.

Features:
  - Mode band chart showing P-R/Rigid frequency bands with transducer overlay
  - Cross-section view of j_0(kr) pressure field for selected breathing mode
  - Analytical frequency response with optional wall transmission overlay
  - Three toggleable analysis layers (L0/L1/L2)

Controls:
    - Sliders adjust cavity radius, temperature, fluid composition, damping
    - Wall preset buttons select wall material (c and rho are coupled)
    - Fluid scenario buttons load common T/NaCl/D2O configurations
    - Checkboxes toggle analysis layers
    - Click on a mode bar in the bands chart to select it for cross-section
    - Keys [1]-[5] select mode for cross-section
    - Press 'r' to reset all parameters to defaults

Requires: numpy, matplotlib
"""

import sys
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

from acoustic_model import (
    fluid_properties, saturation_pressure,
    breathing_modes, band_position, port_band_position,
    wall_transmission, freq_response,
    impedance_ratio, estimated_frequency, pressure_field_j0,
    WALL_PRESETS, DEFAULT_RADIUS_MM, DEFAULT_WALL_THICKNESS,
    DEFAULT_PORT_RADIUS, KA_PR, KA_RG,
)

# -- Practical slider limits (benchtop to pressurized operation) -----------
# Centralised so they're easy to adjust as the design evolves.
SLIDER_LIMITS = {
    'radius':      (5, 50),     # mm — small test cavity to large prototype
    'temperature': (5, 300),    # C — extended for pressurized operation
    'nacl_pct':    (0, 10),     # % — above 10% is corrosion-impractical
    'd2o_pct':     (0, 100),    # % — D2O is purchasable in any fraction
    'damping':     (0.001, 0.20),
    'wall_thick':  (1, 15),     # mm — structural minimum to practical max
    'port_radius': (0.5, 5),    # mm — sight window / backlight (L2 experimental)
}

# Fluid scenarios: (temperature_C, nacl_pct, d2o_pct)
FLUID_SCENARIOS = {
    'Reactor 160\u00b0C 10%D\u2082O':  (160, 0, 10),    # Nominal design point
    'Coumarin 140\u00b0C 10%D\u2082O': (140, 0, 10),    # Dye laser config
    'H\u2082O 25\u00b0C':              (25, 0, 0),       # Benchtop baseline
    'H\u2082O 75\u00b0C':              (75, 0, 0),       # Benchtop warm
    'H\u2082O 95\u00b0C':              (95, 0, 0),       # Benchtop max (unpressurized)
    '5% NaCl':                         (25, 5, 0),       # Salt solution
    '50% D\u2082O':                    (25, 0, 50),      # Heavy water test
}

TRANSDUCERS = {
    40:  '#E53935',
    100: '#1E88E5',
    120: '#43A047',
    135: '#FB8C00',
    200: '#8E24AA',
}

MODE_COLORS = ['#1565C0', '#2E7D32', '#EF6C00', '#AD1457', '#6A1B9A',
               '#00838F', '#4E342E']

DEFAULTS = dict(radius=DEFAULT_RADIUS_MM, temperature=160.0, nacl_pct=0.0,
                d2o_pct=10.0, damping=0.01,
                wall_thick=DEFAULT_WALL_THICKNESS,
                port_radius=DEFAULT_PORT_RADIUS)

# Target scenarios: full state snapshots for transducer-matched configurations.
# Each entry: (slider_overrides, wall_preset_name, selected_mode_index,
#              fluid_scenario_name)
_BENCHTOP = dict(radius=DEFAULT_RADIUS_MM, temperature=25.0, nacl_pct=0.0,
                 d2o_pct=0.0, damping=0.01,
                 wall_thick=DEFAULT_WALL_THICKNESS,
                 port_radius=DEFAULT_PORT_RADIUS)

TARGET_SCENARIOS = {
    'Design\nPoint':  (dict(DEFAULTS),  'Aluminum', 0,
                       'Reactor 160\u00b0C 10%D\u2082O'),
    '40 kHz\n25\u00b0C':  (dict(_BENCHTOP), 'Aluminum', 0,
                           'H\u2082O 25\u00b0C'),
    '95 kHz\n25\u00b0C':  (dict(_BENCHTOP), 'Aluminum', 1,
                           'H\u2082O 25\u00b0C'),
}


# -- Explorer GUI ----------------------------------------------------------

class Explorer:
    NUM_MODES = 5

    def __init__(self):
        self._batch = False
        self.wall_c, self.wall_rho = WALL_PRESETS['Aluminum']
        self.layers = set()  # active analysis layers
        self._ax_resp_twin = None  # twin axis for wall T% overlay
        self.selected_mode = 0  # index into modes list for cross-section

        self.fig = plt.figure(figsize=(16, 10))
        try:
            self.fig.canvas.manager.set_window_title(
                'LENR Cavity Parametric Explorer')
        except AttributeError:
            pass
        self.fig.patch.set_facecolor('#f0f2f5')

        # ===== LAYOUT =====
        # Top row: bands plot (left), cross-section (right)
        # Mid row: response plot (left), presets + layers (right)
        # Bottom strip: sliders in two groups (exploration | geometry)

        # -- Plot axes --
        self.ax_bands = self.fig.add_axes([0.06, 0.54, 0.54, 0.40])
        self.ax_resp = self.fig.add_axes([0.06, 0.24, 0.54, 0.16])

        # -- Cross-section panel (right of bands) --
        self.ax_xsec = self.fig.add_axes([0.65, 0.54, 0.32, 0.40])

        # -- Right column: Presets + Analysis Layers --
        # Fluid scenario (left), Wall material (center), Layers (right)
        self.ax_fl = self.fig.add_axes([0.65, 0.24, 0.14, 0.16])
        self.ax_wl = self.fig.add_axes([0.80, 0.24, 0.08, 0.16])
        self.ax_layers = self.fig.add_axes([0.89, 0.24, 0.09, 0.16])

        self.ax_fl.set_title('Fluid', fontsize=9, fontweight='bold', pad=4)
        self.ax_wl.set_title('Wall', fontsize=9, fontweight='bold', pad=4)
        self.ax_layers.set_title('Layers', fontsize=9, fontweight='bold',
                                 pad=4)

        self.fl_radio = RadioButtons(self.ax_fl, list(FLUID_SCENARIOS.keys()),
                                     active=0, activecolor='#1E88E5')
        self.wl_radio = RadioButtons(self.ax_wl, list(WALL_PRESETS.keys()),
                                     active=0, activecolor='#E53935')
        for lbl in self.fl_radio.labels:
            lbl.set_fontsize(7.5)
        for lbl in self.wl_radio.labels:
            lbl.set_fontsize(8)
        self.fl_radio.on_clicked(self._on_fluid_scenario)
        self.wl_radio.on_clicked(self._on_wall)

        # Layer checkboxes
        layer_labels = ['L1 Wall atten.', 'L2 Port shift']
        self.layer_check = CheckButtons(self.ax_layers, layer_labels,
                                        actives=[False, False])
        for lbl in self.layer_check.labels:
            lbl.set_fontsize(8.5)
        self.layer_check.on_clicked(self._on_layer_toggle)

        # Layer hints — tell the user where to look
        self.fig.text(0.89, 0.225,
                      'L1: red T% curve on response\n'
                      'L2: shifts \u25c6 on bands chart',
                      fontsize=6.5, color='#888', va='top')

        # -- Sliders: two groups --
        # Group 1 (Exploration): radius, temp, nacl, d2o, damping
        # Group 2 (Geometry): wall thickness, port radius

        D = DEFAULTS
        sl_h = 0.015
        gap = 0.032

        # Exploration sliders — left side
        expl_x, expl_w = 0.14, 0.46
        expl_y0 = 0.045

        def make_slider(x, w, idx, label, lo, hi, default, fmt, color):
            ax = self.fig.add_axes([x, expl_y0 + idx * gap, w, sl_h])
            return Slider(ax, label, lo, hi, valinit=default, valfmt=fmt,
                          color=color)

        L = SLIDER_LIMITS
        self.sl = {
            'damping':     make_slider(expl_x, expl_w, 0,
                                       'Damping',        *L['damping'],
                                       D['damping'],     '%.3f', '#546E7A'),
            'd2o_pct':     make_slider(expl_x, expl_w, 1,
                                       'D\u2082O (%)',    *L['d2o_pct'],
                                       D['d2o_pct'],     '%.0f', '#8E24AA'),
            'nacl_pct':    make_slider(expl_x, expl_w, 2,
                                       'NaCl (%)',        *L['nacl_pct'],
                                       D['nacl_pct'],    '%.1f', '#43A047'),
            'temperature': make_slider(expl_x, expl_w, 3,
                                       'Temp (\u00b0C)', *L['temperature'],
                                       D['temperature'], '%.0f', '#1E88E5'),
            'radius':      make_slider(expl_x, expl_w, 4,
                                       'Radius (mm)',     *L['radius'],
                                       D['radius'],      '%.1f', '#FB8C00'),
        }

        # Geometry sliders — right side (aligned with layer checkboxes)
        geo_x, geo_w = 0.74, 0.22
        geo_y0 = 0.045
        geo_gap = 0.032

        self.sl['wall_thick'] = make_slider(geo_x, geo_w, 0,
                                            'Wall',   *L['wall_thick'],
                                            D['wall_thick'],  '%.1f mm',
                                            '#D32F2F')
        self.sl['port_radius'] = make_slider(geo_x, geo_w, 1,
                                             'Port r', *L['port_radius'],
                                             D['port_radius'], '%.1f mm',
                                             '#E64A19')

        # Group labels
        self.fig.text(0.14, expl_y0 + 5 * gap + 0.005,
                      'FLUID & CAVITY', fontsize=8,
                      fontweight='bold', color='#666')
        self.fig.text(0.74, geo_y0 + 2 * gap + 0.005,
                      'GEOMETRY', fontsize=8,
                      fontweight='bold', color='#666')

        # Reset button + target scenario buttons
        self.ax_reset = self.fig.add_axes([0.005, 0.005, 0.06, 0.025])
        self.btn_reset = Button(self.ax_reset, 'Reset',
                                color='#CFD8DC', hovercolor='#90A4AE')
        self.btn_reset.label.set_fontsize(8)
        self.btn_reset.on_clicked(self._on_reset_btn)

        # Target scenario buttons
        self._target_btns = []
        target_names = list(TARGET_SCENARIOS.keys())
        for i, name in enumerate(target_names):
            ax = self.fig.add_axes([0.07 + i * 0.065, 0.005, 0.06, 0.025])
            btn = Button(ax, name, color='#E3F2FD', hovercolor='#90CAF9')
            btn.label.set_fontsize(6.5)
            btn.on_clicked(lambda _, n=name: self._on_target(n))
            self._target_btns.append(btn)

        hint_x = 0.07 + len(target_names) * 0.065 + 0.005
        self.fig.text(hint_x, 0.011, '[R] Reset  [1-5] Mode',
                      fontsize=7, color='#999')

        for s in self.sl.values():
            s.on_changed(self._on_slider)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('button_press_event', self._on_band_click)

        self.redraw()
        plt.show()

    # -- Callbacks --

    def _on_fluid_scenario(self, label):
        T, nacl, d2o = FLUID_SCENARIOS[label]
        self._batch = True
        self.sl['temperature'].set_val(T)
        self.sl['nacl_pct'].set_val(nacl)
        self.sl['d2o_pct'].set_val(d2o)
        self._batch = False
        self.redraw()

    def _on_wall(self, label):
        self.wall_c, self.wall_rho = WALL_PRESETS[label]
        self.redraw()

    def _on_layer_toggle(self, label):
        if self._batch:
            return
        key = {'L1 Wall atten.': 'wall', 'L2 Port shift': 'port'}[label]
        if key in self.layers:
            self.layers.discard(key)
        else:
            self.layers.add(key)
        self.redraw()

    def _on_slider(self, val):
        if not self._batch:
            self.redraw()

    def _reset(self):
        self._batch = True
        for name, val in DEFAULTS.items():
            self.sl[name].set_val(val)
        self.wall_c, self.wall_rho = WALL_PRESETS['Aluminum']
        self.wl_radio.set_active(0)
        self.fl_radio.set_active(0)
        self.layers.clear()
        for i, status in enumerate(self.layer_check.get_status()):
            if status:
                self.layer_check.set_active(i)
        self.selected_mode = 0
        self._batch = False
        self.redraw()

    def _on_reset_btn(self, event):
        self._reset()

    def _on_target(self, name):
        """Load a target scenario (full state snapshot)."""
        vals, wall_name, mode_idx, fluid_name = TARGET_SCENARIOS[name]
        self._batch = True
        for k, v in vals.items():
            self.sl[k].set_val(v)
        self.wall_c, self.wall_rho = WALL_PRESETS[wall_name]
        # Set wall radio to match
        wall_names = list(WALL_PRESETS.keys())
        self.wl_radio.set_active(wall_names.index(wall_name))
        # Set fluid radio to match scenario
        fluid_names = list(FLUID_SCENARIOS.keys())
        self.fl_radio.set_active(fluid_names.index(fluid_name))
        # Clear layers
        self.layers.clear()
        for i, status in enumerate(self.layer_check.get_status()):
            if status:
                self.layer_check.set_active(i)
        self.selected_mode = mode_idx
        self._batch = False
        self.redraw()

    def _on_key(self, event):
        if event.key == 'r':
            self._reset()
        elif event.key in ('1', '2', '3', '4', '5'):
            idx = int(event.key) - 1
            if idx < self.NUM_MODES:
                self.selected_mode = idx
                self.redraw()

    def _on_band_click(self, event):
        if event.inaxes is not self.ax_bands or event.ydata is None:
            return
        # Map y-click to nearest mode index
        idx = int(round(event.ydata))
        if 0 <= idx < self.NUM_MODES:
            self.selected_mode = idx
            self.redraw()

    # -- Drawing --

    def redraw(self):
        v = {k: s.val for k, s in self.sl.items()}

        c_fluid, rho_fluid = fluid_properties(
            v['temperature'], v['nacl_pct'], v['d2o_pct'])

        zr = (self.wall_rho * self.wall_c) / (rho_fluid * c_fluid)
        pos = band_position(zr)
        modes = breathing_modes(v['radius'], c_fluid, self.NUM_MODES)

        f_max = 250
        if modes:
            f_max = max(f_max, max(m['f_rg'] for m in modes) * 1.15)

        self._draw_bands(modes, pos, f_max, c_fluid, v, zr)
        self._draw_response(v, c_fluid, rho_fluid, zr, f_max)
        self._draw_xsec(v, c_fluid, rho_fluid, modes, pos, zr)
        self.fig.canvas.draw_idle()

    def _draw_bands(self, modes, pos, f_max, c_fluid, v, zr):
        ax = self.ax_bands
        ax.clear()

        for i, mode in enumerate(modes):
            color = MODE_COLORS[i % len(MODE_COLORS)]
            y = i

            ax.barh(y, mode['f_rg'] - mode['f_pr'], left=mode['f_pr'],
                    height=0.55, color=color, alpha=0.25,
                    edgecolor=color, linewidth=1.5)

            # Band position: use port model if L2 active, else impedance
            if 'port' in self.layers:
                f_mid = (mode['f_pr'] + mode['f_rg']) / 2
                p = port_band_position(f_mid, c_fluid, v['port_radius'], zr)
            else:
                p = pos

            f_est = mode['f_pr'] + p * (mode['f_rg'] - mode['f_pr'])
            ax.plot(f_est, y, 'D', color=color, markersize=10,
                    markeredgecolor='black', markeredgewidth=0.8, zorder=5)

            ax.text(max(mode['f_pr'] - 3, 1), y, f'(0,{mode["m"]})',
                    ha='right', va='center', fontsize=10,
                    fontweight='bold', color=color)

            ax.text(f_est, y - 0.38, f'{f_est:.1f}',
                    ha='center', va='top', fontsize=8,
                    fontweight='bold', color=color)

            # Highlight selected mode with thicker border
            if i == self.selected_mode:
                ax.barh(y, mode['f_rg'] - mode['f_pr'], left=mode['f_pr'],
                        height=0.55, color='none',
                        edgecolor='black', linewidth=3.0, zorder=4)

        t_freqs = sorted(f for f in TRANSDUCERS if f <= f_max)
        for idx, freq in enumerate(t_freqs):
            color = TRANSDUCERS[freq]
            ax.axvline(freq, color=color, ls='--', alpha=0.6, lw=1.5)
            y_label = len(modes) - 0.2 - (idx % 2) * 0.5
            ax.text(freq + 1, y_label, f'{freq}',
                    ha='left', va='bottom', fontsize=8,
                    color=color, fontweight='bold')

            for i, mode in enumerate(modes):
                if mode['f_pr'] <= freq <= mode['f_rg']:
                    ax.plot(freq, i, '*', color=color, markersize=18,
                            markeredgecolor='black', markeredgewidth=0.5,
                            zorder=6)

        ax.set_xlim(0, f_max)
        ax.set_ylim(-0.6, len(modes) + 0.3)
        ax.set_yticks([])
        ax.tick_params(labelbottom=False)

        # Title reflects active layers
        layer_tags = []
        if 'wall' in self.layers:
            layer_tags.append('Wall atten.')
        if 'port' in self.layers:
            layer_tags.append('Port shift')
        suffix = '  [+' + ', '.join(layer_tags) + ']' if layer_tags else ''

        ax.set_title(
            'Breathing Modes:  bar = P-R..Rigid band  |  '
            '\u25c6 estimated  |  \u2605 transducer match'
            + suffix,
            fontsize=10, fontweight='bold')
        ax.grid(axis='x', alpha=0.15)

    def _draw_response(self, v, c_fluid, rho_fluid, zr, f_max):
        ax = self.ax_resp
        ax.clear()

        # Remove stale twin axis from previous draw
        if self._ax_resp_twin is not None:
            self._ax_resp_twin.remove()
            self._ax_resp_twin = None

        freqs, resp = freq_response(
            v['radius'], c_fluid, v['damping'], zr,
            f_max=f_max, num_modes=self.NUM_MODES,
            wall_d=v['wall_thick'], wall_c=self.wall_c,
            wall_rho=self.wall_rho, rho_fluid=rho_fluid,
            port_r_mm=v['port_radius'], zr_for_port=zr,
            layers=self.layers)

        peak = resp.max()
        resp_n = resp / peak if peak > 0 else resp

        ax.semilogy(freqs, resp_n, color='#333', lw=1.5)

        modes = breathing_modes(v['radius'], c_fluid, self.NUM_MODES)
        for i, mode in enumerate(modes):
            color = MODE_COLORS[i % len(MODE_COLORS)]
            mask = (freqs >= mode['f_pr']) & (freqs <= mode['f_rg'])
            if mask.any():
                ax.fill_between(freqs[mask], 1e-5, resp_n[mask],
                                color=color, alpha=0.15)

        for freq, color in TRANSDUCERS.items():
            if freq <= f_max:
                ax.axvline(freq, color=color, ls='--', alpha=0.4, lw=1.5)

        # Show wall transmission curve if L1 active
        if 'wall' in self.layers and v['wall_thick'] > 0:
            T_wall = np.array([wall_transmission(f, v['wall_thick'],
                               self.wall_c, self.wall_rho,
                               c_fluid, rho_fluid) for f in freqs])
            self._ax_resp_twin = ax.twinx()
            self._ax_resp_twin.fill_between(freqs, 0, T_wall * 100,
                                             color='#E53935', alpha=0.08)
            self._ax_resp_twin.plot(freqs, T_wall * 100, color='#E53935',
                                    lw=1.8, alpha=0.85, label='Wall T%')
            self._ax_resp_twin.set_ylabel('Wall T%', fontsize=8,
                                           color='#E53935', fontweight='bold')
            self._ax_resp_twin.set_ylim(0, 100)
            self._ax_resp_twin.tick_params(axis='y', labelsize=7,
                                            colors='#E53935')

        ax.set_xlim(0, f_max)
        ax.set_ylim(1e-4, 2)
        ax.set_xlabel('Frequency (kHz)', fontsize=10)
        ax.set_ylabel('Response', fontsize=9)
        ax.set_title('Analytical Frequency Response', fontsize=10)
        ax.grid(True, alpha=0.15)

    def _draw_xsec(self, v, c_fluid, rho_fluid, modes, pos, zr):
        ax = self.ax_xsec
        ax.clear()

        a = v['radius']
        m_idx = min(self.selected_mode, len(modes) - 1)
        mode = modes[m_idx]
        color = MODE_COLORS[m_idx % len(MODE_COLORS)]

        # Compute effective ka from band position
        if 'port' in self.layers:
            f_mid = (mode['f_pr'] + mode['f_rg']) / 2
            p = port_band_position(f_mid, c_fluid, v['port_radius'], zr)
        else:
            p = pos
        ka_eff = KA_PR[m_idx] + p * (KA_RG[m_idx] - KA_PR[m_idx])
        f_est = mode['f_pr'] + p * (mode['f_rg'] - mode['f_pr'])

        # Compute pressure field (full circle cross-section)
        X, Z, P = pressure_field_j0(ka_eff, a, n=151)

        # Plot pressure field
        vmax = 1.0
        ax.pcolormesh(X, Z, P, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       shading='gouraud', rasterized=True)

        # Node ring contours where j_0 = 0
        ax.contour(X, Z, P, levels=[0], colors='black',
                    linewidths=0.8, linestyles='dashed')

        # Wall annulus (gray fill, both sides)
        wall_d = v['wall_thick']
        port_r = v['port_radius']
        if wall_d > 0:
            if port_r > 0 and port_r < a:
                gap_angle = np.arcsin(min(port_r / a, 1.0))
            else:
                gap_angle = 0

            th_w = np.linspace(gap_angle, np.pi - gap_angle, 200)
            x_in = a * np.sin(th_w)
            z_in = a * np.cos(th_w)
            r_out = a + wall_d
            x_out = r_out * np.sin(th_w)
            z_out = r_out * np.cos(th_w)

            # Right side wall
            ax.fill(np.concatenate([x_in, x_out[::-1]]),
                    np.concatenate([z_in, z_out[::-1]]),
                    color='#757575', alpha=0.5, zorder=2)
            # Left side wall (mirror)
            ax.fill(np.concatenate([-x_in, -x_out[::-1]]),
                    np.concatenate([z_in, z_out[::-1]]),
                    color='#757575', alpha=0.5, zorder=2)

        # Cavity boundary circle
        th_circ = np.linspace(0, 2 * np.pi, 400)
        ax.plot(a * np.sin(th_circ), a * np.cos(th_circ),
                color='#333', lw=1.5, zorder=3)

        # Annotations
        layer_str = 'L0'
        if 'wall' in self.layers:
            layer_str += '+L1'
        if 'port' in self.layers:
            layer_str += '+L2'

        ax.set_title(f'Mode (0,{mode["m"]})  {f_est:.1f} kHz  '
                     f'ka={ka_eff:.2f}  [{layer_str}]',
                     fontsize=10, fontweight='bold', color=color)

        info_str = (f'c={c_fluid:.0f}  '
                    f'\u03c1={rho_fluid:.0f}  '
                    f'Z={zr:.1f}x')
        T = v['temperature']
        if T > 100:
            psat_bar = saturation_pressure(T)
            psat_psi = psat_bar * 14.5038
            info_str += f'\nPsat={psat_psi:.0f} PSI ({psat_bar:.1f} bar)'
            if psat_psi > 200:
                info_str += ' \u26a0'

        ax.text(0.02, 0.02, info_str,
                fontsize=7.5, transform=ax.transAxes,
                color='#555', family='monospace',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

        ax.set_aspect('equal')
        margin = a * 0.1 + wall_d
        extent = a + wall_d + margin
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_xlabel('X (mm)', fontsize=8)
        ax.set_ylabel('Z (mm)', fontsize=8)
        ax.tick_params(labelsize=7)


if __name__ == '__main__':
    Explorer()
