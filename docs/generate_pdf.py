#!/usr/bin/env python3
"""Generate PDF report from acoustic analysis images and text."""
import os
from fpdf import FPDF

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(DOCS_DIR, "images")
OUTPUT = os.path.join(DOCS_DIR, "acoustic-analysis.pdf")


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 5, "LENR Reactor - Acoustic Analysis Report", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, title)
        self.ln(8)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title)
        self.ln(6)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, text)
        self.ln(3)

    def add_image(self, filename, caption="", w=190):
        path = os.path.join(IMAGES_DIR, filename)
        if not os.path.isfile(path):
            self.body_text(f"[Image not found: {filename}]")
            return
        self.image(path, w=w)
        if caption:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(80, 80, 80)
            self.cell(0, 5, caption, align="C")
            self.ln(8)
        else:
            self.ln(5)


def main():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # --- Title Page ---
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 15, "Acoustic Analysis Report", align="C")
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "LENR Sonoluminescence Reactor", align="C")
    pdf.ln(8)
    pdf.cell(0, 10, "Frequency Sweep Post-Processing & Cavity Mode Predictions", align="C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "Contributor: dbRollin", align="C")
    pdf.ln(6)
    pdf.cell(0, 8, "Repository: github.com/JKocher/LENR", align="C")
    pdf.ln(6)
    pdf.cell(0, 8, "March 2026", align="C")

    # --- Overview ---
    pdf.add_page()
    pdf.section_title("1. Overview")
    pdf.body_text(
        "This report documents analysis performed on the LENR reactor's acoustic simulation "
        "using two new tools: sweep_analysis.py (FEM post-processor) and cavity_modes.py "
        "(analytical mode calculator). During validation, two unit-consistency issues were "
        "identified that were preventing standing wave formation in the simulation."
    )
    pdf.body_text(
        "The reactor is a spherical water-filled cavity (radius 18.517 mm) flanked by "
        "hemispherical aluminum acoustic horns, driven at ultrasonic frequencies to produce "
        "sonoluminescence. The simulation uses Elmer FEM to solve the Helmholtz equation "
        "for acoustic pressure wave propagation."
    )

    # --- Issue 1 ---
    pdf.section_title("2. Finding: Sound Speed Unit Mismatch")
    pdf.body_text(
        "The Elmer mesh is defined in millimeters, but the material properties in case.sif "
        "use SI units (m/s for sound speed, kg/m^3 for density). The Helmholtz equation "
        "computes wavenumber k = omega/c. With c = 1497 m/s interpreted as 1497 mm/s, "
        "the computed wavelength is 0.037 mm  - 400x smaller than the mesh element "
        "size (~1.5 mm). The mesh cannot resolve these waves."
    )
    pdf.body_text(
        "This explains the observation in the Simulation Guide: \"Suspect there are still "
        "simulation issues, since we are not achieving a standing wave at either 40 kHz or "
        "60 kHz.\""
    )
    pdf.body_text(
        "Corrected values: Sound Speed = 1,497,000 mm/s (water), 5,000,000 mm/s (aluminium). "
        "Density = 9.983e-10 tonne/mm^3 (water), 2.7e-9 tonne/mm^3 (aluminium)."
    )

    # --- Issue 2 ---
    pdf.section_title("3. Finding: Diameter/Radius in Python Sim")
    pdf.body_text(
        "The analytical Python simulation uses radius_sphere = 37e-3 (37 mm). The actual "
        "cavity radius from the CAD geometry is 18.517 mm. The value 37 mm is the diameter "
        "(2 x 18.517 ~ 37.034). This models a cavity with 8x the actual volume, "
        "producing incorrect resonance predictions. Historical plot filenames containing "
        "'r18.517' confirm the intended radius."
    )

    # --- Before/After ---
    pdf.add_page()
    pdf.section_title("4. Before & After: Pressure Field at 40 kHz")
    pdf.body_text(
        "With original units, pressure is confined to the horn surfaces with zero penetration "
        "into the water cavity. With corrected units, wave patterns form across the full geometry."
    )
    pdf.add_image("composite_before_after_40kHz.png",
                  "Figure 1: Acoustic pressure at 40 kHz  - original (left) vs corrected (right)")

    # --- Frequency Response ---
    pdf.add_page()
    pdf.section_title("5. Frequency Response: Original vs Corrected")
    pdf.body_text(
        "The original sweep shows monotonic 1/f decay with no resonance peaks. The corrected "
        "sweep reveals clear resonance structure with two breathing modes."
    )
    pdf.add_image("composite_freq_response.png",
                  "Figure 2: Max pressure (top) and center pressure (bottom) vs frequency")
    pdf.body_text(
        "Key findings:\n"
        "* Mode (0,1) at ~40 kHz: center pressure = 1.75x10-5 (first breathing mode)\n"
        "* Mode (0,2) at 95 kHz: center pressure = 509 (massive resonance)"
    )

    # --- FEM vs Analytical ---
    pdf.add_page()
    pdf.section_title("6. FEM vs Analytical Validation")
    pdf.body_text(
        "The analytical tool (cavity_modes.py) predicts resonance bands using spherical Bessel "
        "function zeros. The FEM results fall within the predicted bands for both modes, "
        "providing independent validation."
    )
    pdf.add_image("composite_fem_vs_analytical.png",
                  "Figure 3: FEM peaks vs analytical prediction bands")

    pdf.sub_title("Mode Comparison")
    pdf.body_text(
        "Mode (0,1): Analytical 40.4 - 57.8 kHz, FEM ~40 kHz  - at the pressure-release limit\n"
        "Mode (0,2): Analytical 80.8 - 99.4 kHz, FEM 95 kHz  - within the predicted band"
    )

    # --- 95 kHz Resonance ---
    pdf.add_page()
    pdf.section_title("7. 95 kHz Resonance: Pressure & Temperature")
    pdf.body_text(
        "At the (0,2) breathing mode, acoustic energy focuses at the cavity center. Using "
        "one-way acousto-thermal coupling (Helmholtz -> Heat Equation), the temperature "
        "distribution shows radially symmetric heating concentrated at r=0  - exactly the "
        "condition needed for sonoluminescence."
    )
    pdf.add_image("composite_95kHz_resonance.png",
                  "Figure 4: Acoustic pressure (left) and temperature gradient (right) at 95 kHz")

    # --- Design Tools ---
    pdf.add_page()
    pdf.section_title("8. Design Tools")
    pdf.body_text(
        "cavity_modes.py provides a design chart showing how the fundamental breathing mode "
        "frequency varies with cavity radius, with common transducer frequencies marked. "
        "This enables rapid parametric exploration without running the full FEM solver."
    )
    pdf.add_image("cavity_design_chart.png",
                  "Figure 5: Cavity design chart  - frequency vs radius for mode (0,1)")
    pdf.ln(5)
    pdf.add_image("cavity_modes.png",
                  "Figure 6: Breathing mode shapes and analytical frequency response")

    # --- Recommendations ---
    pdf.add_page()
    pdf.section_title("9. Recommendations")
    pdf.body_text(
        "1. Update case.sif material properties to use mm-consistent units. Only the "
        "Helmholtz-relevant properties need correction (sound speed and density). Thermal "
        "and electromagnetic properties are not used by the current solver.\n\n"
        "2. Target 95 kHz for maximum center pressure amplification. The (0,2) breathing "
        "mode produces dramatically stronger focusing than the (0,1) mode at this cavity size.\n\n"
        "3. Fix radius_sphere in the Python simulation to 18.517e-3 (radius, not diameter).\n\n"
        "4. Use cavity_modes.py for rapid parametric studies before running full FEM sweeps. "
        "The analytical predictions bracket the FEM results reliably."
    )

    pdf.output(OUTPUT)
    print(f"PDF saved: {OUTPUT}")


if __name__ == "__main__":
    main()
