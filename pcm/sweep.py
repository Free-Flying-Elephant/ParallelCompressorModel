"""
Distortion sweep driver for the multiple-segment parallel compressor model.

Sweeps over
    - rotor speed (one speedline per value),
    - distortion intensity  (fractional total-pressure deficit in the
      distorted sector, e.g. 0.05 = 5 % below the clean-sector pressure),
    - distortion extent     (angular width of the distorted sector [deg]),
solves the coupled parallel-compressor problem for every combination, and
plots the *overall* (whole-compressor) total pressure ratio and isentropic
efficiency in two separate figures.

Distortion pattern
------------------
Classical one-per-rev "square-wave" circumferential distortion: one
low-pressure sector of width ``extent`` and a clean sector covering the
rest of the annulus. Because parallel-compressor theory treats every
sector with identical inlet conditions identically, two segments
(distorted + clean) are an exact representation of this pattern, however
many sub-segments one might split each sector into.

Usage (from the directory containing this file and ``hecc.chC``):
    python sweep.py
    python sweep.py --map hecc.chC --beta 0.5 --intensities 0 0.02 0.05 0.1 \
                    --extents 0 30 60 90 120 180 --no-show
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

# The pcm modules import each other with flat (non-package) imports, so make
# the directory of this file importable regardless of the working directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from compressor_map import CompressorMap  # noqa: E402
from map_io import read_hecc_chc  # noqa: E402
from model import ParallelCompressorModel  # noqa: E402


# =============================================================================
# Configuration and result containers
# =============================================================================


@dataclass(frozen=True)
class SweepConfig:
    """
    Parameters of a distortion sweep.

    Attributes
    ----------
    speeds_rpm      : rotor speeds to sweep [1/min]. Each is one speedline.
    intensities     : fractional total-pressure deficits of the distorted
                      sector [-] (0.05 = 5 % lower than the clean sector).
    extents_deg     : angular extents of the distorted sector [deg], each in
                      [0, 360]. 0 and 360 both give a uniform inlet.
    beta            : overall operating-point parameter (see the model).
    p0_clean        : total pressure of the undistorted sector [Pa].
    T0_inlet        : (uniform) inlet total temperature [K].
    exit_area       : compressor discharge area [m^2].
    """

    speeds_rpm: Tuple[float, ...]
    intensities: Tuple[float, ...]
    extents_deg: Tuple[float, ...]
    beta: float = 0.5
    p0_clean: float = 101_325.0
    T0_inlet: float = 288.15
    exit_area: float = 0.005


@dataclass
class SweepResult:
    """
    Sweep output. All arrays have shape (n_speeds, n_intensities, n_extents);
    entries are NaN where the coupled solve failed.

    Attributes
    ----------
    pressure_ratio : overall total-to-total pressure ratio [-].
    efficiency     : overall isentropic efficiency [-].
    surge_margin   : overall (mass-flow-weighted) surge margin [%].
    min_surge_margin : lowest segment surge margin [%] (the critical sector).
    """

    config: SweepConfig
    pressure_ratio: NDArray[np.float64]
    efficiency: NDArray[np.float64]
    surge_margin: NDArray[np.float64]
    min_surge_margin: NDArray[np.float64]
    discharge_p0_ratio: NDArray[np.float64]  # overall discharge p0 / clean-sector inlet p0
    n_failed: int = 0


# =============================================================================
# Sweep driver
# =============================================================================


class DistortionSweep:
    """Runs the parallel-compressor model over a speed/intensity/extent grid."""

    def __init__(self, compressor_map: CompressorMap, config: SweepConfig) -> None:
        self.map: CompressorMap = compressor_map
        self.config: SweepConfig = config

    @staticmethod
    def build_pattern(
        extent_deg: float, intensity: float, p0_clean: float
    ) -> Tuple[List[float], List[float]]:
        """
        Build (segment angles, segment inlet total pressures) for a single
        low-pressure sector of width ``extent_deg`` and the given intensity.

        Degenerate extents (<= 0 or >= 360) collapse to one uniform segment
        (clean for 0, fully depressed for 360) -- this also avoids a
        zero-width segment, which the model rejects.
        """
        p0_low = p0_clean * (1.0 - intensity)
        if extent_deg <= 0.0:
            return [360.0], [p0_clean]
        if extent_deg >= 360.0:
            return [360.0], [p0_low]
        return [extent_deg, 360.0 - extent_deg], [p0_low, p0_clean]

    def _solve_point(
        self, speed_rpm: float, intensity: float, extent_deg: float
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """
        Solve one sweep point. Returns (PR, efficiency, overall SM, min
        segment SM, discharge p0 / clean inlet p0) or ``None`` if the point
        is not a valid equilibrium: solver failure, NaNs, or any segment
        pinned at the map's surge / choke bound (the bounded solve then
        violates the mass-flow / static-pressure coupling, so the numbers
        are meaningless).
        """
        cfg = self.config
        angles, p0s = self.build_pattern(extent_deg, intensity, cfg.p0_clean)
        try:
            model = ParallelCompressorModel(
                compressor_map=self.map,
                rotor_speed_rpm=speed_rpm,
                beta=cfg.beta,
                segment_angles_deg=angles,
                inlet_p0=p0s,
                inlet_T0=[cfg.T0_inlet] * len(angles),
                exit_area=cfg.exit_area,
            )
            # A non-converged solve emits RuntimeWarning: treat as failure so
            # unconverged points never contaminate the plots.
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                res = model.solve()
        except (ValueError, RuntimeWarning, FloatingPointError):
            return None

        o = res.overall

        # Validity: every segment strictly inside the map's beta range AND
        # the total mass flow actually closes on the target. A segment at
        # beta = 0 (surge) or beta = 1 (choke end of map) means the
        # distortion is beyond what the map can represent.
        beta_tol = 1e-4
        betas = np.array([r.beta for r in res.segments])
        b_lo = float(self.map.beta_grid.min())
        b_hi = float(self.map.beta_grid.max())
        if betas.min() <= b_lo + beta_tol or betas.max() >= b_hi - beta_tol:
            return None
        target = model._target_mass_flow_actual  # set by solve()
        if target is None or abs(o.mass_flow - target) / target > 1e-6:
            return None

        values = (
            o.pressure_ratio,
            o.efficiency,
            o.surge_margin,
            min(r.surge_margin for r in res.segments),
            o.p0 / cfg.p0_clean,
        )
        if not all(np.isfinite(v) for v in values):
            return None
        return values

    def run(self) -> SweepResult:
        """Execute the full sweep (speeds x intensities x extents)."""
        cfg = self.config
        shape = (len(cfg.speeds_rpm), len(cfg.intensities), len(cfg.extents_deg))
        pr = np.full(shape, np.nan)
        eff = np.full(shape, np.nan)
        sm = np.full(shape, np.nan)
        sm_min = np.full(shape, np.nan)
        p0_ratio = np.full(shape, np.nan)
        n_failed = 0

        for i, speed in enumerate(cfg.speeds_rpm):
            for j, inten in enumerate(cfg.intensities):
                for k, ext in enumerate(cfg.extents_deg):
                    out = self._solve_point(speed, inten, ext)
                    if out is None:
                        n_failed += 1  # invalid point -> left as NaN (gap in plots)
                        continue
                    (pr[i, j, k], eff[i, j, k], sm[i, j, k],
                     sm_min[i, j, k], p0_ratio[i, j, k]) = out

        return SweepResult(cfg, pr, eff, sm, sm_min, p0_ratio, n_failed)


# =============================================================================
# Output: plots and CSV
# =============================================================================


class SweepPlotter:
    """Draws the sweep results: one figure per quantity, one subplot per speed."""

    def __init__(self, result: SweepResult, max_cols: int = 3) -> None:
        self.result: SweepResult = result
        self.max_cols: int = max_cols

    def plot_quantity(
        self, data: NDArray[np.float64], ylabel: str, title: str
    ) -> Figure:
        """
        One subplot per speedline; x = distortion extent, one curve per
        distortion intensity (colored dark -> light with increasing intensity).
        """
        cfg = self.result.config
        n_speeds = len(cfg.speeds_rpm)
        n_cols = min(self.max_cols, n_speeds)
        n_rows = int(np.ceil(n_speeds / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5.0 * n_cols, 3.8 * n_rows), sharex=True, squeeze=False
        )
        colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(cfg.intensities)))

        for i, speed in enumerate(cfg.speeds_rpm):
            ax = axes[i // n_cols][i % n_cols]
            for j, inten in enumerate(cfg.intensities):
                ax.plot(
                    cfg.extents_deg,
                    data[i, j, :],
                    marker="o",
                    markersize=3,
                    linewidth=1.3,
                    color=colors[j],
                    label=f"{100.0 * inten:g} %",
                )
            ax.set_title(f"{speed:.0f} rpm", fontsize=10)
            ax.grid(True, linewidth=0.4, alpha=0.6)
            if i % n_cols == 0:
                ax.set_ylabel(ylabel)
            if i // n_cols == n_rows - 1:
                ax.set_xlabel("Distortion extent [deg]")
            ax.set_xlim(min(cfg.extents_deg), max(cfg.extents_deg))

        # Hide unused subplots in the last row.
        for i in range(n_speeds, n_rows * n_cols):
            axes[i // n_cols][i % n_cols].set_visible(False)

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Intensity (p0 deficit)", loc="center right", fontsize=8)
        fig.suptitle(f"{title} (beta = {cfg.beta:g})")
        fig.tight_layout(rect=(0.0, 0.0, 0.9, 0.96))
        return fig

    def plot_pressure_ratio(self) -> Figure:
        # PR is referenced to the AREA-AVERAGED inlet p0, so a uniform loss
        # (extent 0 or 360 deg) gives the undistorted value: the curve
        # measures non-uniformity, not how much of the annulus is depressed.
        return self.plot_quantity(
            self.result.pressure_ratio,
            "Overall PR (vs. area-avg. inlet p0) [-]",
            "Overall pressure ratio vs. distortion extent",
        )

    def plot_discharge_pressure(self) -> Figure:
        # Discharge p0 referenced to the CLEAN-sector inlet p0: falls
        # monotonically as more of the annulus is depressed.
        return self.plot_quantity(
            self.result.discharge_p0_ratio,
            "Discharge p0 / clean inlet p0 [-]",
            "Discharge total pressure vs. distortion extent",
        )

    def plot_efficiency(self) -> Figure:
        return self.plot_quantity(
            self.result.efficiency,
            "Overall isentropic efficiency [-]",
            "Overall efficiency vs. distortion extent",
        )


def write_csv(result: SweepResult, path: Path) -> None:
    """Dump every sweep point (long format) to a CSV file."""
    cfg = result.config
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speed_rpm", "intensity", "extent_deg", "pressure_ratio",
                    "efficiency", "surge_margin_overall_pct", "surge_margin_min_pct",
                    "discharge_p0_over_clean_p0"])
        for i, s in enumerate(cfg.speeds_rpm):
            for j, inten in enumerate(cfg.intensities):
                for k, ext in enumerate(cfg.extents_deg):
                    w.writerow([s, inten, ext, result.pressure_ratio[i, j, k],
                                result.efficiency[i, j, k], result.surge_margin[i, j, k],
                                result.min_surge_margin[i, j, k],
                                result.discharge_p0_ratio[i, j, k]])


# =============================================================================
# Command line
# =============================================================================


def _default_map_path() -> Path:
    """Look for hecc.chC next to this file, then in the parent / current dir."""
    for cand in (_HERE / "hecc.chC", _HERE.parent / "hecc.chC", Path.cwd() / "hecc.chC"):
        if cand.exists():
            return cand
    return _HERE / "hecc.chC"


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Parallel-compressor distortion sweep")
    p.add_argument("--map", type=Path, default=_default_map_path(), help="path to .chC map")
    p.add_argument("--speeds", type=float, nargs="+", default=[100.],
                   help="rotor speeds [rpm] (default: every speedline in the map)")
    p.add_argument("--intensities", type=float, nargs="+",
                   default=[0.0, 0.04, 0.08, 0.10, 0.14, 0.18],
                   help="fractional p0 deficits of the distorted sector")
    p.add_argument("--extents", type=float, nargs="+",
                   default=list(np.arange(0, 361, 20)),
                   help="distorted-sector extents [deg]")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--p0", type=float, default=101_325.0, help="clean-sector p0 [Pa]")
    p.add_argument("--T0", type=float, default=288.15, help="inlet T0 [K]")
    p.add_argument("--exit-area", type=float, default=0.007, help="exit area [m^2]")
    p.add_argument("--outdir", type=Path, default=Path("."), help="where to save PNG/CSV")
    p.add_argument("--no-show", action="store_true", help="save only; do not open windows", default=True)
    args = p.parse_args(argv)

    cmap = read_hecc_chc(args.map)
    speeds = tuple(args.speeds) if args.speeds else tuple(float(s) for s in cmap.corrected_speed_grid)

    cfg = SweepConfig(
        speeds_rpm=speeds,
        intensities=tuple(args.intensities),
        extents_deg=tuple(args.extents),
        beta=args.beta,
        p0_clean=args.p0,
        T0_inlet=args.T0,
        exit_area=args.exit_area,
    )
    n_total = len(cfg.speeds_rpm) * len(cfg.intensities) * len(cfg.extents_deg)
    print(f"Sweeping {n_total} cases ...")
    result = DistortionSweep(cmap, cfg).run()
    print(f"Done. {result.n_failed} of {n_total} cases invalid (distorted sector at surge, "
          f"clean sector at choke, or non-converged) and left as gaps.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    plotter = SweepPlotter(result)
    plotter.plot_pressure_ratio().savefig(args.outdir / "sweep_pressure_ratio.png", dpi=130)
    plotter.plot_discharge_pressure().savefig(args.outdir / "sweep_discharge_pressure.png", dpi=130)
    plotter.plot_efficiency().savefig(args.outdir / "sweep_efficiency.png", dpi=130)
    write_csv(result, args.outdir / "sweep_results.csv")
    print(f"Saved sweep_pressure_ratio.png, sweep_discharge_pressure.png, "
          f"sweep_efficiency.png, sweep_results.csv to {args.outdir}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

