"""
Stability-limit sweep for a fixed-extent total-pressure distortion.

For one rotor speed, a fixed circumferential extent (default 90 deg) and a
range of distortion intensities and operating points (beta), this driver
computes with the multiple-segment parallel compressor model:

    * the DISTORTED case: one low-pressure sector of the given extent,
      inlet total pressure p0_clean * (1 - intensity), the rest of the
      annulus at p0_clean;
    * the UNIFORM case with the same area-averaged inlet total pressure,
      p0_clean * (1 - intensity * extent / 360), over the whole annulus
      (the like-for-like reference: both cases share the same nominal
      operating point / target mass flow inside the model);

and finds the STABILITY LIMIT for every beta: the largest intensity for
which the distorted sector is still above the surge line (its local beta
stays > 0, i.e. surge margin > 0). Beyond that intensity the distorted
sector is taken as stalled and the distorted curve is cut.

Outputs: pressure-ratio, efficiency and stability-limit figures plus a CSV.

Usage (from the directory with this file and hecc.chC):
    python stability_sweep.py
    python stability_sweep.py --speed 20670 --betas 0.2 0.4 0.6 0.8 \
                              --extent 90 --max-intensity 0.3 --no-show
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
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
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
class StabilityConfig:
    """
    Parameters of the stability sweep.

    Attributes
    ----------
    speed_rpm      : rotor speed [1/min] (one speedline).
    betas          : operating points (nominal, undistorted beta) to sweep.
    extent_deg     : angular extent of the distorted sector [deg], in (0, 360).
    max_intensity  : largest fractional total-pressure drop considered [-];
                     also the upper bound of the stability-limit search.
    n_intensity    : number of evenly spaced intensities in [0, max_intensity].
    p0_clean       : total pressure of the undistorted sector [Pa].
    T0_inlet       : uniform inlet total temperature [K].
    exit_area      : compressor discharge area [m^2].
    beta_tol       : a segment whose local beta is within beta_tol of the map's
                     surge (0) or choke (1) bound counts as being on that
                     bound. The bounded solver approaches a bound
                     asymptotically, so this must not be too tight; 1e-4 in
                     beta corresponds to ~1e-3 % in intensity here.
    closure_tol    : relative tolerance for the mass-flow / static-pressure
                     constraints; a solve that misses them (e.g. because a
                     segment is pinned at the beta bound) is not a valid
                     equilibrium and counts as unstable.
    """

    speed_rpm: float
    betas: Tuple[float, ...]
    extent_deg: float = 90.0
    max_intensity: float = 0.30
    n_intensity: int = 31
    p0_clean: float = 101_325.0
    T0_inlet: float = 288.15
    exit_area: float = 0.005
    beta_tol: float = 1e-4
    closure_tol: float = 1e-6


@dataclass(frozen=True)
class CasePoint:
    """
    Overall (whole-compressor) result of one solved case.

    Attributes
    ----------
    intensity        : fractional p0 drop of the distorted sector [-].
    p0_in_avg        : area-averaged inlet total pressure [Pa].
    p0_out           : mass-flow-weighted discharge total pressure [Pa].
    pressure_ratio   : overall total pressure ratio (p0_out / p0_in_avg) [-].
    efficiency       : overall isentropic efficiency [-].
    mass_flow        : total actual mass flow [kg/s].
    surge_margin     : overall (mass-flow-weighted) surge margin [%].
    min_surge_margin : lowest segment surge margin [%] (critical sector).
    beta_min         : lowest local beta over the segments [-].
    beta_max         : highest local beta over the segments [-].
    status           : "ok"    valid equilibrium, every segment on the map
                                between the surge and choke bounds;
                       "surge" a segment is pinned on the surge line (beta = 0);
                       "choke" a segment is pinned on the choke bound
                                (beta = 1, end of the map) with no surge;
                       "fail"  solver failure / non-finite result.
    """

    intensity: float
    p0_in_avg: float
    p0_out: float
    pressure_ratio: float
    efficiency: float
    mass_flow: float
    surge_margin: float
    min_surge_margin: float
    beta_min: float
    beta_max: float
    status: str

    @property
    def stable(self) -> bool:
        return self.status == "ok"


@dataclass
class BetaCurve:
    """
    All results for one operating point (beta).

    Attributes
    ----------
    beta            : nominal operating-point parameter.
    distorted       : distorted-case points for every grid intensity below the
                      stability limit, plus one at the limit itself.
    uniform         : area-averaged uniform-loss points on the full grid.
    limit_intensity : largest intensity with a valid equilibrium, or ``None``
                      if the case stays valid up to ``max_intensity``.
    limit_reason    : why the valid range ends at ``limit_intensity``:
                      "surge" = distorted sector reached the surge line (the
                      stability limit); "choke" = the clean sectors ran out
                      of map at the choke bound first (so the surge limit
                      lies beyond what the map can represent); "fail" =
                      solver problem; ``None`` if there is no limit.
    """

    beta: float
    distorted: List[CasePoint] = field(default_factory=list)
    uniform: List[CasePoint] = field(default_factory=list)
    limit_intensity: Optional[float] = None
    limit_reason: Optional[str] = None


# =============================================================================
# Sweep driver
# =============================================================================


class StabilitySweep:
    """Runs the distorted / uniform comparison and finds the stability limits."""

    def __init__(self, compressor_map: CompressorMap, config: StabilityConfig) -> None:
        if not 0.0 < config.extent_deg < 360.0:
            raise ValueError("extent_deg must be strictly between 0 and 360")
        self.map: CompressorMap = compressor_map
        self.config: StabilityConfig = config

    # ---------------------------------------------------------------- layouts
    def _distorted_layout(self, intensity: float) -> Tuple[List[float], List[float]]:
        """
        Segment angles and inlet p0 for one low-pressure sector.

        The annulus is cut into equal cells whose size divides both the
        extent and 360 deg (e.g. 90 deg -> 4 cells, 1 distorted + 3 clean).
        Cells with identical inlet conditions behave identically in parallel
        compressor theory, so this is exactly the one-sector pattern, and it
        keeps every segment the same size.
        """
        cfg = self.config
        milli = int(round(cfg.extent_deg * 1000.0))
        cell = float(np.gcd(milli, 360_000)) / 1000.0
        n_cells = int(round(360.0 / cell))
        n_low = int(round(cfg.extent_deg / cell))
        p_low = cfg.p0_clean * (1.0 - intensity)
        p0s = [p_low] * n_low + [cfg.p0_clean] * (n_cells - n_low)
        return [cell] * n_cells, p0s

    def _uniform_layout(self, intensity: float) -> Tuple[List[float], List[float]]:
        """One full-annulus segment at the area-averaged inlet total pressure."""
        cfg = self.config
        p_avg = cfg.p0_clean * (1.0 - intensity * cfg.extent_deg / 360.0)
        return [360.0], [p_avg]

    # ----------------------------------------------------------------- solves
    def _solve(self, beta: float, intensity: float, uniform: bool) -> Optional[CasePoint]:
        """
        Solve one case. Returns ``None`` if the solver raised or did not
        converge. Otherwise returns the overall result with the ``stable``
        flag set from the surge-line and constraint-closure checks.
        """
        cfg = self.config
        angles, p0s = (
            self._uniform_layout(intensity) if uniform else self._distorted_layout(intensity)
        )
        try:
            model = ParallelCompressorModel(
                compressor_map=self.map,
                rotor_speed_rpm=cfg.speed_rpm,
                beta=beta,
                segment_angles_deg=angles,
                inlet_p0=p0s,
                inlet_T0=[cfg.T0_inlet] * len(angles),
                exit_area=cfg.exit_area,
            )
            # A non-converged solve emits RuntimeWarning -> treat as failure.
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                res = model.solve()
        except (ValueError, RuntimeWarning, FloatingPointError):
            return None

        segs = res.segments
        o = res.overall
        betas = np.array([s.beta for s in segs])
        p_static = np.array([s.p_static for s in segs])
        sm_min = min(s.surge_margin for s in segs)

        # Valid equilibrium = both coupling constraints actually satisfied
        # AND every segment strictly inside the map's beta range. When a
        # segment is pinned at a bound the bounded least-squares solution
        # misses at least one constraint, so closure alone is not enough
        # to say *which* edge of the map was hit -- check the betas too.
        target = model._target_mass_flow_actual  # set by solve()
        assert target is not None
        mass_ok = abs(o.mass_flow - target) / target < cfg.closure_tol
        static_ok = (p_static.max() - p_static.min()) / p_static.mean() < cfg.closure_tol
        beta_lo = float(self.map.beta_grid.min())
        beta_hi = float(self.map.beta_grid.max())
        at_surge = float(betas.min()) <= beta_lo + cfg.beta_tol
        at_choke = float(betas.max()) >= beta_hi - cfg.beta_tol
        if at_surge:
            status = "surge"  # surge is checked first: it is the stability limit
        elif at_choke:
            status = "choke"
        elif mass_ok and static_ok:
            status = "ok"
        else:
            status = "fail"

        p0_in_avg = float(sum(a * p for a, p in zip(angles, p0s)) / 360.0)
        if not all(np.isfinite(v) for v in (o.pressure_ratio, o.efficiency, o.p0)):
            return None
        return CasePoint(
            intensity=intensity,
            p0_in_avg=p0_in_avg,
            p0_out=o.p0,
            pressure_ratio=o.pressure_ratio,
            efficiency=o.efficiency,
            mass_flow=o.mass_flow,
            surge_margin=o.surge_margin,
            min_surge_margin=sm_min,
            beta_min=float(betas.min()),
            beta_max=float(betas.max()),
            status=status,
        )

    def _status(self, beta: float, intensity: float) -> str:
        """Status string of the distorted case ("fail" if the solve failed)."""
        pt = self._solve(beta, intensity, uniform=False)
        return "fail" if pt is None else pt.status

    def find_stability_limit(self, beta: float) -> Tuple[Optional[float], Optional[str]]:
        """
        Largest intensity with a valid equilibrium, found by bisection on
        [0, max_intensity], and the reason the valid range ends there
        ("surge", "choke" or "fail").

        Assumes validity is monotonic in intensity (a deeper pressure drop
        only pushes the distorted sector toward surge and the clean sectors
        toward choke). Returns ``(None, None)`` if the case is still valid
        at ``max_intensity`` and ``(0.0, reason)`` if it is invalid even
        without distortion.
        """
        hi = self.config.max_intensity
        s0 = self._status(beta, 0.0)
        if s0 != "ok":
            return 0.0, s0
        reason = self._status(beta, hi)
        if reason == "ok":
            return None, None
        lo = 0.0
        for _ in range(30):  # interval shrinks to max_intensity * 2^-30
            mid = 0.5 * (lo + hi)
            s = self._status(beta, mid)
            if s == "ok":
                lo = mid
            else:
                hi, reason = mid, s
        return lo, reason

    def run(self) -> List[BetaCurve]:
        """Compute the curves for every beta."""
        cfg = self.config
        grid = np.linspace(0.0, cfg.max_intensity, cfg.n_intensity)
        curves: List[BetaCurve] = []
        for beta in cfg.betas:
            curve = BetaCurve(beta=beta)
            curve.limit_intensity, curve.limit_reason = self.find_stability_limit(beta)
            lim = curve.limit_intensity

            # Distorted: grid points strictly below the limit + the limit itself.
            intensities: List[float] = [
                float(i) for i in grid if lim is None or i < lim - 1e-9
            ]
            if lim is not None:
                intensities.append(lim)
            for i in intensities:
                pt = self._solve(beta, i, uniform=False)
                if pt is not None:
                    curve.distorted.append(pt)

            # Uniform area-averaged loss: full grid (always above surge).
            for i in grid:
                pt = self._solve(beta, float(i), uniform=True)
                if pt is not None:
                    curve.uniform.append(pt)
            curves.append(curve)
        return curves


# =============================================================================
# Output: plots, CSV, console summary
# =============================================================================


class StabilityPlotter:
    """Draws the pressure-ratio, efficiency and stability-limit figures."""

    def __init__(self, curves: Sequence[BetaCurve], config: StabilityConfig) -> None:
        self.curves: Sequence[BetaCurve] = curves
        self.config: StabilityConfig = config
        self._colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(curves)))

    def _title_suffix(self) -> str:
        c = self.config
        return f"{c.speed_rpm:.0f} rpm, {c.extent_deg:g} deg sector"

    def _plot_metric(self, attr: str, ylabel: str, title: str) -> Figure:
        """
        One curve per beta: solid = distorted (cut at the stability limit,
        marked with an x), dashed = uniform area-averaged loss.
        """
        fig, ax = plt.subplots(figsize=(10.0, 5.2))
        for color, curve in zip(self._colors, self.curves):
            xd = [100.0 * p.intensity for p in curve.distorted]
            yd = [getattr(p, attr) for p in curve.distorted]
            xu = [100.0 * p.intensity for p in curve.uniform]
            yu = [getattr(p, attr) for p in curve.uniform]
            ax.plot(xd, yd, color=color, linewidth=1.6, marker="o", markersize=2.5,
                    label=f"beta = {curve.beta:g}")
            ax.plot(xu, yu, color=color, linewidth=1.2, linestyle="--")
            # Mark where the valid range of the distorted curve ends:
            # X = distorted sector on the surge line (stability limit),
            # hollow square = clean sectors at the choke end of the map.
            if curve.limit_intensity is not None and yd:
                if curve.limit_reason == "surge":
                    ax.plot(xd[-1], yd[-1], marker="X", markersize=9, color=color,
                            markeredgecolor="black", linestyle="none", zorder=5)
                else:
                    ax.plot(xd[-1], yd[-1], marker="s", markersize=7,
                            markerfacecolor="white", markeredgecolor=color,
                            markeredgewidth=1.6, linestyle="none", zorder=5)

        ax.set_xlabel("Inlet total pressure drop of distorted sector [%]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n({self._title_suffix()})")
        ax.grid(True, linewidth=0.4, alpha=0.6)

        # Colour legend (beta) + line-style legend (case) in one box.
        handles, labels = ax.get_legend_handles_labels()
        handles += [
            Line2D([], [], color="black", linewidth=1.6, label="distorted sector"),
            Line2D([], [], color="black", linewidth=1.2, linestyle="--",
                   label="uniform, area-averaged"),
            Line2D([], [], color="gray", marker="X", markeredgecolor="black",
                   markersize=9, linestyle="none", label="stability limit (surge)"),
            Line2D([], [], color="gray", marker="s", markerfacecolor="white",
                   markeredgecolor="gray", markersize=7, linestyle="none",
                   label="end of map (clean sectors choked)"),
        ]
        # Legend outside the axes so it never hides the curves.
        ax.legend(handles=handles, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        return fig

    def plot_pressure_ratio(self) -> Figure:
        return self._plot_metric("pressure_ratio", "Overall total pressure ratio [-]",
                                 "Overall pressure ratio: distorted vs. uniform loss")

    def plot_efficiency(self) -> Figure:
        return self._plot_metric("efficiency", "Overall isentropic efficiency [-]",
                                 "Overall efficiency: distorted vs. uniform loss")

    def plot_stability_limit(self) -> Figure:
        """
        Largest valid p0 drop vs. beta. Filled circles are true surge limits
        (line connects them); hollow squares = the map's choke end was
        reached first (surge limit lies beyond); hollow triangles = valid up
        to the largest intensity searched.
        """
        cfg = self.config
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        surge = [(c.beta, 100.0 * c.limit_intensity) for c in self.curves
                 if c.limit_reason == "surge" and c.limit_intensity is not None]
        if surge:
            ax.plot(*zip(*surge), color="tab:blue", linewidth=1.4, marker="o",
                    markersize=6, zorder=3)
        for c in self.curves:
            if c.limit_reason == "choke" and c.limit_intensity is not None:
                ax.plot(c.beta, 100.0 * c.limit_intensity, marker="s", markerfacecolor="white",
                        markeredgecolor="tab:blue", markersize=8, linestyle="none", zorder=3)
            elif c.limit_intensity is None:
                ax.plot(c.beta, 100.0 * cfg.max_intensity, marker="^", markerfacecolor="white",
                        markeredgecolor="tab:blue", markersize=8, linestyle="none", zorder=3)
        ax.set_xlabel("Nominal operating point beta [-]  (0 = surge, 1 = choke)")
        ax.set_ylabel("Max. p0 drop of the sector [%]")
        ax.set_title(f"Stability limit of the distorted sector\n({self._title_suffix()})")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, linewidth=0.4, alpha=0.6)
        ax.legend(handles=[
            Line2D([], [], color="tab:blue", marker="o", label="distorted sector at surge"),
            Line2D([], [], color="tab:blue", marker="s", markerfacecolor="white",
                   linestyle="none", label="clean sectors at choke first (no surge limit)"),
            Line2D([], [], color="tab:blue", marker="^", markerfacecolor="white",
                   linestyle="none", label=f"valid up to {100 * cfg.max_intensity:g} %"),
        ], fontsize=8, loc="best")
        fig.tight_layout()
        return fig


def write_csv(curves: Sequence[BetaCurve], path: Path) -> None:
    """Dump every solved point (long format) to CSV."""
    cols = ["beta", "case", "intensity", "p0_in_avg_Pa", "p0_out_Pa", "pressure_ratio",
            "efficiency", "mass_flow_kg_s", "surge_margin_overall_pct",
            "surge_margin_min_pct", "beta_local_min", "beta_local_max", "status"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in curves:
            for case, pts in (("distorted", c.distorted), ("uniform", c.uniform)):
                for p in pts:
                    w.writerow([c.beta, case, p.intensity, p.p0_in_avg, p.p0_out,
                                p.pressure_ratio, p.efficiency, p.mass_flow,
                                p.surge_margin, p.min_surge_margin, p.beta_min,
                                p.beta_max, p.status])


def print_limits(curves: Sequence[BetaCurve], config: StabilityConfig) -> None:
    """Console table of the stability limits and what ended each curve."""
    print(f"\nStability limit ({config.extent_deg:g} deg sector, {config.speed_rpm:.0f} rpm)")
    print(f"{'beta':>6} {'max p0 drop [%]':>16}  note")
    for c in curves:
        if c.limit_intensity is None:
            val, note = f"> {100 * config.max_intensity:g}", "valid over the whole range"
        else:
            val = f"{100 * c.limit_intensity:.3f}"
            note = {"surge": "distorted sector reaches surge (stability limit)",
                    "choke": "clean sectors hit choke end of map first",
                    "fail": "solver failure"}.get(c.limit_reason or "", "")
        print(f"{c.beta:6.2f} {val:>16}  {note}")


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
    p = argparse.ArgumentParser(description="Parallel-compressor stability-limit sweep")
    p.add_argument("--map", type=Path, default=_default_map_path(), help="path to .chC map")
    p.add_argument("--speed", type=float, default=21789.0, help="rotor speed [rpm]")
    p.add_argument("--betas", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--extent", type=float, default=90.0, help="distorted-sector extent [deg]")
    p.add_argument("--max-intensity", type=float, default=0.30,
                   help="largest fractional p0 drop (also stability-search bound)")
    p.add_argument("--n-intensity", type=int, default=31, help="grid points in [0, max]")
    p.add_argument("--p0", type=float, default=101_325.0, help="clean-sector p0 [Pa]")
    p.add_argument("--T0", type=float, default=288.15, help="inlet T0 [K]")
    p.add_argument("--exit-area", type=float, default=0.007, help="exit area [m^2]")
    p.add_argument("--outdir", type=Path, default=Path("."), help="where to save PNG/CSV")
    p.add_argument("--no-show", action="store_true", help="save only; do not open windows")
    args = p.parse_args(argv)

    cmap = read_hecc_chc(args.map)
    cfg = StabilityConfig(
        speed_rpm=args.speed,
        betas=tuple(args.betas),
        extent_deg=args.extent,
        max_intensity=args.max_intensity,
        n_intensity=args.n_intensity,
        p0_clean=args.p0,
        T0_inlet=args.T0,
        exit_area=args.exit_area,
    )
    curves = StabilitySweep(cmap, cfg).run()
    print_limits(curves, cfg)

    args.outdir.mkdir(parents=True, exist_ok=True)
    plotter = StabilityPlotter(curves, cfg)
    plotter.plot_pressure_ratio().savefig(args.outdir / "stab_pressure_ratio.png", dpi=130)
    plotter.plot_efficiency().savefig(args.outdir / "stab_efficiency.png", dpi=130)
    plotter.plot_stability_limit().savefig(args.outdir / "stab_limit_vs_beta.png", dpi=130)
    write_csv(curves, args.outdir / "stab_results.csv")
    print(f"\nSaved stab_pressure_ratio.png, stab_efficiency.png, stab_limit_vs_beta.png, "
          f"stab_results.csv to {args.outdir}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
