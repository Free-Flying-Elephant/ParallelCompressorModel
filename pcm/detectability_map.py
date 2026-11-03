"""
Detectability study: how large must a total-pressure distortion be (extent x
intensity, at which operating point) before its effect on overall pressure
ratio and efficiency can be told apart from a plain uniform pressure loss?

For every (beta, extent, intensity):
    distorted case : one low-pressure sector (see stability_sweep.py)
    uniform case   : whole annulus at the SAME area-averaged inlet p0
and the "distortion-induced loss" is the difference between the two:
    dPR  = 100 * (PR_uniform  - PR_distorted) / PR_uniform   [% of PR]
    dEta = 100 * (eta_uniform - eta_distorted)               [points]

It also runs a consistency check of the corrected-parameter treatment: in the
model a uniform p0 loss must leave PR and eta untouched and scale the mass
flow with delta = p0 / p_ref.

Usage (from the directory with this file and hecc.chC):
    python detectability_map.py --no-show
    python detectability_map.py --speed 20670 --betas 0.3 0.5 0.7 \
        --pr-threshold 0.5 --eta-threshold 0.5 --no-show
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from compressor_map import CompressorMap  # noqa: E402
from map_io import read_hecc_chc  # noqa: E402
from stability_sweep import CasePoint, StabilityConfig, StabilitySweep  # noqa: E402


# =============================================================================
# Configuration and results
# =============================================================================


@dataclass(frozen=True)
class DetectConfig:
    """
    Parameters of the detectability study.

    Attributes
    ----------
    speed_rpm, betas, extents_deg : operating points and sector widths to run.
    max_intensity, n_intensity    : intensity grid, evenly spaced in [0, max].
    pr_threshold_pct              : smallest pressure-ratio difference [% of PR]
                                    considered distinguishable (measurement /
                                    CFD uncertainty -- replace with your own).
    eta_threshold_pts             : same for efficiency [percentage points].
    p0_clean, T0_inlet, exit_area : as in StabilityConfig.
    """

    speed_rpm: float
    betas: Tuple[float, ...]
    extents_deg: Tuple[float, ...]
    max_intensity: float = 0.20
    n_intensity: int = 21
    pr_threshold_pct: float = 0.5
    eta_threshold_pts: float = 0.5
    p0_clean: float = 101_325.0
    T0_inlet: float = 288.15
    exit_area: float = 0.005


@dataclass
class DetectResult:
    """
    Arrays are indexed [beta, extent, intensity]. Loss arrays are NaN for
    points that are not a valid equilibrium (surge/choke reached).
    """

    config: DetectConfig
    intensities: NDArray[np.float64]
    loss_pr_pct: NDArray[np.float64]
    loss_eta_pts: NDArray[np.float64]
    status: List[List[List[str]]]
    surge_limit: NDArray[np.float64]  # [beta, extent]; NaN if none up to max
    limit_reason: List[List[Optional[str]]]


# =============================================================================
# Study driver
# =============================================================================


class DetectabilityStudy:
    """Runs the distorted/uniform pairs and the corrected-parameter check."""

    def __init__(self, compressor_map: CompressorMap, config: DetectConfig) -> None:
        self.map = compressor_map
        self.config = config

    def _sweep(self, beta: float, extent: float) -> StabilitySweep:
        """A StabilitySweep configured for one (beta, extent) pair."""
        c = self.config
        return StabilitySweep(self.map, StabilityConfig(
            speed_rpm=c.speed_rpm, betas=(beta,), extent_deg=extent,
            max_intensity=c.max_intensity, n_intensity=c.n_intensity,
            p0_clean=c.p0_clean, T0_inlet=c.T0_inlet, exit_area=c.exit_area))

    def run(self) -> DetectResult:
        c = self.config
        grid = np.linspace(0.0, c.max_intensity, c.n_intensity)
        shape = (len(c.betas), len(c.extents_deg), len(grid))
        d_pr = np.full(shape, np.nan)
        d_eta = np.full(shape, np.nan)
        status = [[["" for _ in grid] for _ in c.extents_deg] for _ in c.betas]
        limit = np.full(shape[:2], np.nan)
        reason: List[List[Optional[str]]] = [[None] * shape[1] for _ in range(shape[0])]

        for ib, beta in enumerate(c.betas):
            for ie, ext in enumerate(c.extents_deg):
                sw = self._sweep(beta, ext)
                lim, why = sw.find_stability_limit(beta)  # precise limit by bisection
                if lim is not None:
                    limit[ib, ie] = lim
                reason[ib][ie] = why
                for ii, inten in enumerate(grid):
                    # _solve is package-internal; reuse it so both studies
                    # share exactly the same case definition and checks.
                    pd = sw._solve(beta, float(inten), uniform=False)
                    pu = sw._solve(beta, float(inten), uniform=True)
                    if pd is None or pu is None:
                        status[ib][ie][ii] = "fail"
                        continue
                    status[ib][ie][ii] = pd.status
                    if pd.status == "ok":
                        d_pr[ib, ie, ii] = 100.0 * (pu.pressure_ratio - pd.pressure_ratio) / pu.pressure_ratio
                        d_eta[ib, ie, ii] = 100.0 * (pu.efficiency - pd.efficiency)
        return DetectResult(c, grid, d_pr, d_eta, status, limit, reason)

    def corrected_parameter_check(self) -> List[Tuple[float, float, float, float]]:
        """
        For each beta, run the UNIFORM case over the whole intensity grid and
        return (beta, max |PR - PR(0)|, max |eta - eta(0)|, max relative
        deviation of mass_flow / delta from its value at zero loss).
        Exactly zero (to solver tolerance) means the map is used purely in
        corrected variables.
        """
        c = self.config
        out: List[Tuple[float, float, float, float]] = []
        print(c.betas)
        for beta in c.betas:
            print(beta)
            sw = self._sweep(beta, c.extents_deg[0])
            pts: List[CasePoint] = []
            for inten in np.linspace(0.0, c.max_intensity, c.n_intensity):
                p = sw._solve(beta, float(inten), uniform=True)
                if p is not None:
                    pts.append(p)
            pr0, eta0 = pts[0].pressure_ratio, pts[0].efficiency
            # p_ref is the model's reference pressure (T0 = T_ref here -> theta = 1)
            md = np.array([p.mass_flow / (p.p0_in_avg / 101_325.0) for p in pts])
            out.append((beta,
                        max(abs(p.pressure_ratio - pr0) for p in pts),
                        max(abs(p.efficiency - eta0) for p in pts),
                        float(np.max(np.abs(md / md[0] - 1.0)))))
        return out


# =============================================================================
# Reporting
# =============================================================================


def first_crossing(x: NDArray[np.float64], y: NDArray[np.float64], thr: float) -> Optional[float]:
    """Smallest x where y reaches thr (linear interpolation); NaNs ignored."""
    ok = np.isfinite(y)
    xs, ys = x[ok], y[ok]
    for k in range(len(xs)):
        if ys[k] >= thr:
            if k == 0:
                return float(xs[0])
            return float(xs[k - 1] + (thr - ys[k - 1]) * (xs[k] - xs[k - 1]) / (ys[k] - ys[k - 1]))
    return None


def summary_rows(res: DetectResult) -> List[List[str]]:
    """Per (beta, extent): intensity needed for each threshold vs. the limit."""
    c = res.config
    rows: List[List[str]] = []
    for ib, beta in enumerate(c.betas):
        for ie, ext in enumerate(c.extents_deg):
            i_pr = first_crossing(res.intensities, res.loss_pr_pct[ib, ie], c.pr_threshold_pct)
            i_eta = first_crossing(res.intensities, res.loss_eta_pts[ib, ie], c.eta_threshold_pts)
            lim = res.surge_limit[ib, ie]
            why = res.limit_reason[ib][ie]

            def f(v: Optional[float]) -> str:
                return "-" if v is None else f"{100 * v:.1f}"

            end = "-" if not np.isfinite(lim) else f"{100 * lim:.1f} ({why})"
            rows.append([f"{beta:g}", f"{ext:g}", f(i_pr), f(i_eta), end])
    return rows


def print_summary(res: DetectResult, check: Sequence[Tuple[float, float, float, float]]) -> None:
    c = res.config
    print(f"\nCorrected-parameter check, uniform p0 loss 0..{100 * c.max_intensity:g} % "
          f"({c.speed_rpm:.0f} rpm):")
    print(f"{'beta':>6} {'max|dPR|':>10} {'max|deta|':>10} {'max dev (mdot/delta)':>22}")
    for b, dpr, deta, dm in check:
        print(f"{b:6.2f} {dpr:10.2e} {deta:10.2e} {dm:22.2e}")
    print(f"\nIntensity [% p0 drop] needed for  dPR >= {c.pr_threshold_pct:g} %  /  "
          f"deta >= {c.eta_threshold_pts:g} pt,  vs. end of valid range:")
    print(f"{'beta':>6} {'extent':>7} {'for dPR':>8} {'for deta':>9}  limit (reason)")
    for r in summary_rows(res):
        print(f"{r[0]:>6} {r[1]:>7} {r[2]:>8} {r[3]:>9}  {r[4]}")


def write_csv(res: DetectResult, path: Path) -> None:
    c = res.config
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beta", "extent_deg", "intensity", "loss_PR_pct", "loss_eta_pts", "status"])
        for ib, b in enumerate(c.betas):
            for ie, e in enumerate(c.extents_deg):
                for ii, inten in enumerate(res.intensities):
                    w.writerow([b, e, inten, res.loss_pr_pct[ib, ie, ii],
                                res.loss_eta_pts[ib, ie, ii], res.status[ib][ie][ii]])


def plot_loss(res: DetectResult, quantity: str) -> Figure:
    """
    One panel per beta; x = intensity, one curve per extent; dashed line =
    detection threshold; X = distorted sector reached surge, hollow square =
    clean sectors reached choke (end of valid range).
    """
    c = res.config
    is_pr = quantity == "pr"
    data = res.loss_pr_pct if is_pr else res.loss_eta_pts
    thr = c.pr_threshold_pct if is_pr else c.eta_threshold_pts
    ylabel = "PR loss vs. uniform [% of PR]" if is_pr else "Efficiency loss vs. uniform [pts]"
    fig, axes = plt.subplots(1, len(c.betas), figsize=(4.6 * len(c.betas), 4.2),
                             squeeze=False, sharex=True)
    colors = plt.get_cmap("plasma")(np.linspace(0.0, 0.85, len(c.extents_deg)))
    for ib, beta in enumerate(c.betas):
        ax = axes[0][ib]
        for ie, ext in enumerate(c.extents_deg):
            y = data[ib, ie]
            ok = np.isfinite(y)
            ax.plot(100 * res.intensities[ok], y[ok], color=colors[ie], marker="o",
                    markersize=2.5, linewidth=1.4, label=f"{ext:g} deg")
            if ok.any() and res.limit_reason[ib][ie] is not None:
                last = np.where(ok)[0][-1]
                if res.limit_reason[ib][ie] == "surge":
                    ax.plot(100 * res.intensities[last], y[last], marker="X", markersize=9,
                            color=colors[ie], markeredgecolor="black", linestyle="none", zorder=5)
                else:
                    ax.plot(100 * res.intensities[last], y[last], marker="s", markersize=7,
                            markerfacecolor="white", markeredgecolor=colors[ie],
                            markeredgewidth=1.5, linestyle="none", zorder=5)
        ax.axhline(thr, color="gray", linestyle="--", linewidth=1.0)
        ax.set_title(f"beta = {beta:g}", fontsize=10)
        ax.set_xlabel("p0 drop of distorted sector [%]")
        ax.grid(True, linewidth=0.4, alpha=0.6)
        if ib == 0:
            ax.set_ylabel(ylabel)
            ax.legend(title="Extent", fontsize=8, loc="upper left")
    fig.suptitle(f"Distortion-induced loss vs. uniform loss at equal area-averaged p0 "
                 f"({c.speed_rpm:.0f} rpm); dashed = threshold; X = surge, square = choke",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# =============================================================================
# Command line
# =============================================================================


def _default_map_path() -> Path:
    for cand in (_HERE / "hecc.chC", _HERE.parent / "hecc.chC", Path.cwd() / "hecc.chC"):
        if cand.exists():
            return cand
    return _HERE / "hecc.chC"


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Distortion vs. uniform-loss detectability study")
    p.add_argument("--map", type=Path, default=_default_map_path())
    p.add_argument("--speed", type=float, default=21789.0, help="rotor speed [rpm]")
    p.add_argument("--betas", type=float, nargs="+", default=[0.5, 0.7])
    p.add_argument("--extents", type=float, nargs="+", default=[60, 90, 120, 180])
    p.add_argument("--max-intensity", type=float, default=0.10)
    p.add_argument("--n-intensity", type=int, default=11)
    p.add_argument("--pr-threshold", type=float, default=0.5, help="[%% of PR]")
    p.add_argument("--eta-threshold", type=float, default=0.5, help="[points]")
    p.add_argument("--exit-area", type=float, default=0.007)
    p.add_argument("--outdir", type=Path, default=Path("."))
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args(argv)

    cfg = DetectConfig(
        speed_rpm=args.speed, betas=tuple(args.betas), extents_deg=tuple(args.extents),
        max_intensity=args.max_intensity, n_intensity=args.n_intensity,
        pr_threshold_pct=args.pr_threshold, eta_threshold_pts=args.eta_threshold,
        exit_area=args.exit_area)
    study = DetectabilityStudy(read_hecc_chc(args.map), cfg)
    check = study.corrected_parameter_check()
    res = study.run()
    print_summary(res, check)

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_loss(res, "pr").savefig(args.outdir / "detect_pressure_ratio.png", dpi=130)
    plot_loss(res, "eta").savefig(args.outdir / "detect_efficiency.png", dpi=130)
    write_csv(res, args.outdir / "detect_results.csv")
    print(f"\nSaved detect_pressure_ratio.png, detect_efficiency.png, detect_results.csv to {args.outdir}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
