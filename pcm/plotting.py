"""
Matplotlib visualization of a :class:`~parallel_compressor_model.compressor_map.CompressorMap`
(as constant-speedline curves) and of a
:class:`~parallel_compressor_model.segments.ParallelCompressorResult` overlaid
on it as operating points annotated by surge margin.

Either the pressure-ratio map or the efficiency map can be drawn (see the
``quantity`` parameter on both functions below), since both are plotted the
same way: a speedline table against the shared mass-flow grid, with solved
operating points overlaid.

This module is the one place in the package that depends on ``matplotlib``
(the rest of the package uses only NumPy/SciPy).
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from compressor_map import CompressorMap
from segments import ParallelCompressorResult, SegmentResult

# Which map quantity to plot on the y-axis. Both are plotted against the same
# (shared) mass-flow axis, so only the y-data source, label and title differ.
MapQuantity = Literal["pressure_ratio", "efficiency"]


def _quantity_table(compressor_map: CompressorMap, quantity: MapQuantity) -> np.ndarray:
    """The map's (n_speed, n_beta) table backing ``quantity``."""
    if quantity == "pressure_ratio":
        return compressor_map.pressure_ratio_table
    if quantity == "efficiency":
        return compressor_map.efficiency_table
    raise ValueError(f"Unknown quantity {quantity!r}; expected 'pressure_ratio' or 'efficiency'")


def _quantity_value(result: SegmentResult, quantity: MapQuantity) -> float:
    """The solved-result field backing ``quantity`` for a single segment/overall point."""
    if quantity == "pressure_ratio":
        return result.pressure_ratio
    if quantity == "efficiency":
        return result.efficiency
    raise ValueError(f"Unknown quantity {quantity!r}; expected 'pressure_ratio' or 'efficiency'")


def _quantity_label_and_title(quantity: MapQuantity) -> Tuple[str, str]:
    """(y-axis label, plot title) for ``quantity``."""
    if quantity == "pressure_ratio":
        return "Total-to-total pressure ratio [-]", "Compressor map -- pressure ratio"
    if quantity == "efficiency":
        return "Total-to-total isentropic efficiency [-]", "Compressor map -- efficiency"
    raise ValueError(f"Unknown quantity {quantity!r}; expected 'pressure_ratio' or 'efficiency'")


def plot_compressor_map(
    compressor_map: CompressorMap,
    ax: Optional[Axes] = None,
    speed_units: str = "RPM",
    quantity: MapQuantity = "pressure_ratio",
) -> Axes:
    """
    Plot a compressor map as constant-speedline curves (``quantity`` vs.
    mass flow), one line per row of ``compressor_map.corrected_speed_grid``.

    Each speedline is drawn by tracing the chosen quantity's table
    (``pressure_ratio_table`` or ``efficiency_table``) against
    ``mass_flow_table`` across the beta grid, with markers at the underlying
    beta grid points and a label giving the corrected speed.

    Parameters
    ----------
    compressor_map : map to plot.
    ax : existing Axes to draw into; a new figure/Axes is created if omitted.
    speed_units : unit string appended to each speedline's legend label.
    quantity : which map quantity to plot on the y-axis -- "pressure_ratio"
        (default) or "efficiency".

    Returns
    -------
    Axes
        The Axes the map was drawn on (for further composition, e.g. via
        ``plot_parallel_compressor_result``).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    y_table = _quantity_table(compressor_map, quantity)
    y_label, title = _quantity_label_and_title(quantity)

    for i, speed in enumerate(compressor_map.corrected_speed_grid):
        ax.plot(
            compressor_map.mass_flow_table[i, :],
            y_table[i, :],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=f"{speed:.0f} {speed_units}",
        )

    ax.set_xlabel("Corrected mass flow [kg/s]")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.legend(title="Corrected speed", fontsize=8, loc="best")
    return ax


def plot_parallel_compressor_result(
    result: ParallelCompressorResult,
    compressor_map: Optional[CompressorMap] = None,
    ax: Optional[Axes] = None,
    show_overall: bool = True,
    cmap_name: str = "RdYlGn",
    quantity: MapQuantity = "pressure_ratio",
) -> Axes:
    """
    Overlay a :class:`ParallelCompressorResult` onto a compressor map plot,
    one marker per segment (plus the overall point, by default), placed at
    each result's (``mass_flow_corrected``, <quantity>) -- i.e. its exact
    location on the shared map -- and annotated with its surge margin.

    Parameters
    ----------
    result : solved parallel-compressor result (``ParallelCompressorModel.solve()``).
    compressor_map : if given, the background speedlines are drawn first via
        ``plot_compressor_map`` (using the same ``quantity``). Omit to
        overlay onto an already-prepared ``ax`` (e.g. one from a prior
        ``plot_compressor_map`` call).
    ax : existing Axes to draw into; a new figure/Axes is created if omitted
        and ``compressor_map`` is also omitted.
    show_overall : if True (default), also plot the whole-compressor
        aggregate point (``result.overall``), styled distinctly from the
        per-segment points.
    cmap_name : Matplotlib colormap used to color points by surge margin
        (low surge margin -> red end, high surge margin -> green end, for
        the default "RdYlGn").
    quantity : which map quantity to plot the operating points against --
        "pressure_ratio" (default) or "efficiency". Must match the
        background map's ``quantity`` when both are supplied, or the points
        will not line up with the speedlines.

    Returns
    -------
    Axes
        The Axes the result was drawn on.
    """
    if ax is None:
        if compressor_map is not None:
            ax = plot_compressor_map(compressor_map, quantity=quantity)
        else:
            _, ax = plt.subplots(figsize=(7, 5))
    elif compressor_map is not None:
        plot_compressor_map(compressor_map, ax=ax, quantity=quantity)

    segments = result.segments
    surge_margins = np.array([r.surge_margin for r in segments], dtype=np.float64)

    # Color scale spans the surge margins actually present (padded slightly
    # so the most/least critical segment isn't drawn at the colormap's
    # extreme edge); falls back to a fixed [0, 30] % range if all segments
    # happen to share the same surge margin.
    sm_min, sm_max = float(surge_margins.min()), float(surge_margins.max())
    if np.isclose(sm_min, sm_max):
        sm_min, sm_max = 0.0, 30.0
    else:
        pad = 0.1 * (sm_max - sm_min)
        sm_min, sm_max = sm_min - pad, sm_max + pad
    norm = Normalize(vmin=sm_min, vmax=sm_max)
    cmap = plt.get_cmap(cmap_name)

    def _plot_point(r: SegmentResult, marker: str, size: float, label: Optional[str]) -> None:
        color = cmap(norm(r.surge_margin))
        y = _quantity_value(r, quantity)
        ax.scatter(
            r.mass_flow_corrected,
            y,
            s=size,
            marker=marker,
            c=[color],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=label,
        )
        ax.annotate(
            f"{r.surge_margin:.1f}%",
            xy=(r.mass_flow_corrected, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    for i, r in enumerate(segments):
        _plot_point(r, marker="o", size=70, label=f"Segment {i + 1} ({r.angle_deg:.0f} deg)")

    if show_overall:
        _plot_point(result.overall, marker="*", size=220, label="Overall")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Surge margin [%]")

    ax.legend(fontsize=8, loc="best")
    return ax
