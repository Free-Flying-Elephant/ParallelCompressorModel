"""
I/O utilities for reading compressor maps from vendor/legacy file formats and
converting them onto the regular (corrected_speed, beta) grid required by
:class:`~parallel_compressor_model.compressor_map.CompressorMap`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from compressor_map import CompressorMap

# 1 lbm/s in kg/s -- used to convert the mass-flow column of .chC files
# (given in lb/s) to SI units, consistent with the rest of the model.
_LBM_S_TO_KG_S: float = 0.45359237


def _parse_hecc_chc(path: Union[str, Path]) -> List[Tuple[float, NDArray[np.float64]]]:
    """
    Parse a ``.chC`` "constant speedlines" compressor map file.

    File format
    -----------
    Line 1            : number of speedlines, ``n_speedlines``.
    For each speedline:
        - one line   : ``rpm  n_points``
        - ``n_points`` lines, each: ``mass_flow[lb/s]  pressure_ratio[-]  efficiency[-]``

    Points within a speedline are stored in the file in decreasing-mass-flow
    / increasing-pressure-ratio order, i.e. from the choke end of the
    speedline (first point in the file) to the surge end (last point).

    Whitespace (spaces or tabs, any amount) separates the numbers, so the
    file is tokenized as a flat whitespace-delimited stream rather than
    parsed column-by-column.

    Parameters
    ----------
    path : path to the ``.chC`` file.

    Returns
    -------
    List of ``(rpm, points)`` tuples, one per speedline (file order), where
    ``points`` is an array of shape ``(n_points, 3)`` with columns
    ``[mass_flow, pressure_ratio, efficiency]``, in file order
    (choke -> surge).
    """
    tokens = Path(path).read_text().split()
    pos = 0

    n_speedlines = int(tokens[pos])
    pos += 1

    speedlines: List[Tuple[float, NDArray[np.float64]]] = []
    for _ in range(n_speedlines):
        rpm = float(tokens[pos])
        n_points = int(tokens[pos + 1])
        pos += 2

        points = np.empty((n_points, 3), dtype=np.float64)
        for i in range(n_points):
            points[i, 0] = float(tokens[pos])  # mass flow [lb/s]
            points[i, 1] = float(tokens[pos + 1])  # pressure ratio [-]
            points[i, 2] = float(tokens[pos + 2])  # efficiency [-]
            pos += 3

        speedlines.append((rpm, points))

    return speedlines


def read_hecc_chc(
    path: Union[str, Path],
    n_beta: Optional[int] = None,
    convert_mass_flow_to_si: bool = True,
) -> CompressorMap:
    """
    Read a ``.chC`` constant-speedline compressor map file and convert it
    into a :class:`CompressorMap` on a regular (corrected_speed, beta) grid.

    Each speedline in the source file has its own, independent number of
    points and its own mass-flow range, so the raw file cannot be used
    directly as a regular grid (``RegularGridInterpolator`` requires one).
    This function:

        1. Parses every speedline (rpm, and its (mass_flow, PR, eff) points)
           via :func:`_parse_hecc_chc`.
        2. Assigns each point within a speedline a "raw beta" coordinate
           based on its position in the file: beta = 1 at the choke end
           (first point), beta = 0 at the surge end (last point) -- matching
           the beta convention documented on :class:`CompressorMap`
           (beta = 0 on the surge line, beta = 1 on the choke line).
        3. Interpolates (1-D, per speedline, via ``numpy.interp``) mass
           flow, pressure ratio and efficiency onto a common beta grid
           shared by every speedline (``n_beta`` points, evenly spaced
           over [0, 1]).
        4. Assembles the resulting regular grids into a ``CompressorMap``.

    Parameters
    ----------
    path : path to the ``.chC`` file.
    n_beta : number of points in the common beta grid. Defaults to the
        largest number of points found on any single speedline in the
        file, so no resolution is lost on the most-refined speedline.
    convert_mass_flow_to_si : if True (default), convert the file's mass
        flow column from lb/s to kg/s. If False, mass flow is left in lb/s
        as read from the file.

    Returns
    -------
    CompressorMap
        Compressor map with ``corrected_speed_grid`` set to the RPM values
        read from the file (file order) and ``beta_grid`` set to
        ``np.linspace(0.0, 1.0, n_beta)``.
    """
    speedlines = _parse_hecc_chc(path)
    n_speedlines = len(speedlines)

    if n_beta is None:
        n_beta = max(points.shape[0] for _, points in speedlines)
        if n_beta is None: raise TypeError("Nember of Beta Points is NoneType") 

    corrected_speed_grid = np.array([rpm for rpm, _ in speedlines], dtype=np.float64)
    beta_grid = np.linspace(0.0, 1.0, n_beta)

    mass_flow = np.empty((n_speedlines, n_beta), dtype=np.float64)
    pressure_ratio = np.empty((n_speedlines, n_beta), dtype=np.float64)
    efficiency = np.empty((n_speedlines, n_beta), dtype=np.float64)

    for row, (_, points) in enumerate(speedlines):
        n_points = points.shape[0]

        # Raw beta per point, in file order: 1.0 (choke) -> 0.0 (surge).
        raw_beta = np.linspace(1.0, 0.0, n_points)

        # np.interp requires its x-coordinates in increasing order, so flip
        # both the raw beta coordinate and the data before interpolating.
        raw_beta_inc = raw_beta[::-1]
        mass_flow[row, :] = np.interp(beta_grid, raw_beta_inc, points[::-1, 0])
        pressure_ratio[row, :] = np.interp(beta_grid, raw_beta_inc, points[::-1, 1])
        efficiency[row, :] = np.interp(beta_grid, raw_beta_inc, points[::-1, 2])

    if convert_mass_flow_to_si:
        mass_flow = mass_flow * _LBM_S_TO_KG_S

    return CompressorMap(
        corrected_speed_grid=corrected_speed_grid,
        beta_grid=beta_grid,
        mass_flow=mass_flow,
        pressure_ratio=pressure_ratio,
        efficiency=efficiency,
    )
