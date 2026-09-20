"""
Compressor performance map: storage and interpolation over the
(corrected speed, beta) grid.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator


class CompressorMap:
    """
    Compressor performance map, parameterized by corrected rotor speed and
    the auxiliary "beta" coordinate (beta = 0 on the surge line, beta = 1 on
    the choke line of each speed line).

    Map data is supplied on a regular (corrected_speed, beta) grid; queries
    at arbitrary (corrected_speed, beta) points are resolved by bilinear
    interpolation via ``scipy.interpolate.RegularGridInterpolator``.
    """

    def __init__(
        self,
        corrected_speed_grid: NDArray[np.float64],
        beta_grid: NDArray[np.float64],
        mass_flow: NDArray[np.float64],
        pressure_ratio: NDArray[np.float64],
        efficiency: NDArray[np.float64],
    ) -> None:
        """
        Parameters
        ----------
        corrected_speed_grid : 1-D array, shape (n_speed,). Grid of corrected
            rotor speeds spanning the map, strictly increasing.
        beta_grid : 1-D array, shape (n_beta,). Grid of beta values spanning
            each speed line, strictly increasing (conventionally 0..1, with
            0 = surge and 1 = choke).
        mass_flow : 2-D array, shape (n_speed, n_beta). Corrected mass flow
            at each (speed, beta) grid point [kg/s].
        pressure_ratio : 2-D array, shape (n_speed, n_beta). Total-to-total
            pressure ratio at each grid point [-].
        efficiency : 2-D array, shape (n_speed, n_beta). Total-to-total
            isentropic efficiency at each grid point [-].

        Raises
        ------
        ValueError
            If any table's shape is inconsistent with the two grids, or if
            either grid is not strictly increasing.
        """
        corrected_speed_grid = np.asarray(corrected_speed_grid, dtype=np.float64)
        beta_grid = np.asarray(beta_grid, dtype=np.float64)
        mass_flow = np.asarray(mass_flow, dtype=np.float64)
        pressure_ratio = np.asarray(pressure_ratio, dtype=np.float64)
        efficiency = np.asarray(efficiency, dtype=np.float64)

        self._validate_grids(corrected_speed_grid, beta_grid)
        self._validate_table_shapes(
            corrected_speed_grid,
            beta_grid,
            mass_flow=mass_flow,
            pressure_ratio=pressure_ratio,
            efficiency=efficiency,
        )

        self.corrected_speed_grid: NDArray[np.float64] = corrected_speed_grid
        self.beta_grid: NDArray[np.float64] = beta_grid
        self.mass_flow_table: NDArray[np.float64] = mass_flow
        self.pressure_ratio_table: NDArray[np.float64] = pressure_ratio
        self.efficiency_table: NDArray[np.float64] = efficiency

        # Interpolator handles, built below by _build_interpolators().
        self._mass_flow_interp: Optional[RegularGridInterpolator] = None
        self._pressure_ratio_interp: Optional[RegularGridInterpolator] = None
        self._efficiency_interp: Optional[RegularGridInterpolator] = None

        self._build_interpolators()

    # -------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------
    @staticmethod
    def _validate_grids(
        corrected_speed_grid: NDArray[np.float64], beta_grid: NDArray[np.float64]
    ) -> None:
        """Check that both grids are 1-D and strictly increasing."""
        if corrected_speed_grid.ndim != 1 or corrected_speed_grid.size < 1:
            raise ValueError("corrected_speed_grid must be a non-empty 1-D array")
        if beta_grid.ndim != 1 or beta_grid.size < 1:
            raise ValueError("beta_grid must be a non-empty 1-D array")
        if corrected_speed_grid.size > 1 and np.any(np.diff(corrected_speed_grid) <= 0):
            raise ValueError("corrected_speed_grid must be strictly increasing")
        if beta_grid.size > 1 and np.any(np.diff(beta_grid) <= 0):
            raise ValueError("beta_grid must be strictly increasing")

    @staticmethod
    def _validate_table_shapes(
        corrected_speed_grid: NDArray[np.float64],
        beta_grid: NDArray[np.float64],
        **tables: NDArray[np.float64],
    ) -> None:
        """Check that every named 2-D table matches (n_speed, n_beta)."""
        expected_shape: Tuple[int, int] = (corrected_speed_grid.size, beta_grid.size)
        for name, table in tables.items():
            if table.shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {table.shape}, expected {expected_shape} "
                    f"= (len(corrected_speed_grid), len(beta_grid))"
                )

    # -------------------------------------------------------------------
    # Interpolator setup
    # -------------------------------------------------------------------
    def _build_interpolators(self) -> None:
        """
        Build the ``RegularGridInterpolator`` instances for mass flow,
        pressure ratio and efficiency over the (corrected_speed, beta) grid.
        Called once from ``__init__``.

        ``bounds_error=False`` with ``fill_value=None`` enables linear
        extrapolation slightly outside the grid, since the coupled
        per-segment solve (``scipy.optimize.least_squares``) may probe beta
        values marginally outside [beta_grid.min(), beta_grid.max()] during
        iteration.
        """
        grid = (self.corrected_speed_grid, self.beta_grid)
        self._mass_flow_interp = RegularGridInterpolator(
            grid, self.mass_flow_table, method="linear", bounds_error=False, fill_value=np.nan
        )
        self._pressure_ratio_interp = RegularGridInterpolator(
            grid, self.pressure_ratio_table, method="linear", bounds_error=False, fill_value=np.nan
        )
        self._efficiency_interp = RegularGridInterpolator(
            grid, self.efficiency_table, method="linear", bounds_error=False, fill_value=np.nan
        )

    # -------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------
    def get_pressure_ratio(self, corrected_speed: float, beta: float) -> float:
        """Interpolate total-to-total pressure ratio at (corrected_speed, beta)."""
        assert self._pressure_ratio_interp is not None  # built in __init__
        return float(self._pressure_ratio_interp([[corrected_speed, beta]])[0])

    def get_efficiency(self, corrected_speed: float, beta: float) -> float:
        """Interpolate total-to-total isentropic efficiency at (corrected_speed, beta)."""
        assert self._efficiency_interp is not None  # built in __init__
        return float(self._efficiency_interp([[corrected_speed, beta]])[0])

    def get_mass_flow(self, corrected_speed: float, beta: float) -> float:
        """Interpolate corrected mass flow at (corrected_speed, beta)."""
        assert self._mass_flow_interp is not None  # built in __init__
        return float(self._mass_flow_interp([[corrected_speed, beta]])[0])

    def get_surge_mass_flow(self, corrected_speed: float) -> float:
        """
        Corrected mass flow on the surge line (beta = beta_grid.min(), i.e.
        beta = 0 by convention) at the given corrected speed. Used as the
        reference flow for surge margin.
        """
        surge_beta = float(self.beta_grid.min())
        return self.get_mass_flow(corrected_speed, surge_beta)

    def get_surge_pressure_ratio(self, corrected_speed: float) -> float:
        """
        Total-to-total pressure ratio on the surge line (beta =
        beta_grid.min(), i.e. beta = 0 by convention) at the given corrected
        speed. Used as the reference pressure ratio for surge margin
        (pressure-ratio-based definition).
        """
        surge_beta = float(self.beta_grid.min())
        return self.get_pressure_ratio(corrected_speed, surge_beta)
