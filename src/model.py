"""
Multiple-segment parallel compressor solver.

See the package ``__init__.py`` for the theory background (Reid, 1969).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, least_squares

from .compressor_map import CompressorMap
from .gas_properties import GasProperties, ReferenceConditions
from .segment import ParallelCompressorResult, Segment, SegmentResult


class ParallelCompressorModel:
    """
    Multiple-segment parallel compressor model.

    Divides the compressor annulus into a variable number of angular
    segments, each operating independently on a shared ``CompressorMap`` but
    subject to a possibly non-uniform inlet total pressure and/or
    temperature. Segments are coupled by (a) a common discharge static
    pressure and (b) a shared rotor speed / overall operating-point
    constraint, and are solved simultaneously for their individual
    operating points (local beta values).
    """

    def __init__(
        self,
        compressor_map: CompressorMap,
        rotor_speed_rpm: float,
        beta: float,
        segment_angles_deg: Sequence[float],
        inlet_p0: Sequence[float],
        inlet_T0: Sequence[float],
        gas: GasProperties = GasProperties(),
        reference: ReferenceConditions = ReferenceConditions(),
        exit_area: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        compressor_map     : shared compressor performance map used by every
                              segment.
        rotor_speed_rpm     : mechanical rotor speed [1/min], common to the
                              whole compressor (all segments).
        beta                : overall/reference beta parameter, locating the
                              nominal (undistorted) operating point on the
                              map; also used as the initial guess for every
                              segment's local beta.
        segment_angles_deg  : angular extent of each segment [deg]. Must sum
                              to 360.
        inlet_p0            : inlet total pressure per segment [Pa]. Same
                              length as ``segment_angles_deg``.
        inlet_T0            : inlet total temperature per segment [K]. Same
                              length as ``segment_angles_deg``.
        gas                 : working-fluid thermodynamic properties.
        reference           : reference (standard-day) conditions used for
                              speed/flow correction.
        exit_area           : discharge flow area [m^2], used to relate
                              static pressure/temperature to mass flow at
                              the common exit plane.
        """
        self._validate_inputs(segment_angles_deg, inlet_p0, inlet_T0)

        self.map: CompressorMap = compressor_map
        self.rotor_speed_rpm: float = rotor_speed_rpm
        self.beta: float = beta
        self.gas: GasProperties = gas
        self.reference: ReferenceConditions = reference
        self.exit_area: float = exit_area

        # TODO: build self.segments from the parallel input arrays.
        self.segments: List[Segment] = []
        ...

    # -------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------
    @staticmethod
    def _validate_inputs(
        segment_angles_deg: Sequence[float],
        inlet_p0: Sequence[float],
        inlet_T0: Sequence[float],
    ) -> None:
        """
        Validate that the segment input arrays are consistent:
            - ``segment_angles_deg``, ``inlet_p0``, ``inlet_T0`` are the same
              length (>= 1).
            - ``segment_angles_deg`` sums to 360 degrees.
        Raises ``ValueError`` on any inconsistency.
        """
        ...

    # -------------------------------------------------------------------
    # Corrected-parameter / compressible-flow helpers
    # -------------------------------------------------------------------
    def _corrected_speed(self, T0_in: float) -> float:
        """
        Corrected ("referred") rotor speed for a segment, given its inlet
        total temperature, referenced to standard-day conditions:
        N_corrected = N / sqrt(theta), theta = T0_in / T_ref.
        """
        ...

    def _corrected_mass_flow_to_actual(
        self, corrected_mass_flow: float, p0_in: float, T0_in: float
    ) -> float:
        """
        Convert a corrected mass flow (as read from the compressor map) to
        actual mass flow for a segment's specific inlet conditions:
        m_actual = m_corrected * (p0_in / p_ref) / sqrt(T0_in / T_ref).
        """
        ...

    def _static_from_total(
        self, p0: float, T0: float, mass_flow: float, area: float
    ) -> Tuple[float, float]:
        """
        Resolve static pressure and static temperature from total
        conditions, mass flow, and flow area.

        Solves the compressible-flow continuity relation
        (m_dot = f(Mach, p0, T0, area, gas)) for Mach number using
        ``scipy.optimize.brentq``, then applies isentropic total/static
        relations to obtain static pressure and temperature.

        Returns
        -------
        (p_static, T_static)
        """
        ...

    def _mass_flow_function_of_mach(
        self, mach: float, p0: float, T0: float, area: float
    ) -> float:
        """
        Compressible-flow mass flow as a function of Mach number for given
        total conditions and area (root-function target used inside
        ``_static_from_total``).
        """
        ...

    # -------------------------------------------------------------------
    # Core coupled solve
    # -------------------------------------------------------------------
    def _segment_residuals(self, betas: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Residual vector passed to ``scipy.optimize.least_squares``.

        For a trial vector of per-segment local beta values, computes each
        segment's discharge static pressure (and any overall mass-flow /
        operating-point constraint), and returns the residuals that must be
        driven to zero:
            - pairwise (or vs. common target) static pressure differences
              across segments.
            - overall consistency between the segment-aggregate mass flow /
              speed and the requested overall beta / operating point.
        """
        ...

    def solve(self) -> ParallelCompressorResult:
        """
        Solve the coupled multi-segment system and return per-segment and
        whole-compressor performance.

        Intended steps:
            1. Compute corrected speed for each segment from its inlet T0.
            2. Initialize per-segment beta guesses (e.g. all equal to
               ``self.beta``).
            3. Run ``scipy.optimize.least_squares`` on ``_segment_residuals``
               to find the per-segment beta vector satisfying the common
               exit static pressure and overall operating-point constraints.
            4. Compute full performance for each converged segment via
               ``_segment_performance``.
            5. Aggregate into whole-compressor performance via
               ``_overall_performance``.

        Returns
        -------
        ParallelCompressorResult
            Per-segment results (in input order) plus the whole-compressor
            aggregate.
        """
        ...

    # -------------------------------------------------------------------
    # Per-segment / overall performance
    # -------------------------------------------------------------------
    def _segment_performance(self, segment: Segment, beta_local: float) -> SegmentResult:
        """
        Compute all required outputs for a single segment at its converged
        local beta value: total pressure, static pressure, total
        temperature, static temperature, mass flow, surge margin,
        efficiency, and beta.
        """
        ...

    def _surge_margin(self, corrected_speed: float, mass_flow_corrected: float) -> float:
        """
        Surge margin for a segment at the given corrected speed and
        corrected mass flow, relative to the map's surge line at that speed.
        """
        ...

    def _overall_performance(self, segment_results: List[SegmentResult]) -> SegmentResult:
        """
        Aggregate per-segment results into a single mass-flow-weighted
        whole-compressor performance record (total pressure, total
        temperature, static pressure, static temperature, mass flow,
        efficiency, surge margin, beta).
        """
        ...
