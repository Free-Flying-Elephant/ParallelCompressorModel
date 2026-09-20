"""
Multiple-segment parallel compressor solver.

See the package ``__init__.py`` for the theory background (Reid, 1969).
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, least_squares

from compressor_map import CompressorMap
from gas_properties import GasProperties, ReferenceConditions
from segments import ParallelCompressorResult, Segment, SegmentResult


class ParallelCompressorModel:
    """
    Multiple-segment parallel compressor model.

    Divides the compressor annulus into a variable number of angular
    segments, each operating independently on a shared ``CompressorMap`` but
    subject to a possibly non-uniform inlet total pressure and/or
    temperature. Segments are coupled by two constraints, solved
    simultaneously for each segment's local beta:

        1. All segments discharge to the same exit static pressure (they
           share a common downstream plenum/duct).
        2. The segments' actual mass flows sum to a single target total mass
           flow -- the flow a downstream throttle/duct imposes on the whole
           machine, evaluated from ``rotor_speed_rpm`` and the requested
           overall ``beta`` at mean (undistorted) inlet conditions. This is
           the "operating point" the machine would sit at if the inlet flow
           were uniform.
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
                              the common exit plane. Divided among segments
                              in proportion to their angular extent.
        """
        self._validate_inputs(segment_angles_deg, inlet_p0, inlet_T0)

        self.map: CompressorMap = compressor_map
        self.rotor_speed_rpm: float = rotor_speed_rpm
        self.beta: float = beta
        self.gas: GasProperties = gas
        self.reference: ReferenceConditions = reference
        self.exit_area: float = exit_area

        self.segments: List[Segment] = [
            Segment(angle_deg=float(a), p0_in=float(p), T0_in=float(t))
            for a, p, t in zip(segment_angles_deg, inlet_p0, inlet_T0)
        ]

        # Target total actual mass flow [kg/s] for the coupled solve; set by
        # solve() before the residual function is used (see
        # _target_total_mass_flow).
        self._target_mass_flow_actual: Optional[float] = None

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
            - all angles, pressures and temperatures are positive.
        Raises ``ValueError`` on any inconsistency.
        """
        n = len(segment_angles_deg)
        if n < 1:
            raise ValueError("At least one segment is required")
        if len(inlet_p0) != n or len(inlet_T0) != n:
            raise ValueError(
                f"segment_angles_deg ({n}), inlet_p0 ({len(inlet_p0)}) and "
                f"inlet_T0 ({len(inlet_T0)}) must all have the same length"
            )

        angle_sum = float(np.sum(segment_angles_deg))
        if not np.isclose(angle_sum, 360.0, atol=1e-6, rtol=0.0):
            raise ValueError(f"segment_angles_deg must sum to 360 degrees, got {angle_sum}")

        if any(a <= 0.0 for a in segment_angles_deg):
            raise ValueError("segment_angles_deg entries must all be positive")
        if any(p <= 0.0 for p in inlet_p0):
            raise ValueError("inlet_p0 entries must all be positive")
        if any(t <= 0.0 for t in inlet_T0):
            raise ValueError("inlet_T0 entries must all be positive")

    # -------------------------------------------------------------------
    # Corrected-parameter / compressible-flow helpers
    # -------------------------------------------------------------------
    def _corrected_speed(self, T0_in: float) -> float:
        """
        Corrected ("referred") rotor speed for a segment, given its inlet
        total temperature, referenced to standard-day conditions:
        N_corrected = N / sqrt(theta), theta = T0_in / T_ref.
        """
        theta = T0_in / self.reference.T_ref
        return self.rotor_speed_rpm / np.sqrt(theta)

    def _corrected_mass_flow_to_actual(
        self, corrected_mass_flow: float, p0_in: float, T0_in: float
    ) -> float:
        """
        Convert a corrected mass flow (as read from the compressor map) to
        actual mass flow for a segment's specific inlet conditions:
        m_actual = m_corrected * (p0_in / p_ref) / sqrt(T0_in / T_ref).
        """
        delta = p0_in / self.reference.p_ref
        theta = T0_in / self.reference.T_ref
        return corrected_mass_flow * delta / np.sqrt(theta)

    def _mass_flow_function_of_mach(
        self, mach: float, p0: float, T0: float, area: float
    ) -> float:
        """
        Compressible-flow mass flow as a function of Mach number for given
        total conditions and area (root-function target used inside
        ``_static_from_total``):

            m_dot(M) = (p0 * A / sqrt(T0)) * sqrt(gamma / R)
                       * M * (1 + (gamma - 1)/2 * M^2) ** (-(gamma+1)/(2*(gamma-1)))
        """
        gamma = self.gas.gamma
        R = self.gas.R
        term = 1.0 + 0.5 * (gamma - 1.0) * mach**2
        exponent = -(gamma + 1.0) / (2.0 * (gamma - 1.0))
        return (p0 * area / np.sqrt(T0)) * np.sqrt(gamma / R) * mach * term**exponent

    def _static_from_total(
        self, p0: float, T0: float, mass_flow: float, area: float
    ) -> Tuple[float, float]:
        """
        Resolve static pressure and static temperature from total
        conditions, mass flow, and flow area.

        Solves the compressible-flow continuity relation
        (m_dot = f(Mach, p0, T0, area, gas)) for Mach number using
        ``scipy.optimize.brentq`` over the subsonic branch (0, 1), then
        applies isentropic total/static relations to obtain static pressure
        and temperature.

        Returns
        -------
        (p_static, T_static)
        """
        gamma = self.gas.gamma
        mach_floor, mach_ceiling = 1e-6, 1.0 - 1e-9

        if mass_flow <= 0.0:
            mach = mach_floor
        else:
            m_choke = self._mass_flow_function_of_mach(mach_ceiling, p0, T0, area)
            if mass_flow >= m_choke:
                # Requested flow meets/exceeds the choking limit for this
                # area -- clip to just below Mach 1 rather than fail, since
                # this may occur transiently during the coupled iteration.
                mach = mach_ceiling
            else:
                def residual(m: float) -> float:
                    return self._mass_flow_function_of_mach(m, p0, T0, area) - mass_flow

                mach = brentq(residual, mach_floor, mach_ceiling)

        assert type(mach) is float
        term = 1.0 + 0.5 * (gamma - 1.0) * mach**2
        T_static = T0 / term
        p_static = p0 / term ** (gamma / (gamma - 1.0))
        return p_static, T_static

    # -------------------------------------------------------------------
    # Core coupled solve
    # -------------------------------------------------------------------
    def _target_total_mass_flow(self) -> float:
        """
        Total actual mass flow [kg/s] the whole compressor is targeted to
        deliver, representing the operating point set by a downstream
        throttle/duct.

        Evaluated from the shared rotor speed and the requested overall
        ``beta``, at the mean (arithmetic average) inlet total pressure and
        temperature across segments -- i.e. the flow the machine would pass
        at that beta if the inlet were uniform.
        """
        T0_nominal = float(np.mean([seg.T0_in for seg in self.segments]))
        p0_nominal = float(np.mean([seg.p0_in for seg in self.segments]))
        N_nominal = self._corrected_speed(T0_nominal)
        m_corrected_nominal = self.map.get_mass_flow(N_nominal, self.beta)
        return self._corrected_mass_flow_to_actual(m_corrected_nominal, p0_nominal, T0_nominal)

    def _segment_residuals(self, betas: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Residual vector passed to ``scipy.optimize.least_squares``.

        For a trial vector of per-segment local beta values, computes each
        segment's full performance (via ``_segment_performance``) and
        returns the residuals that must be driven to zero:
            - for segments 1..n-1: discharge static pressure minus segment
              0's discharge static pressure (equalizes exit static pressure
              across all segments).
            - the last entry: total actual mass flow across all segments
              minus the target total mass flow (``_target_total_mass_flow``),
              enforcing the overall operating-point / throttle constraint.

        The residual vector has exactly ``len(self.segments)`` entries,
        matching the number of unknown per-segment betas.
        """
        results = [
            self._segment_performance(seg, float(b)) for seg, b in zip(self.segments, betas)
        ]
        p_static = np.array([r.p_static for r in results], dtype=np.float64)
        total_mass_flow = sum(r.mass_flow for r in results)
        assert self._target_mass_flow_actual is not None  # set by solve()

        # Non-dimensionalize: the raw pressure residuals are O(1e5) Pa while
        # the raw mass-flow residual is O(1) kg/s. Without scaling,
        # least_squares treats the tiny (but not yet converged) mass-flow
        # residual as negligible next to the large pressure residuals and
        # can terminate on step-size tolerance before it is actually zeroed.
        pressure_scale = self.reference.p_ref
        mass_flow_scale = self._target_mass_flow_actual

        residuals = np.empty(len(self.segments), dtype=np.float64)
        residuals[:-1] = (p_static[1:] - p_static[0]) / pressure_scale
        residuals[-1] = (total_mass_flow - self._target_mass_flow_actual) / mass_flow_scale
        return residuals

    def solve(self) -> ParallelCompressorResult:
        """
        Solve the coupled multi-segment system and return per-segment and
        whole-compressor performance.

        Steps:
            1. Compute the target total mass flow (``_target_total_mass_flow``).
            2. Initialize per-segment beta guesses, all equal to ``self.beta``.
            3. Run ``scipy.optimize.least_squares`` on ``_segment_residuals``,
               bounded to the map's beta range, to find the per-segment beta
               vector satisfying the common exit static pressure and overall
               mass-flow constraints.
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
        self._target_mass_flow_actual = self._target_total_mass_flow()

        n = len(self.segments)
        betas0 = np.full(n, self.beta, dtype=np.float64)
        beta_lo = float(self.map.beta_grid.min())
        beta_hi = float(self.map.beta_grid.max())

        result = least_squares(
            self._segment_residuals,
            betas0,
            bounds=(beta_lo, beta_hi),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        if not result.success:
            warnings.warn(
                "Parallel-compressor coupled solve did not fully converge "
                f"(status={result.status}, message={result.message!r}); "
                "returning the best available result.",
                RuntimeWarning,
                stacklevel=2,
            )

        converged_betas = result.x
        segment_results = [
            self._segment_performance(seg, float(b))
            for seg, b in zip(self.segments, converged_betas)
        ]
        overall = self._overall_performance(segment_results)
        return ParallelCompressorResult(segments=segment_results, overall=overall)

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
        gamma = self.gas.gamma

        corrected_speed = self._corrected_speed(segment.T0_in)

        # The map gives the full-annulus (360 deg) corrected mass flow at
        # this (speed, beta); a segment spanning only part of the annulus
        # carries that same fraction of it (its blade-passage area is that
        # same fraction of the total annulus flow area).
        angle_fraction = segment.angle_deg / 360.0
        mass_flow_corrected_full = self.map.get_mass_flow(corrected_speed, beta_local)
        mass_flow_corrected_segment = mass_flow_corrected_full * angle_fraction
        mass_flow_actual = self._corrected_mass_flow_to_actual(
            mass_flow_corrected_segment, segment.p0_in, segment.T0_in
        )

        pressure_ratio = self.map.get_pressure_ratio(corrected_speed, beta_local)
        efficiency = self.map.get_efficiency(corrected_speed, beta_local)

        p0_out = segment.p0_in * pressure_ratio
        # Isentropic-efficiency-corrected total temperature rise:
        # T0_out = T0_in * (1 + (PR^((gamma-1)/gamma) - 1) / eta)
        T0_out = segment.T0_in * (
            1.0 + (pressure_ratio ** ((gamma - 1.0) / gamma) - 1.0) / efficiency
        )

        segment_area = self.exit_area * (segment.angle_deg / 360.0)
        p_static, T_static = self._static_from_total(
            p0_out, T0_out, mass_flow_actual, segment_area
        )

        surge_margin = self._surge_margin(corrected_speed, mass_flow_corrected_full)

        return SegmentResult(
            angle_deg=segment.angle_deg,
            p0=p0_out,
            p_static=p_static,
            T0=T0_out,
            T_static=T_static,
            mass_flow=mass_flow_actual,
            surge_margin=surge_margin,
            efficiency=efficiency,
            beta=beta_local,
        )

    def _surge_margin(self, corrected_speed: float, mass_flow_corrected: float) -> float:
        """
        Flow-based surge margin [%] for a segment at the given corrected
        speed and corrected mass flow, relative to the map's surge line
        (beta = 0) at that speed:

            SM = (m_corrected - m_surge_corrected) / m_surge_corrected * 100

        Positive values indicate operation away from surge (higher beta,
        more flow); values approaching zero indicate proximity to surge.
        """
        surge_mass_flow = self.map.get_surge_mass_flow(corrected_speed)
        return (mass_flow_corrected - surge_mass_flow) / surge_mass_flow * 100.0

    def _overall_performance(self, segment_results: List[SegmentResult]) -> SegmentResult:
        """
        Aggregate per-segment results into a single mass-flow-weighted
        whole-compressor performance record (total pressure, total
        temperature, static pressure, static temperature, mass flow,
        efficiency, surge margin, beta).

        - ``mass_flow`` : sum of segment mass flows.
        - ``p0``, ``T0``, ``T_static``, ``surge_margin`` : mass-flow-weighted
          averages across segments.
        - ``p_static`` : arithmetic mean of the (by construction, nearly
          equal) segment static pressures.
        - ``efficiency`` : "power-averaged" overall efficiency -- the ratio
          of the mass-flow-weighted ideal (isentropic) specific work to the
          mass-flow-weighted actual specific work, which reduces exactly to
          each segment's own efficiency definition applied at the aggregate
          level.
        - ``beta`` : the requested overall beta (the nominal operating point
          the target mass flow was evaluated at); segments individually sit
          at different local betas.
        """
        total_mass_flow = sum(r.mass_flow for r in segment_results)
        if total_mass_flow <= 0.0:
            raise ValueError("Total mass flow must be positive to compute overall performance")

        p0_overall = sum(r.mass_flow * r.p0 for r in segment_results) / total_mass_flow
        T0_overall = sum(r.mass_flow * r.T0 for r in segment_results) / total_mass_flow
        T_static_overall = sum(r.mass_flow * r.T_static for r in segment_results) / total_mass_flow
        p_static_overall = float(np.mean([r.p_static for r in segment_results]))
        surge_margin_overall = (
            sum(r.mass_flow * r.surge_margin for r in segment_results) / total_mass_flow
        )

        gamma = self.gas.gamma
        ideal_work = 0.0
        actual_work = 0.0
        for seg, r in zip(self.segments, segment_results):
            pressure_ratio = r.p0 / seg.p0_in
            ideal_i = seg.T0_in * (pressure_ratio ** ((gamma - 1.0) / gamma) - 1.0)
            actual_i = r.T0 - seg.T0_in
            ideal_work += r.mass_flow * ideal_i
            actual_work += r.mass_flow * actual_i
        efficiency_overall = ideal_work / actual_work if actual_work > 0.0 else float("nan")

        return SegmentResult(
            angle_deg=360.0,
            p0=p0_overall,
            p_static=p_static_overall,
            T0=T0_overall,
            T_static=T_static_overall,
            mass_flow=total_mass_flow,
            surge_margin=surge_margin_overall,
            efficiency=efficiency_overall,
            beta=self.beta,
        )
