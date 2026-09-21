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
        inlet_area: float = 1.0,
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
                              to 360. Segments are assumed arranged
                              sequentially around the annulus in the order
                              given (segment 0 starting at a reference angle
                              of 0 deg) -- this ordering matters for the
                              DC(theta) distortion descriptors, which depend
                              on circumferential position, not just extent.
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
        inlet_area          : compressor face (inlet) flow area [m^2], used
                              only by the DC(theta) distortion descriptors
                              (``distortion_coefficient``, ``dc60``,
                              ``dc90``) to compute the average inlet dynamic
                              pressure. Distinct from ``exit_area``.
        """
        self._validate_inputs(segment_angles_deg, inlet_p0, inlet_T0)

        self.map: CompressorMap = compressor_map
        self.rotor_speed_rpm: float = rotor_speed_rpm
        self.beta: float = beta
        self.gas: GasProperties = gas
        self.reference: ReferenceConditions = reference
        self.exit_area: float = exit_area
        self.inlet_area: float = inlet_area

        self.segments: List[Segment] = [
            Segment(angle_deg=float(a), p0_in=float(p), T0_in=float(t))
            for a, p, t in zip(segment_angles_deg, inlet_p0, inlet_T0)
        ]

        # Nominal operating point and solve target, set by
        # _compute_nominal_operating_point() (called from solve()).
        self._target_mass_flow_actual: Optional[float] = None
        self._nominal_corrected_speed: Optional[float] = None
        self._nominal_mass_flow_corrected: Optional[float] = None

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
    def _compute_nominal_operating_point(self) -> None:
        """
        Compute and cache the nominal operating point used both (a) as the
        target total mass flow for the coupled solve, and (b) as the
        reference (corrected_speed, mass_flow_corrected) reported on the
        whole-compressor aggregate result.

        Evaluated from the shared rotor speed and the requested overall
        ``beta``, at the mean (arithmetic average) inlet total pressure and
        temperature across segments -- i.e. the point the machine would sit
        at if the inlet were uniform. Sets ``self._nominal_corrected_speed``,
        ``self._nominal_mass_flow_corrected`` and
        ``self._target_mass_flow_actual``.
        """
        T0_nominal = float(np.mean([seg.T0_in for seg in self.segments]))
        p0_nominal = float(np.mean([seg.p0_in for seg in self.segments]))
        self._nominal_corrected_speed = self._corrected_speed(T0_nominal)
        self._nominal_mass_flow_corrected = self.map.get_mass_flow(
            self._nominal_corrected_speed, self.beta
        )
        self._target_mass_flow_actual = self._corrected_mass_flow_to_actual(
            self._nominal_mass_flow_corrected, p0_nominal, T0_nominal
        )

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
        self._compute_nominal_operating_point()

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
            pressure_ratio=pressure_ratio,
            surge_margin=surge_margin,
            efficiency=efficiency,
            beta=beta_local,
            corrected_speed=corrected_speed,
            mass_flow_corrected=mass_flow_corrected_full,
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
        - ``corrected_speed``, ``mass_flow_corrected`` : the nominal
          (mean-inlet-condition) operating point on the shared map that the
          solve was targeted at.
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

        p0_in_mean = float(np.mean([seg.p0_in for seg in self.segments]))
        pressure_ratio_overall = p0_overall / p0_in_mean

        assert self._nominal_corrected_speed is not None  # set by solve()
        assert self._nominal_mass_flow_corrected is not None  # set by solve()

        return SegmentResult(
            angle_deg=360.0,
            p0=p0_overall,
            p_static=p_static_overall,
            T0=T0_overall,
            T_static=T_static_overall,
            mass_flow=total_mass_flow,
            pressure_ratio=pressure_ratio_overall,
            surge_margin=surge_margin_overall,
            efficiency=efficiency_overall,
            beta=self.beta,
            corrected_speed=self._nominal_corrected_speed,
            mass_flow_corrected=self._nominal_mass_flow_corrected,
        )

    # -------------------------------------------------------------------
    # Circumferential inlet-distortion descriptors: DC(theta), DC60, DC90
    # -------------------------------------------------------------------
    #
    # DC(theta) is the classical circumferential distortion coefficient
    # (SAE ARP1420; also widely known from Reid, 1969, the same reference
    # underlying the parallel-compressor coupling above):
    #
    #     DC(theta) = (P0bar - P0_theta,low) / qbar
    #
    # where P0bar is the angle-weighted average inlet total pressure over
    # the full 360-degree annulus, P0_theta,low is the *lowest* angle-
    # weighted average inlet total pressure found over any contiguous arc
    # of extent ``theta`` degrees (the "worst" low-pressure sector), and
    # qbar is the average dynamic pressure at the compressor face. DC60 and
    # DC90 are DC(theta) evaluated at theta = 60 and 90 degrees.
    #
    # This is purely an *inlet* characterization -- it depends only on the
    # segment definitions (angle_deg, p0_in) and the total inlet mass flow,
    # not on how the compressor itself responds -- so it can be evaluated
    # before or after solve().

    def _segment_boundaries_deg(self) -> NDArray[np.float64]:
        """
        Cumulative angular boundaries of the segments, assuming they are
        arranged sequentially around the annulus in the order given
        (segment 0 starting at 0 deg). Returns an array of length n+1:
        ``boundaries[i]`` is the start angle of segment i, and
        ``boundaries[-1] == 360.0``.
        """
        return np.concatenate(
            ([0.0], np.cumsum([seg.angle_deg for seg in self.segments]))
        )

    def _angle_weighted_average(self, values: Sequence[float]) -> float:
        """Angle-weighted average of a per-segment quantity over the full annulus."""
        weighted_sum = sum(seg.angle_deg * v for seg, v in zip(self.segments, values))
        return weighted_sum / 360.0

    def _arc_average_p0(self, start_deg: float, extent_deg: float) -> float:
        """
        Angle-weighted average inlet total pressure over the contiguous arc
        ``[start_deg, start_deg + extent_deg)`` of the annulus, wrapping
        past 360 degrees as needed, given the piecewise-constant per-segment
        ``p0_in`` profile implied by ``self.segments`` (see
        ``_segment_boundaries_deg``).
        """
        boundaries = self._segment_boundaries_deg()
        start = start_deg % 360.0
        end = start + extent_deg

        # Split the (possibly wrapping) arc into up to two non-wrapping
        # sub-arcs so overlap with each segment can be computed as ordinary
        # 1-D interval overlap.
        sub_arcs = [(start, min(end, 360.0))]
        if end > 360.0:
            sub_arcs.append((0.0, end - 360.0))

        weighted_sum = 0.0
        for arc_start, arc_end in sub_arcs:
            for i, seg in enumerate(self.segments):
                seg_start, seg_end = boundaries[i], boundaries[i + 1]
                overlap = min(arc_end, seg_end) - max(arc_start, seg_start)
                if overlap > 0.0:
                    weighted_sum += overlap * seg.p0_in

        return weighted_sum / extent_deg

    def _worst_sector_average_p0(self, extent_deg: float) -> float:
        """
        Minimum angle-weighted average inlet total pressure over any
        contiguous arc of extent ``extent_deg`` around the annulus (the
        "P0_theta,low" term of DC(theta)).

        Since the inlet total pressure profile is piecewise-constant, the
        sliding-window average is a piecewise-linear function of the
        window's start angle, so its minimum occurs at one of the window's
        "critical" start angles -- where either the leading or trailing
        edge coincides with a segment boundary. Only those candidates need
        to be evaluated.
        """
        boundaries = self._segment_boundaries_deg()[:-1]  # segment start angles
        candidate_starts = np.concatenate([boundaries, (boundaries - extent_deg) % 360.0])
        return min(self._arc_average_p0(float(s), extent_deg) for s in candidate_starts)

    def _average_dynamic_pressure(self, p0_avg: float, T0_avg: float, mass_flow: float) -> float:
        """
        Average dynamic pressure at the compressor face inlet:
        qbar = P0bar - Psbar, where Psbar is the static pressure
        corresponding to the total inlet mass flow passing uniformly
        through ``self.inlet_area`` at the average inlet total conditions
        (via the same compressible-flow Mach-number solve used elsewhere in
        the model).
        """
        p_static, _ = self._static_from_total(p0_avg, T0_avg, mass_flow, self.inlet_area)
        return p0_avg - p_static

    def distortion_coefficient(self, extent_deg: float) -> float:
        """
        Circumferential distortion coefficient DC(theta) for an arc extent
        of ``extent_deg`` degrees:

            DC(theta) = (P0bar - P0_theta,low) / qbar

        Uses the total inlet mass flow (``_target_mass_flow_actual``),
        computing it via ``_compute_nominal_operating_point`` first if
        ``solve()`` has not already been called.
        """
        if self._target_mass_flow_actual is None:
            self._compute_nominal_operating_point()
        assert self._target_mass_flow_actual is not None

        p0_avg = self._angle_weighted_average([seg.p0_in for seg in self.segments])
        T0_avg = self._angle_weighted_average([seg.T0_in for seg in self.segments])
        p0_low = self._worst_sector_average_p0(extent_deg)
        q_avg = self._average_dynamic_pressure(p0_avg, T0_avg, self._target_mass_flow_actual)

        return (p0_avg - p0_low) / q_avg

    def dc60(self) -> float:
        """DC(60): circumferential distortion coefficient for a 60-degree sector."""
        return self.distortion_coefficient(60.0)

    def dc90(self) -> float:
        """DC(90): circumferential distortion coefficient for a 90-degree sector."""
        return self.distortion_coefficient(90.0)
