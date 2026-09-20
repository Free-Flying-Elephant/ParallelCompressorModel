"""
Input and output data structures for individual angular inlet segments and
for the whole-compressor aggregate result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Segment:
    """
    Definition of a single angular inlet segment (model input).

    Attributes
    ----------
    angle_deg : angular extent of this segment [deg].
    p0_in     : inlet total pressure for this segment [Pa].
    T0_in     : inlet total temperature for this segment [K].
    """

    angle_deg: float
    p0_in: float
    T0_in: float


@dataclass
class SegmentResult:
    """
    Computed performance quantities for a single segment, or for the
    whole-compressor aggregate (model output).

    Attributes
    ----------
    angle_deg            : angular extent represented by this result [deg]
                            (360.0 for the whole-compressor aggregate).
    p0                    : total pressure at compressor discharge [Pa].
    p_static              : static pressure at compressor discharge [Pa]
                            (equal across all segments by construction).
    T0                    : total temperature at compressor discharge [K].
    T_static              : static temperature at compressor discharge [K].
    mass_flow             : (actual, non-corrected) mass flow [kg/s].
    pressure_ratio         : total-to-total pressure ratio, p0 / inlet p0 [-].
    surge_margin           : surge margin at the segment's operating point [-].
    efficiency             : total-to-total isentropic efficiency [-].
    beta                   : local beta coordinate on the compressor map [-].
    corrected_speed         : corrected rotor speed the map lookup was
                              evaluated at [1/min].
    mass_flow_corrected      : full-annulus-equivalent corrected mass flow
                              read from the shared compressor map at
                              (corrected_speed, beta) [kg/s] -- directly
                              comparable to the map's own mass-flow axis,
                              e.g. for plotting the operating point on the map.
    """

    angle_deg: float
    p0: float
    p_static: float
    T0: float
    T_static: float
    mass_flow: float
    pressure_ratio: float
    surge_margin: float
    efficiency: float
    beta: float
    corrected_speed: float
    mass_flow_corrected: float


@dataclass
class ParallelCompressorResult:
    """
    Full result of a parallel-compressor solve.

    Attributes
    ----------
    segments : per-segment results, in the same order as the input segment
               arrays.
    overall  : mass-flow-weighted whole-compressor aggregate result.
    """

    segments: List[SegmentResult]
    overall: SegmentResult
