"""
Multiple-Segment Parallel Compressor Model
===========================================

Structural skeleton only. This package defines the classes, methods,
attributes, and full input/output contracts for a multiple-segment parallel
compressor model. Numerical implementations are left as stubs (``...`` /
``TODO`` comments) and will be filled in in a later step.

Theory
------
Classical "parallel compressor" theory (Reid, C., *The Response of Axial Flow
Compressors to Intermittent and Periodic Distortions*, NASA TN D-5401, 1969):
the compressor annulus is divided into a number of angular segments. Each
segment is treated as an independent compressor operating on the *same*
overall (undistorted) compressor map, but each may see a different inlet
total pressure and/or total temperature. The segments are coupled by two
physical constraints:

    1. All segments discharge into a common downstream plenum/duct, so their
       exit *static* pressure must be equal.
    2. All segments share the same rotor (same mechanical/corrected speed
       relationship), and together must be consistent with the overall
       operating point (beta) requested for the machine.

Given these constraints, each segment settles onto a different point on the
shared compressor map (i.e. a different local "beta"), and the aggregate
(mass-flow-weighted) performance differs from the undistorted case. This is
the mechanism by which parallel compressor theory predicts loss of
surge margin and efficiency under inlet distortion.

Design Notes
------------
- The compressor map is parameterized by (corrected speed, beta), where beta
  is an auxiliary coordinate along each speed line (beta = 0 at surge,
  beta = 1 at choke). This avoids the multi-valued mass-flow-vs-pressure-ratio
  relationship that a plain (speed, mass flow) parameterization would have
  near surge.
- Interpolation over the map is done with
  ``scipy.interpolate.RegularGridInterpolator``.
- The coupled per-segment system (equal exit static pressure, consistent
  mass flow) is intended to be solved with ``scipy.optimize.least_squares``.
- Conversion between total and static conditions (given mass flow and flow
  area) requires solving the compressible-flow continuity relation for Mach
  number; intended to be done with ``scipy.optimize.brentq``.

Package layout
--------------
- ``gas_properties`` : ``GasProperties``, ``ReferenceConditions``.
- ``segments``        : ``Segment``, ``SegmentResult``, ``ParallelCompressorResult``.
- ``compressor_map``   : ``CompressorMap``.
- ``map_io``             : file readers that build a ``CompressorMap`` from
                           vendor/legacy map file formats (e.g. ``.chC``
                           constant-speedline files) via ``read_hecc_chc``.
- ``model``             : ``ParallelCompressorModel`` (the solver).
- ``__main__``           : example-usage entry point
                           (``python -m parallel_compressor_model``).
"""

from .compressor_map import CompressorMap
from .gas_properties import GasProperties, ReferenceConditions
from .map_io import read_hecc_chc
from .model import ParallelCompressorModel
from .segment import ParallelCompressorResult, Segment, SegmentResult

__all__ = [
    "GasProperties",
    "ReferenceConditions",
    "Segment",
    "SegmentResult",
    "ParallelCompressorResult",
    "CompressorMap",
    "read_hecc_chc",
    "ParallelCompressorModel",
]
