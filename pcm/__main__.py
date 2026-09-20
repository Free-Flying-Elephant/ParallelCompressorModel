"""
Example usage: reads the ``hecc.chC`` compressor map, applies a total-
pressure distortion pattern across four 90-degree segments, solves the
coupled parallel-compressor system, and prints per-segment and overall
performance.

Run with ``python -m parallel_compressor_model`` from a directory containing
``hecc.chC`` (or edit ``MAP_FILE`` below to point at your own map file).
"""

from pathlib import Path

from map_io import read_hecc_chc
from model import ParallelCompressorModel

MAP_FILE = Path(__file__).with_name("hecc.chC")
AE = 3.14159265 / 4 * (312.5 ** 2 - 301.5 ** 2) * 1e-6

def main() -> None:
    compressor_map = read_hecc_chc(MAP_FILE)

    # One low-pressure sector (e.g. representing an inlet distortion screen
    # or crosswind ingestion) among three sectors at ambient pressure.
    model = ParallelCompressorModel(
        compressor_map=compressor_map,
        rotor_speed_rpm=20670.0,
        beta=0.6,
        segment_angles_deg=[90.0, 90.0, 90.0, 90.0],
        inlet_p0=[101_325.0, 90_000.0, 101_325.0, 101_325.0],
        inlet_T0=[288.15, 288.15, 288.15, 288.15],
        exit_area=AE,
    )
    result = model.solve()

    header = (
        f"{'angle[deg]':>10} {'beta':>8} {'p0[Pa]':>10} {'p_s[Pa]':>10} "
        f"{'T0[K]':>8} {'T_s[K]':>8} {'mdot[kg/s]':>11} {'SM[%]':>7} {'eff':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in result.segments:
        print(
            f"{r.angle_deg:10.1f} {r.beta:8.4f} {r.p0:10.1f} {r.p_static:10.1f} "
            f"{r.T0:8.2f} {r.T_static:8.2f} {r.mass_flow:11.4f} "
            f"{r.surge_margin:7.2f} {r.efficiency:6.4f}"
        )
    print("-" * len(header))
    o = result.overall
    print(
        f"{'OVERALL':>10} {o.beta:8.4f} {o.p0:10.1f} {o.p_static:10.1f} "
        f"{o.T0:8.2f} {o.T_static:8.2f} {o.mass_flow:11.4f} "
        f"{o.surge_margin:7.2f} {o.efficiency:6.4f}"
    )


if __name__ == "__main__":
    main()
