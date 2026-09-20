"""
Example usage / CLI entry point: reads a JSON input file describing the
compressor map file, the parallel-compressor model inputs, and the desired
output plot filename; solves the coupled system; prints per-segment and
overall performance; and saves the annotated compressor-map plot.

Run with:
    python __main__.py [input_file.json]

If no input file is given, ``input.json`` next to this script is used.

Input file format
------------------
A JSON object with the following keys:

    compressor_map_file    : path to the .chC compressor map file. Relative
                              paths are resolved against the input file's
                              own directory.
    rotor_speed_rpm         : mechanical rotor speed [1/min].
    beta                     : overall beta operating-point parameter.
    segments                 : object with three parallel arrays, all the
                              same length (one entry per angular segment):
                                  "angles_deg" : segment angular extents
                                                 [deg] (must sum to 360).
                                  "inlet_p0"    : segment inlet total
                                                 pressures [Pa].
                                  "inlet_T0"    : segment inlet total
                                                 temperatures [K].
    exit_area (optional)      : discharge flow area [m^2]. Defaults to 1.0.
    gas_properties (optional)          : object overriding GasProperties
                              defaults ("cp", "gamma", "R").
    reference_conditions (optional)    : object overriding
                              ReferenceConditions defaults ("p_ref", "T_ref").
    output_map_file          : filename the annotated compressor-map plot is
                              saved to. Relative paths are resolved against
                              the input file's own directory.

See ``input.json`` in this directory for a worked example.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt

from gas_properties import GasProperties, ReferenceConditions
from map_io import read_hecc_chc
from model import ParallelCompressorModel
from plotting import plot_parallel_compressor_result

DEFAULT_INPUT_FILE = Path(__file__).with_name("input.json")


def _load_input(input_file: Path) -> Dict[str, Any]:
    """Read and parse the JSON input file."""
    with input_file.open("r") as f:
        return json.load(f)


def _build_model(config: Dict[str, Any], base_dir: Path) -> ParallelCompressorModel:
    """
    Construct a ``ParallelCompressorModel`` (and its ``CompressorMap``) from
    a parsed input-file dict. See the module docstring for the expected
    keys. ``base_dir`` is the directory relative paths are resolved against
    (the input file's own directory).
    """
    map_path = Path(config["compressor_map_file"])
    if not map_path.is_absolute():
        map_path = base_dir / map_path
    compressor_map = read_hecc_chc(map_path)

    segments_cfg = config["segments"]

    gas = GasProperties(**config.get("gas_properties", {}))
    reference = ReferenceConditions(**config.get("reference_conditions", {}))

    return ParallelCompressorModel(
        compressor_map=compressor_map,
        rotor_speed_rpm=float(config["rotor_speed_rpm"]),
        beta=float(config["beta"]),
        segment_angles_deg=segments_cfg["angles_deg"],
        inlet_p0=segments_cfg["inlet_p0"],
        inlet_T0=segments_cfg["inlet_T0"],
        gas=gas,
        reference=reference,
        exit_area=float(config.get("exit_area", 1.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Solve a multiple-segment parallel compressor model from a JSON "
            "input file and plot the result on the compressor map."
        )
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Path to the JSON input file (default: {DEFAULT_INPUT_FILE.name})",
    )
    args = parser.parse_args()

    input_file: Path = args.input_file
    config = _load_input(input_file)
    model = _build_model(config, base_dir=input_file.resolve().parent)
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

    output_map_path = Path(config["output_map_file"])
    if not output_map_path.is_absolute():
        output_map_path = input_file.resolve().parent / output_map_path

    plot_parallel_compressor_result(result, compressor_map=model.map)
    plt.tight_layout()
    plt.savefig(output_map_path, dpi=130)
    print(f"\nSaved plot to {output_map_path}")


if __name__ == "__main__":
    main()
