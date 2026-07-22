"""Generate point-based representative transform report for Rubik's Cube myperms."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.myperm_effects import MypermEffectAnalyzer
from core.myperm_keys import format_myperm_key, make_myperm_key, normalize_myperm_registry
from core.myperm_points import MypermPointCalculator, load_myperm_points
from cube.rubiks_cube import Rubiks_3


DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "myperm_point_representatives.csv"


def collect_rows(cube, point_table, name_prefixes = (), quiet = False):
    analyzer = MypermEffectAnalyzer(cube)
    calculator = MypermPointCalculator(cube, point_table)
    legacy_names_by_current = _legacy_names_by_current(cube)
    current_names = [
        name
        for name in sorted(cube.myperms2)
        if _matches_prefix(name, legacy_names_by_current.get(name, ()), name_prefixes)
    ]

    rows = []
    for row_index, current_name in enumerate(current_names, start = 1):
        if not quiet and (row_index == 1 or row_index % 100 == 0 or row_index == len(current_names)):
            print(f"scoring {row_index}/{len(current_names)}: {current_name}", flush = True)

        current_key = make_myperm_key(current_name, 0)
        current_point = calculator.point_for_key(current_key)
        best_row = _best_transform_row(cube, calculator, current_name)
        best_key = make_myperm_key(current_name, best_row["best_transform_index"])
        best_effect = analyzer.analyze(best_key)
        best_moves = tuple(cube.myperms[best_key])

        rows.append({
            "puzzle":f"rubiks-{cube.size}",
            "current_name":current_name,
            "legacy_names":";".join(sorted(legacy_names_by_current.get(current_name, ()))),
            "current_key":format_myperm_key(current_key),
            "current_point":current_point,
            "best_transform_index":best_row["best_transform_index"],
            "best_point":best_row["best_point"],
            "point_delta":best_row["best_point"] - current_point,
            "best_existing_key":format_myperm_key(best_key),
            "best_effect_name":best_effect.concise_name(),
            "proposed_source_name":"",
            "moved_count":best_effect.moved_count,
            "orientation_count":best_effect.orientation_count,
            "move_count":len(best_moves),
            "moves":" ".join(str(move).strip() for move in _display_moves(cube, best_moves)),
        })

    _add_proposed_source_names(rows)
    return rows


def _legacy_names_by_current(cube):
    legacy_names_by_current = defaultdict(list)
    for legacy_name, current_name in getattr(cube, "myperm_name_aliases", {}).items():
        if legacy_name != current_name:
            legacy_names_by_current[current_name].append(legacy_name)
    return legacy_names_by_current


def _matches_prefix(current_name, legacy_names, prefixes):
    if not prefixes:
        return True
    names = (current_name,) + tuple(legacy_names)
    return any(name.startswith(prefix) for prefix in prefixes for name in names)


def _best_transform_row(cube, calculator, current_name):
    best_point = None
    best_transform_index = None
    for key in cube.myperms:
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == current_name):
            continue
        point = calculator.point_for_key(key)
        if best_point is None or point > best_point or (
            point == best_point and key[1] < best_transform_index
        ):
            best_point = point
            best_transform_index = key[1]
    if best_transform_index is None:
        raise ValueError(f"no expanded myperms for {current_name!r}")
    return {
        "best_transform_index":best_transform_index,
        "best_point":best_point,
    }


def _display_moves(cube, moves):
    if hasattr(cube, "format_moves"):
        return tuple(cube.format_moves(moves))
    return moves


def _add_proposed_source_names(rows):
    grouped = defaultdict(list)
    for row in rows:
        if _keeps_source_myperm_name(row["current_name"]):
            row["proposed_source_name"] = row["current_name"]
            continue
        grouped[row["best_effect_name"]].append(row)

    for effect_name, matching_rows in grouped.items():
        if len(matching_rows) == 1:
            matching_rows[0]["proposed_source_name"] = effect_name
            continue
        for variant_index, row in enumerate(
            sorted(matching_rows, key = lambda item:(item["current_name"], item["legacy_names"])),
            start = 1,
        ):
            row["proposed_source_name"] = f"{effect_name}~v{variant_index:02d}"


def _keeps_source_myperm_name(name):
    return str(name).startswith(("OuterCenterBar", "MidCenterBar"))


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents = True, exist_ok = True)
    fieldnames = (
        "puzzle",
        "current_name",
        "legacy_names",
        "current_key",
        "current_point",
        "best_transform_index",
        "best_point",
        "point_delta",
        "best_existing_key",
        "best_effect_name",
        "proposed_source_name",
        "moved_count",
        "orientation_count",
        "move_count",
        "moves",
    )
    with output_path.open("w", encoding = "utf-8", newline = "") as output_file:
        writer = csv.DictWriter(output_file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, output_path):
    puzzle_counts = Counter(row["puzzle"] for row in rows)
    point_gain_rows = sum(row["point_delta"] > 0 for row in rows)
    source_name_change_rows = sum(row["current_name"] != row["proposed_source_name"] for row in rows)
    transform_only_point_gain_rows = sum(
        row["point_delta"] > 0 and row["current_name"] == row["proposed_source_name"]
        for row in rows
    )
    print(f"wrote {len(rows)} point representative rows to {output_path}")
    print("puzzles:", ", ".join(f"{name}={count}" for name, count in sorted(puzzle_counts.items())))
    print(f"rows with point gain: {point_gain_rows}")
    print(f"rows with proposed source name change: {source_name_change_rows}")
    print(f"rows with transform-only point gain: {transform_only_point_gain_rows}")


def build_parser():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--rubiks-size", type = int, default = 7, choices = range(2, 8))
    parser.add_argument("--points", type = Path, default = PROJECT_ROOT / "Points.txt")
    parser.add_argument("--output", type = Path, default = DEFAULT_OUTPUT)
    parser.add_argument(
        "--name-prefix",
        action = "append",
        default = [],
        help = "Limit rows to current or legacy names with this prefix. Can be passed multiple times.",
    )
    parser.add_argument("--quiet", action = "store_true")
    parser.add_argument(
        "--full-init",
        action = "store_true",
        help = "Use the normal Rubiks_3 initialization path instead of expanding only requested source names.",
    )
    return parser


def build_report_cube(size, name_prefixes = (), full_init = False):
    if full_init or not name_prefixes:
        return Rubiks_3(size = size)

    cube = Rubiks_3(size = size, RegisterMyperms = False)
    cube._register_myperms2()
    cube.myperms2 = normalize_myperm_registry(cube.myperms2)
    legacy_names_by_current = _legacy_names_by_current(cube)
    selected_names = [
        name
        for name in sorted(cube.myperms2)
        if _matches_prefix(name, legacy_names_by_current.get(name, ()), name_prefixes)
    ]
    cube._expand_registered_myperms(names = selected_names)
    return cube


def main():
    args = build_parser().parse_args()
    name_prefixes = tuple(args.name_prefix)
    cube = build_report_cube(args.rubiks_size, name_prefixes = name_prefixes, full_init = args.full_init)
    point_table = load_myperm_points(args.points)
    rows = collect_rows(cube, point_table, name_prefixes = name_prefixes, quiet = args.quiet)
    write_csv(rows, args.output)
    print_summary(rows, args.output)


if __name__ == "__main__":
    main()
