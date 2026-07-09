"""Generate old-to-effect-based myperm name proposals as CSV."""

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
from core.myperm_keys import format_myperm_key, make_myperm_key
from cto.cube import CtoCube
from cube.rubiks_cube import Rubiks_3
from fto.cube import FtoCube
from megaminx.cube import MegaminxCube
from pyraminx.cube import MasterPyraminxCube, PyraminxCube
from skewb.cube import SkewbCube


DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "myperm_name_proposals.csv"


def build_puzzles(rubiks_size):
    return (
        (f"rubiks-{rubiks_size}", Rubiks_3(size = rubiks_size)),
        ("megaminx", MegaminxCube()),
        ("pyraminx", PyraminxCube()),
        ("master-pyraminx", MasterPyraminxCube()),
        ("skewb", SkewbCube()),
        ("fto", FtoCube()),
        ("cto", CtoCube()),
    )


def collect_rows(puzzles):
    rows = []
    for puzzle_name, puzzle in puzzles:
        analyzer = MypermEffectAnalyzer(puzzle)
        aliases = getattr(puzzle, "myperm_name_aliases", {})
        name_pairs = aliases.items() if aliases else ((name, name) for name in puzzle.myperms2)
        for old_name, current_name in sorted(name_pairs):
            key = make_myperm_key(current_name, 0)
            effect = analyzer.analyze(key)
            effect_name = effect.concise_name()
            moves = tuple(puzzle.myperms[key])
            if hasattr(puzzle, "format_moves"):
                display_moves = tuple(puzzle.format_moves(moves))
            else:
                display_moves = moves
            rows.append({
                "puzzle":puzzle_name,
                "old_name":old_name,
                "old_key":format_myperm_key(make_myperm_key(old_name, 0)),
                "current_name":current_name,
                "proposed_name":current_name,
                "effect_name":effect_name,
                "moved_count":effect.moved_count,
                "orientation_count":effect.orientation_count,
                "move_count":len(moves),
                "moves":" ".join(str(move).strip() for move in display_moves),
            })

    _add_collision_variants(rows)
    return rows


def _add_collision_variants(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["puzzle"], row["effect_name"])].append(row["current_name"])
    collision_sizes = {
        key:len(set(current_names))
        for key, current_names in grouped.items()
    }
    for row in rows:
        row["collision_size"] = collision_sizes[(row["puzzle"], row["effect_name"])]


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents = True, exist_ok = True)
    fieldnames = (
        "puzzle",
        "old_name",
        "old_key",
        "current_name",
        "proposed_name",
        "effect_name",
        "moved_count",
        "orientation_count",
        "move_count",
        "collision_size",
        "moves",
    )
    with output_path.open("w", encoding = "utf-8", newline = "") as output_file:
        writer = csv.DictWriter(output_file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, output_path):
    puzzle_counts = Counter(row["puzzle"] for row in rows)
    collision_rows = sum(row["collision_size"] > 1 for row in rows)
    orientation_rows = sum(row["orientation_count"] > 0 for row in rows)
    print(f"wrote {len(rows)} proposals to {output_path}")
    print("puzzles:", ", ".join(f"{name}={count}" for name, count in sorted(puzzle_counts.items())))
    print(f"rows with orientation changes: {orientation_rows}")
    print(f"rows requiring a collision variant: {collision_rows}")


def build_parser():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--rubiks-size", type = int, default = 7, choices = range(2, 8))
    parser.add_argument("--output", type = Path, default = DEFAULT_OUTPUT)
    return parser


def main():
    args = build_parser().parse_args()
    rows = collect_rows(build_puzzles(args.rubiks_size))
    write_csv(rows, args.output)
    print_summary(rows, args.output)


if __name__ == "__main__":
    main()
