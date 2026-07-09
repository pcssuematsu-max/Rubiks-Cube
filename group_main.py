"""Launch the Rubiks AI UI with a small finite-group puzzle."""

import argparse

from ui.frame import Frame
from ui.frame_config import FrameConfig


def build_group_frame_config(kind="symmetric", degree=3, dimension=2, modulus=2,
                             generators=None, family="GL", name=None,
                             auto_add_inverses=False, ai_search_modes=("search2",),
                             **frame_options):
    """Return a compact one-AI configuration for a finite group puzzle."""
    normalized_kind = str(kind).lower()
    cube_size = degree if normalized_kind == "symmetric" else dimension
    options = dict(
        puzzle_type="group",
        cube_size=cube_size,
        group_kind=normalized_kind,
        group_degree=degree,
        group_dimension=dimension,
        group_modulus=modulus,
        group_generators=generators,
        group_family=family,
        group_name=name,
        group_auto_add_inverses=auto_add_inverses,
        ai_search_modes=tuple(ai_search_modes),
        search3_progress=(False,) * len(ai_search_modes),
    )
    options.update(frame_options)
    return FrameConfig(**options)


def parse_args():
    parser = argparse.ArgumentParser(description="Finite group puzzle using the Rubiks AI")
    parser.add_argument("--kind", choices=("symmetric", "linear"), default="symmetric")
    parser.add_argument("--degree", type=int, default=3, help="n for a subgroup of S_n")
    parser.add_argument("--dimension", type=int, default=2, help="d for a subgroup of GL(d, F_p)")
    parser.add_argument("--modulus", type=int, default=2, help="prime p for F_p")
    parser.add_argument("--family", choices=("GL", "SL"), default="GL")
    parser.add_argument("--auto-add-inverses", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = build_group_frame_config(
        kind=args.kind,
        degree=args.degree,
        dimension=args.dimension,
        modulus=args.modulus,
        family=args.family,
        auto_add_inverses=args.auto_add_inverses,
    )
    frame = Frame(config=config)
    frame.pack()
    frame.mainloop()
