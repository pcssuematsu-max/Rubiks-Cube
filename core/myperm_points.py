"""Point scoring for choosing representative myperm transforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.myperm_effects import MypermEffectAnalyzer
from core.myperm_keys import make_myperm_key


SECTION_ALIASES = {
    "Corners":"C",
    "Corner":"C",
    "Edges":"E",
    "Edge":"E",
    "OuterEdge":"OE",
    "OuterEdges":"OE",
    "OE":"OE",
    "MidEdge":"ME",
    "Wing":"W",
    "XCenter":"CtrX",
    "PlusCenter":"CtrPlus",
    "ObliqueCenter":"CtrObl",
    "CoreCenter":"CtrCore",
}


@dataclass(frozen = True)
class MypermPointTable:
    """Point values loaded from Points.txt."""

    points_by_part: dict
    default_by_part: dict

    def point_for_part(self, part_code, position):
        """Return the configured score for one source position."""
        canonical_part = self._canonical_part_code(part_code)
        canonical_position = self._canonical_position(canonical_part, position)
        if canonical_part == "EAll":
            return self._mid_edge_point(canonical_position)
        if canonical_part == "C":
            return self._corner_point(canonical_position)
        if canonical_part == "E":
            return self._edge_point(canonical_position)
        if canonical_part == "OE":
            return self._outer_edge_point(canonical_position)
        if canonical_part == "ME":
            return self._mid_edge_point(canonical_position)
        if canonical_part == "W":
            return self._wing_point(canonical_position)
        if str(canonical_part).startswith("Ctr"):
            return self._center_point(canonical_part, canonical_position)
        return self._lookup(canonical_part, canonical_position)

    def edge_bundle_point(self, cube, source_position):
        """Return the score of a collapsed MidEdge+Wing edge bundle."""
        base_position = self._strip_edge_axis(source_position)
        point = self._mid_edge_point(base_position)
        if cube is None or not hasattr(cube, "edge_index"):
            return point

        for piece in cube.edge_index:
            try:
                axis = cube._edge_axis_label(piece)
            except AttributeError:
                continue
            if axis in {"M", "E", "S"}:
                continue
            faces = "".join(cube._move_face_label(index) for index in piece)
            if faces not in {base_position, base_position[::-1]}:
                continue
            wing_position = self._canonical_wing_position(f"{faces}@{axis}")
            point += self._wing_point(wing_position)
        return point

    def _lookup(self, part_code, position):
        section = self.points_by_part.get(part_code, {})
        if position in section:
            return section[position]
        normalized_position = self._normalize_position(position)
        if normalized_position in section:
            return section[normalized_position]
        return self.default_by_part.get(part_code, 0)

    def _corner_point(self, position):
        section = self.points_by_part.get("C", {})
        if position in section:
            return section[position]
        faces = frozenset(self._position_faces(position))
        for configured_position, point in section.items():
            if frozenset(self._position_faces(configured_position)) == faces:
                return point
        return self.default_by_part.get("C", 0)

    def _edge_point(self, position):
        return self._lookup_first("E", self._edge_position_aliases(position))

    def _outer_edge_point(self, position):
        return self._lookup_first("OE", self._outer_edge_position_aliases(position))

    def _outer_edge_position_aliases(self, position):
        position = str(position)
        if "@" not in position:
            return self._edge_position_aliases(position)
        edge, suffix = position.split("@", 1)
        faces = self._position_faces(edge)
        if len(faces) != 2:
            return (position,)
        return (
            f"{self._join_faces(faces)}@{suffix}",
            f"{self._join_faces(tuple(reversed(faces)))}@{suffix}",
        )

    def _mid_edge_point(self, position):
        position = self._strip_edge_axis(position)
        return self._lookup_first("ME", (position, position[::-1]))

    def _wing_point(self, position):
        position = self._canonical_wing_position(position)
        if "@" not in position:
            return self._lookup("W", position)
        edge, axis = position.split("@", 1)
        return self._lookup_first("W", (position, f"{edge[::-1]}@{axis}"))

    def _center_point(self, part_code, position):
        return self._lookup_first(part_code, self._center_position_aliases(position))

    def _lookup_first(self, part_code, positions):
        for position in positions:
            section = self.points_by_part.get(part_code, {})
            if position in section:
                return section[position]
            normalized_position = self._normalize_position(position)
            if normalized_position in section:
                return section[normalized_position]
        return self.default_by_part.get(part_code, 0)

    def _edge_position_aliases(self, position):
        faces = self._position_faces(self._strip_edge_axis(position))
        if len(faces) != 2:
            return (position,)
        return (
            self._join_faces(faces),
            self._join_faces(tuple(reversed(faces))),
        )

    def _center_position_aliases(self, position):
        position = str(position)
        aliases = [position]
        if "@" not in position:
            return tuple(aliases)
        face, coordinates = position.split("@", 1)
        parts = coordinates.split(".")
        if len(parts) == 2:
            aliases.append(f"{face}@{parts[1]}.{parts[0]}")
        return tuple(dict.fromkeys(aliases))

    def _canonical_part_code(self, part_code):
        if part_code == "E":
            if "E" in self.points_by_part:
                return "E"
            return "ME"
        if part_code == "EAll":
            return "EAll"
        if str(part_code).startswith("W"):
            return "W"
        return part_code

    def _canonical_position(self, part_code, position):
        if part_code in {"ME", "EAll"}:
            return self._strip_edge_axis(position)
        if part_code == "W":
            return self._canonical_wing_position(position)
        return position

    def _strip_edge_axis(self, position):
        return str(position).split("@", 1)[0]

    def _canonical_wing_position(self, position):
        position = str(position)
        if "@" not in position:
            return position
        edge, axis = position.split("@", 1)
        axis = axis.lstrip("0123456789")
        return f"{edge}@{axis}"

    def _normalize_position(self, position):
        return self._join_faces(self._position_faces(position))

    def _position_faces(self, position):
        position = str(position)
        if "@" in position:
            position = position.split("@", 1)[0]
        if position.startswith("(") and position.endswith(")"):
            return tuple(part.strip() for part in position[1:-1].split(",") if part.strip())
        if "." in position:
            return tuple(part for part in position.split(".") if part)
        return tuple(position)

    def _join_faces(self, faces):
        faces = tuple(faces)
        if any(len(face) > 1 for face in faces):
            return ".".join(faces)
        return "".join(faces)


class MypermPointCalculator:
    """Calculate myperm points from sticker-effect analysis."""

    def __init__(self, cube, point_table):
        self.cube = cube
        self.point_table = point_table
        self.analyzer = MypermEffectAnalyzer(cube)
        self._group_rows = self._build_group_rows()

    def point_for_key(self, key_or_moves, include_internal_centers = False):
        """Analyze one myperm key or move sequence and return its point total."""
        _, moves = self.analyzer._resolve_moves(key_or_moves)
        return self.point_for_moves(moves, include_internal_centers = include_internal_centers)

    def point_for_moves(self, moves, include_internal_centers = False):
        """Return the point total for a move sequence without building effect names."""
        sticker_state = np.arange(len(self.cube.state_0), dtype = int)
        for move in moves:
            sticker_state = sticker_state[np.asarray(self.cube.move[move], dtype = int)]

        total = 0
        for group_row in self._group_rows:
            group = group_row["group"]
            pieces = group_row["pieces"]
            source_lookup = group_row["source_lookup"]
            positions = group_row["positions"]
            part_codes = group_row["part_codes"]
            piece_size = group_row["piece_size"]

            for destination_index, destination_piece in enumerate(pieces):
                actual_stickers = tuple(int(sticker_state[index]) for index in destination_piece)
                source_index = source_lookup.get(frozenset(actual_stickers))
                if source_index is None:
                    raise ValueError(
                        f"{group} destination {destination_piece} does not contain one registered physical piece"
                    )
                source_piece = pieces[source_index]
                if source_index == destination_index and actual_stickers == source_piece:
                    continue

                source_position = positions[source_index]
                destination_position = positions[destination_index]
                part_code = part_codes[source_index]
                if (
                    not include_internal_centers
                    and group == "Center"
                    and piece_size == 1
                    and str(part_code).startswith("Ctr")
                    and source_position.split("@", 1)[0] == destination_position.split("@", 1)[0]
                ):
                    continue
                total += self.point_table.point_for_part(part_code, source_position)
        return total

    def point_for_effect(self, effect, include_internal_centers = False):
        """Return the point total for an already analyzed effect."""
        total = 0
        for component in effect.components:
            if not include_internal_centers and component.is_internal_center_permutation():
                continue
            for transfer in component.transfers:
                if component.part_code == "EAll":
                    total += self.point_table.edge_bundle_point(self.cube, transfer.source)
                else:
                    total += self.point_table.point_for_part(component.part_code, transfer.source)
        return total

    def _build_group_rows(self):
        rows = []
        for group, pieces in self.analyzer._groups.items():
            rows.append(
                {
                    "group":group,
                    "pieces":pieces,
                    "piece_size":len(pieces[0]),
                    "source_lookup":{frozenset(piece):index for index, piece in enumerate(pieces)},
                    "positions":tuple(
                        self.analyzer._position_name(group, piece)
                        for piece in pieces
                    ),
                    "part_codes":tuple(
                        self.analyzer._part_code(group, piece)
                        for piece in pieces
                    ),
                }
            )
        return tuple(rows)


@dataclass(frozen = True)
class MypermPointTransform:
    """Highest-point transform choice for one move sequence."""

    moves: tuple
    transform_index: int
    point: float


def point_representative_transform(cube, moves, point_table = None, include_internal_centers = False):
    """Return the transform of moves with the highest myperms_point score."""
    if point_table is None:
        point_table = load_myperm_points(
            Path(__file__).resolve().parent.parent / "Points.txt",
            puzzle = getattr(cube, "myperm_point_puzzle", None),
        )

    calculator = MypermPointCalculator(cube, point_table)
    transform_count = len(getattr(cube, "transformation_keys", ()))
    if transform_count == 0 or not hasattr(cube, "transform"):
        return MypermPointTransform(
            tuple(moves),
            0,
            calculator.point_for_moves(moves, include_internal_centers = include_internal_centers),
        )

    best = None
    for transform_index in range(transform_count):
        transformed_moves = tuple(cube.transform(moves, transform_index))
        point = calculator.point_for_moves(
            transformed_moves,
            include_internal_centers = include_internal_centers,
        )
        row = MypermPointTransform(transformed_moves, transform_index, point)
        if best is None or row.point > best.point or (
            row.point == best.point and row.transform_index < best.transform_index
        ):
            best = row

    return best


def reindex_myperms_by_points(cube, point_table, names = None):
    """Reassign each registered myperm's transform #00 to its highest-point transform."""
    calculator = MypermPointCalculator(cube, point_table)
    names = tuple(names) if names is not None else tuple(getattr(cube, "myperms2", ()))
    reindex_by_name = {}
    point_rows_by_name = {}

    for name in names:
        transform_indices = sorted(
            key[1]
            for key in cube.myperms
            if isinstance(key, tuple) and len(key) == 2 and key[0] == name
        )
        if not transform_indices:
            continue

        scored_rows = []
        for transform_index in transform_indices:
            key = make_myperm_key(name, transform_index)
            scored_rows.append(
                {
                    "old_index":transform_index,
                    "point":calculator.point_for_key(key),
                }
            )
        ordered_rows = sorted(scored_rows, key = lambda row:(-row["point"], row["old_index"]))
        reindex_by_name[name] = {
            row["old_index"]:new_index
            for new_index, row in enumerate(ordered_rows)
        }
        point_rows_by_name[name] = tuple(
            {
                "old_index":row["old_index"],
                "new_index":new_index,
                "point":row["point"],
            }
            for new_index, row in enumerate(ordered_rows)
        )

    if not reindex_by_name:
        cube.myperm_transform_key_aliases = {}
        cube.myperm_transform_points = {}
        return {}

    reindexed_myperms = {}
    transform_key_aliases = {}
    for key, moves in cube.myperms.items():
        if isinstance(key, tuple) and len(key) == 2 and key[0] in reindex_by_name:
            new_index = reindex_by_name[key[0]][key[1]]
            new_key = make_myperm_key(key[0], new_index)
            transform_key_aliases[key] = new_key
        else:
            new_key = key
        if new_key in reindexed_myperms:
            raise ValueError(f"point-based myperm key collision: {new_key!r}")
        reindexed_myperms[new_key] = moves

    cube.myperms = reindexed_myperms
    cube.myperm_transform_key_aliases = transform_key_aliases
    cube.myperm_transform_points = point_rows_by_name
    return reindex_by_name


def parse_myperm_points_text(text, puzzle = None):
    """Parse Points.txt content into a MypermPointTable."""
    points_by_part = {}
    default_by_part = {}
    current_part = None
    current_puzzle = None
    target_puzzle = None if puzzle is None else str(puzzle).strip().lower()

    for line_number, raw_line in enumerate(text.splitlines(), start = 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("###"):
            current_puzzle = line.lstrip("#").strip().lower()
            current_part = None
            continue
        if not _point_line_applies_to_puzzle(current_puzzle, target_puzzle):
            continue
        if line.startswith("#"):
            section_name = line[1:].strip().rstrip(":")
            current_part = SECTION_ALIASES.get(section_name, section_name)
            points_by_part.setdefault(current_part, {})
            continue
        if current_part is None:
            raise ValueError(f"point entry before section at line {line_number}")
        if ":" not in line:
            continue

        for token in line.split():
            if ":" not in token:
                raise ValueError(f"invalid point token {token!r} at line {line_number}")
            name, value_text = token.split(":", 1)
            value = _parse_point_value(value_text, line_number)
            if name == "Others":
                default_by_part[current_part] = value
            else:
                if current_part in {"C", "E"}:
                    normalized_name = MypermPointTable({}, {})._normalize_position(name)
                else:
                    normalized_name = name
                points_by_part.setdefault(current_part, {})[normalized_name] = value

    return MypermPointTable(points_by_part = points_by_part, default_by_part = default_by_part)


def load_myperm_points(path = "Points.txt", puzzle = None):
    """Load a point table from Points.txt."""
    if puzzle is None:
        puzzle = "cube"
    return parse_myperm_points_text(Path(path).read_text(encoding = "utf-8"), puzzle = puzzle)


def _point_line_applies_to_puzzle(current_puzzle, target_puzzle):
    if target_puzzle is None:
        return current_puzzle is None
    if current_puzzle == target_puzzle:
        return True
    if target_puzzle in {"cube", "rubiks", "rubiks_cube"} and current_puzzle is None:
        return True
    return False


def _parse_point_value(value_text, line_number):
    try:
        value = float(value_text)
    except ValueError as exc:
        raise ValueError(f"invalid point value {value_text!r} at line {line_number}") from exc
    if value.is_integer():
        return int(value)
    return value
