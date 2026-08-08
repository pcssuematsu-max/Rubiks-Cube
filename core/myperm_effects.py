"""Sticker-identity based analysis and concise names for myperm effects."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from core.myperm_keys import make_myperm_key, resolve_myperm_key


@dataclass(frozen = True)
class PieceTransfer:
    """One physical piece's source, destination, and destination orientation."""

    group: str
    part_code: str
    source_index: int
    destination_index: int
    source: str
    destination: str
    oriented_destination: str
    orientation: str = ""


@dataclass(frozen = True)
class EffectComponent:
    """Moved pieces belonging to one puzzle piece group."""

    group: str
    part_code: str
    piece_size: int
    transfers: tuple[PieceTransfer, ...]
    cycles: tuple[tuple[str, ...], ...]

    @property
    def moved_count(self):
        return len(self.transfers)

    @property
    def orientation_count(self):
        return sum(bool(transfer.orientation) for transfer in self.transfers)

    def concise_name(self, max_positions = 6):
        permutation_transfers = tuple(
            transfer
            for transfer in self.transfers
            if transfer.source_index != transfer.destination_index
        )
        orientation_transfers = tuple(
            transfer for transfer in self.transfers if transfer.orientation
        )

        if not permutation_transfers:
            return self._orientation_only_name(orientation_transfers, max_positions)

        if orientation_transfers and not self.part_code.startswith("W"):
            return self._mapping_name(max_positions)

        cycle_lengths = tuple(sorted(len(cycle) for cycle in self.cycles))
        if cycle_lengths and all(length == 2 for length in cycle_lengths):
            operation = f"{len(permutation_transfers)}s"
        elif len(self.cycles) == 1:
            operation = f"{len(self.cycles[0])}"
        else:
            shape = self._format_cycle_shape(cycle_lengths)
            operation = f"{len(permutation_transfers)}p[{shape}]"

        positions = ""
        if len(permutation_transfers) <= max_positions:
            positions = self._format_cycles()

        separator = "-" if self.part_code.startswith("W") and self.part_code != "W" else ""
        name = f"{self.part_code}{separator}{operation}{positions}"
        return name

    def is_internal_center_permutation(self):
        """Return whether every moved center stays on its original face."""
        if self.group != "Center" or self.piece_size != 1 or not self.part_code.startswith("Ctr"):
            return False
        return all(
            transfer.source.split("@", 1)[0] == transfer.destination.split("@", 1)[0]
            for transfer in self.transfers
        )

    def _orientation_only_name(self, transfers, max_positions):
        operation = f"{len(transfers)}"
        entries = [
            f"{self._display_position(transfer.source)}>{self._display_position(transfer.oriented_destination)}"
            for transfer in sorted(transfers, key = lambda item:item.source)
        ]
        if len(entries) <= max_positions:
            positions = f"[{';'.join(entries)}]"
        elif self.piece_size == 2 and all(transfer.orientation == "flip" for transfer in transfers):
            positions = "[XY>YX]"
        else:
            positions = ""
        separator = "-" if self.part_code.startswith("W") and self.part_code != "W" else ""
        return f"{self.part_code}{separator}{operation}{positions}"

    def _mapping_name(self, max_positions):
        oriented_cycles = self._format_oriented_piece_cycles(max_positions)
        if oriented_cycles:
            separator = "-" if self.part_code.startswith("W") and self.part_code != "W" else ""
            return f"{self.part_code}{separator}{len(self.transfers)}{oriented_cycles}"

        entries = [
            f"{self._display_position(transfer.source)}>{self._display_position(transfer.oriented_destination)}"
            for transfer in sorted(self.transfers, key = lambda item:(item.source, item.destination))
        ]
        if len(entries) <= max_positions:
            positions = f"[{';'.join(entries)}]"
            shape = ""
        else:
            positions = ""
            cycle_lengths = tuple(sorted(len(cycle) for cycle in self.cycles))
            shape = f"[{self._format_cycle_shape(cycle_lengths)}]" if cycle_lengths else ""
            if self.piece_size == 2 and all(transfer.orientation == "flip" for transfer in self.transfers):
                positions = "[XY>YX]"
        separator = "-" if self.part_code.startswith("W") and self.part_code != "W" else ""
        return f"{self.part_code}{separator}{len(self.transfers)}{shape}{positions}"

    def _format_oriented_piece_cycles(self, max_positions):
        if self.part_code not in {"C", "E", "ME", "EAll"} or self.piece_size not in {2, 3}:
            return ""
        if not self.cycles or len(self.transfers) > max_positions:
            return ""
        if sum(len(cycle) for cycle in self.cycles) != len(self.transfers):
            return ""

        oriented_map = {}
        for transfer in self.transfers:
            source = self._display_position(transfer.source)
            oriented_destination = self._display_position(transfer.oriented_destination)
            for source_state, destination_state in self._oriented_equivalent_states(source, oriented_destination):
                oriented_map[source_state] = destination_state

        formatted = []
        for physical_cycle in self.cycles:
            start = self._display_position(physical_cycle[0])
            current = start
            oriented_cycle = []
            for _ in range(len(oriented_map) + 1):
                oriented_cycle.append(current)
                current = oriented_map.get(current)
                if current is None:
                    return ""
                if current == start:
                    break
            else:
                return ""

            if len(oriented_cycle) != len(physical_cycle):
                return ""
            formatted.append(">".join(oriented_cycle))
        return f"[{';'.join(formatted)}]" if formatted else ""

    def _oriented_equivalent_states(self, source, destination):
        if self.piece_size == 2:
            return (
                (source, destination),
                (self._reverse_edge_label(source), self._reverse_edge_label(destination)),
            )
        return tuple(
            (self._rotate_piece_label(source, shift), self._rotate_piece_label(destination, shift))
            for shift in range(self.piece_size)
        )

    def _reverse_edge_label(self, label):
        if "@" in label:
            base, suffix = label.split("@", 1)
            return f"{self._reverse_edge_base_label(base)}@{suffix}"
        return self._reverse_edge_base_label(label)

    def _reverse_edge_base_label(self, label):
        if "." in label:
            return ".".join(reversed(label.split(".")))
        return label[::-1]

    def _rotate_piece_label(self, label, shift):
        if "." in label:
            parts = label.split(".")
            return ".".join(parts[shift:] + parts[:shift])
        return label[shift:] + label[:shift]

    def _format_cycles(self):
        formatted = []
        for cycle in self.cycles:
            cycle = tuple(self._display_position(position) for position in cycle)
            if len(cycle) == 2:
                formatted.append(f"{cycle[0]}<>{cycle[1]}")
            else:
                formatted.append(">".join(cycle))
        return f"[{';'.join(formatted)}]" if formatted else ""

    def _display_position(self, position):
        if self.part_code in {"E", "ME", "EAll"} and position.endswith("@M"):
            return position[:-2]
        if self.part_code.startswith("W") and "@" in position:
            base, axis = position.split("@", 1)
            return f"{base}@{axis.lstrip('0123456789')}"
        return position

    def _format_cycle_shape(self, cycle_lengths):
        counts = []
        for length in sorted(set(cycle_lengths)):
            count = cycle_lengths.count(length)
            counts.append(f"{length}x{count}" if count > 1 else str(length))
        return "+".join(counts)


@dataclass(frozen = True)
class MypermEffect:
    """Structured effect of one myperm sequence."""

    original_key: object
    components: tuple[EffectComponent, ...]

    @property
    def moved_count(self):
        return sum(component.moved_count for component in self.components)

    @property
    def orientation_count(self):
        return sum(component.orientation_count for component in self.components)

    def concise_name(self, max_positions = 6, max_length = 160):
        visible_components = tuple(
            component
            for component in self.components
            if not component.is_internal_center_permutation()
        )
        if not visible_components:
            return "Identity"
        center_bar_name = self._center_bar_name(visible_components, max_positions)
        if center_bar_name:
            return center_bar_name
        name = "+".join(
            component.concise_name(max_positions = max_positions)
            for component in visible_components
        )
        if len(name) > max_length:
            name = "+".join(
                component.concise_name(max_positions = 0)
                for component in visible_components
            )
        return name

    def _center_bar_name(self, components, max_positions):
        part_codes = {component.part_code for component in components}
        if part_codes == {"CtrPlus"}:
            return self._center_mid_bar_name(components, max_positions)
        if part_codes != {"CtrX", "CtrPlus", "CtrObl"}:
            return ""
        if any(component.group != "Center" or component.piece_size != 1 for component in components):
            return ""

        transfers = tuple(transfer for component in components for transfer in component.transfers)
        bar_by_position = self._center_bar_positions(transfers)
        if not bar_by_position:
            return ""
        return self._center_bar_name_from_position_map(
            transfers,
            bar_by_position,
            prefix = "CtrBar",
            max_positions = max_positions,
        )

    def _center_mid_bar_name(self, components, max_positions):
        if any(component.group != "Center" or component.piece_size != 1 for component in components):
            return ""

        transfers = tuple(transfer for component in components for transfer in component.transfers)
        bar_by_position = self._center_mid_bar_positions(transfers)
        if not bar_by_position:
            return ""
        return self._center_bar_name_from_position_map(
            transfers,
            bar_by_position,
            prefix = "CtrMidBar",
            max_positions = max_positions,
        )

    def _center_bar_name_from_position_map(self, transfers, bar_by_position, prefix, max_positions):
        bar_map = {}
        for transfer in transfers:
            source_bar = bar_by_position.get(transfer.source)
            destination_bar = bar_by_position.get(transfer.destination)
            if source_bar is None or destination_bar is None:
                return ""
            if source_bar == destination_bar:
                continue
            if source_bar in bar_map and bar_map[source_bar] != destination_bar:
                return ""
            bar_map[source_bar] = destination_bar

        if not bar_map:
            return ""
        if not set(bar_map.values()).issubset(set(bar_map)):
            return ""

        cycles = self._mapping_cycles(bar_map)
        if not cycles:
            return ""
        cycle_lengths = tuple(sorted(len(cycle) for cycle in cycles))
        moved_count = sum(cycle_lengths)
        if cycle_lengths and all(length == 2 for length in cycle_lengths):
            operation = f"{moved_count}s"
        elif len(cycles) == 1:
            operation = str(cycle_lengths[0])
        else:
            operation = f"{moved_count}p[{self._format_cycle_shape(cycle_lengths)}]"

        positions = ""
        if moved_count <= max_positions:
            positions = self._format_bar_cycles(cycles)
        return f"{prefix}{operation}{positions}"

    def _center_bar_positions(self, transfers):
        positions = {
            position
            for transfer in transfers
            for position in (transfer.source, transfer.destination)
        }
        candidate_counts = defaultdict(int)
        candidates_by_position = {}
        for position in positions:
            candidates = self._center_bar_candidates(position)
            if not candidates:
                return {}
            candidates_by_position[position] = candidates
            for candidate in candidates:
                candidate_counts[candidate] += 1

        bar_by_position = {}
        for position, candidates in candidates_by_position.items():
            valid_candidates = [
                candidate
                for candidate in candidates
                if candidate_counts[candidate] >= 3
            ]
            if len(valid_candidates) != 1:
                return {}
            bar_by_position[position] = valid_candidates[0]
        return bar_by_position

    def _center_bar_candidates(self, position):
        if "@" not in position:
            return ()
        face, coordinates = position.split("@", 1)
        parts = coordinates.split(".")
        if len(parts) != 2:
            return ()
        return tuple(
            f"{face}@{coordinate}"
            for coordinate in parts
            if coordinate not in {"M", "E", "S"}
        )

    def _center_mid_bar_positions(self, transfers):
        positions = {
            position
            for transfer in transfers
            for position in (transfer.source, transfer.destination)
        }
        candidate_counts = defaultdict(int)
        candidates_by_position = {}
        for position in positions:
            candidates = self._center_mid_bar_candidates(position)
            if not candidates:
                return {}
            candidates_by_position[position] = candidates
            for candidate in candidates:
                candidate_counts[candidate] += 1

        bar_by_position = {}
        for position, candidates in candidates_by_position.items():
            valid_candidates = [
                candidate
                for candidate in candidates
                if candidate_counts[candidate] >= 2
            ]
            if len(valid_candidates) != 1:
                return {}
            bar_by_position[position] = valid_candidates[0]
        return bar_by_position

    def _center_mid_bar_candidates(self, position):
        if "@" not in position:
            return ()
        face, coordinates = position.split("@", 1)
        parts = coordinates.split(".")
        if len(parts) != 2:
            return ()
        candidates = []
        for coordinate in parts:
            if coordinate in {"M", "E", "S"}:
                continue
            axis = coordinate.lstrip("0123456789")
            if not axis:
                return ()
            candidates.append(f"{face}@{axis}")
        return tuple(candidates)

    def _mapping_cycles(self, mapping):
        cycles = []
        visited = set()
        for start in sorted(mapping):
            if start in visited:
                continue
            cycle = []
            current = start
            while current not in visited:
                visited.add(current)
                cycle.append(current)
                current = mapping.get(current)
                if current is None:
                    return ()
            if current != start:
                return ()
            if len(cycle) > 1:
                cycles.append(tuple(cycle))
        return tuple(cycles)

    def _format_bar_cycles(self, cycles):
        entries = []
        for cycle in cycles:
            if len(cycle) == 2:
                entries.append(f"{cycle[0]}<>{cycle[1]}")
            else:
                entries.append(">".join(cycle))
        return f"[{';'.join(entries)}]" if entries else ""

    def _format_cycle_shape(self, cycle_lengths):
        counts = []
        for length in sorted(set(cycle_lengths)):
            count = cycle_lengths.count(length)
            counts.append(f"{length}x{count}" if count > 1 else str(length))
        return "+".join(counts)


class MypermEffectAnalyzer:
    """Analyze physical piece transfers using unique solved sticker identities."""

    def __init__(self, cube):
        self.cube = cube
        self._groups = self._piece_groups()

    def analyze(self, key_or_moves):
        key, moves = self._resolve_moves(key_or_moves)
        sticker_state = np.arange(len(self.cube.state_0), dtype = int)
        for move in moves:
            sticker_state = sticker_state[np.asarray(self.cube.move[move], dtype = int)]

        components = []
        for group, pieces in self._groups.items():
            components.extend(self._analyze_group(group, pieces, sticker_state))
        if self._is_rubiks_cube() and "Edge" in self._groups:
            components = self._combine_rubiks_edge_bundle(components, sticker_state)
        components.sort(key = lambda component:(component.part_code, component.group))
        return MypermEffect(key, tuple(components))

    def proposed_name(self, key_or_moves, max_positions = 6):
        return self.analyze(key_or_moves).concise_name(max_positions = max_positions)

    def _resolve_moves(self, key_or_moves):
        resolved_key = resolve_myperm_key(self.cube, key_or_moves)
        if resolved_key is not None:
            return resolved_key, tuple(self.cube.myperms[resolved_key])
        if isinstance(key_or_moves, str):
            if key_or_moves in self.cube.myperms2:
                return key_or_moves, tuple(self.cube.myperms2[key_or_moves])
        return None, tuple(key_or_moves)

    def _piece_groups(self):
        if hasattr(self.cube, "group_pieces"):
            groups = self.cube.group_pieces
        else:
            groups = {
                "Corner":getattr(self.cube, "corner_index", ()),
                "Edge":getattr(self.cube, "edge_index", ()),
                "Center":getattr(self.cube, "center_index", ()),
            }
        return {
            str(group):tuple(tuple(int(index) for index in piece) for piece in pieces)
            for group, pieces in groups.items()
            if pieces
        }

    def _analyze_group(self, group, pieces, sticker_state):
        source_lookup = {frozenset(piece):index for index, piece in enumerate(pieces)}
        transfers_by_part = defaultdict(list)

        for destination_index, destination_piece in enumerate(pieces):
            actual_stickers = tuple(int(sticker_state[index]) for index in destination_piece)
            source_index = source_lookup.get(frozenset(actual_stickers))
            if source_index is None:
                raise ValueError(
                    f"{group} destination {destination_piece} does not contain one registered physical piece"
                )
            source_piece = pieces[source_index]
            orientation_permutation = self._orientation_permutation(source_piece, actual_stickers)
            orientation = self._orientation_code(source_piece, actual_stickers, orientation_permutation)
            if source_index == destination_index and not orientation:
                continue
            part_code = self._part_code(group, destination_piece)
            destination = self._position_name(group, destination_piece)
            transfers_by_part[(part_code, len(destination_piece))].append(
                PieceTransfer(
                    group = group,
                    part_code = part_code,
                    source_index = source_index,
                    destination_index = destination_index,
                    source = self._position_name(group, source_piece),
                    destination = destination,
                    oriented_destination = self._oriented_destination_name(
                        group,
                        destination_piece,
                        destination,
                        part_code,
                        orientation_permutation,
                    ),
                    orientation = orientation,
                )
            )

        components = []
        for (part_code, piece_size), transfers in transfers_by_part.items():
            cycles = self._cycles(transfers)
            components.append(
                EffectComponent(
                    group = group,
                    part_code = part_code,
                    piece_size = piece_size,
                    transfers = tuple(transfers),
                    cycles = cycles,
                )
            )
        return components

    def _orientation_permutation(self, source_piece, actual_stickers):
        return tuple(source_piece.index(sticker) for sticker in actual_stickers)

    def _orientation_code(self, source_piece, actual_stickers, permutation = None):
        if permutation is None:
            permutation = self._orientation_permutation(source_piece, actual_stickers)
        identity = tuple(range(len(source_piece)))
        if permutation == identity:
            return ""
        if len(source_piece) == 2 and permutation == (1, 0):
            return "flip"
        for shift in range(1, len(source_piece)):
            if permutation == tuple((index + shift) % len(source_piece) for index in identity):
                if len(source_piece) == 3:
                    return "+" if shift == 1 else "-"
                return f"rot{shift}"
        faces = tuple(self._sticker_face(sticker) for sticker in actual_stickers)
        separator = "" if all(len(face) == 1 for face in faces) else "."
        return "as:" + separator.join(faces)

    def _cycles(self, transfers):
        permutation = {
            transfer.source_index:transfer.destination_index
            for transfer in transfers
            if transfer.source_index != transfer.destination_index
        }
        positions = {transfer.source_index:transfer.source for transfer in transfers}
        positions.update({transfer.destination_index:transfer.destination for transfer in transfers})
        visited = set()
        cycles = []
        for start in sorted(permutation, key = lambda index:positions[index]):
            if start in visited:
                continue
            cycle_indices = []
            current = start
            while current not in visited and current in permutation:
                visited.add(current)
                cycle_indices.append(current)
                current = permutation[current]
            if len(cycle_indices) > 1:
                cycle = tuple(positions[index] for index in cycle_indices)
                cycle = self._rotate_cycle_to_smallest(cycle)
                cycles.append(cycle)
        return tuple(sorted(cycles))

    def _combine_rubiks_edge_bundle(self, components, sticker_state):
        """Collapse MidEdge and every Wing when each full edge moves uniformly."""
        pieces = self._groups["Edge"]
        if not any(self._part_code("Edge", piece).startswith("W") for piece in pieces):
            return components

        source_lookup = {frozenset(piece):index for index, piece in enumerate(pieces)}
        mappings = {}
        members_by_edge = defaultdict(list)
        for destination_index, destination_piece in enumerate(pieces):
            actual_stickers = tuple(int(sticker_state[index]) for index in destination_piece)
            source_index = source_lookup[frozenset(actual_stickers)]
            source_piece = pieces[source_index]
            source_edge = self._edge_base_position(source_piece)
            destination_edge = self._edge_base_position(destination_piece)
            mappings[source_index] = {
                "destination_index":destination_index,
                "source_edge":source_edge,
                "destination_edge":destination_edge,
                "orientation":self._orientation_code(source_piece, actual_stickers),
                "part_code":self._part_code("Edge", source_piece),
            }
            members_by_edge[source_edge].append(source_index)

        bundle_rows = []
        for source_edge, member_indices in members_by_edge.items():
            member_rows = [mappings[index] for index in member_indices]
            middle_rows = [row for row in member_rows if row["part_code"] in {"E", "ME"}]
            if len(middle_rows) != 1:
                return components
            middle_row = middle_rows[0]
            if any(row["destination_edge"] != middle_row["destination_edge"] for row in member_rows):
                return components
            if any(row["orientation"] != middle_row["orientation"] for row in member_rows):
                return components

            middle_changed = (
                source_edge != middle_row["destination_edge"]
                or bool(middle_row["orientation"])
            )
            any_member_changed = any(
                index != row["destination_index"] or bool(row["orientation"])
                for index, row in zip(member_indices, member_rows)
            )
            if middle_changed != any_member_changed:
                return components
            if middle_changed:
                bundle_rows.append((source_edge, middle_row["destination_edge"], middle_row["orientation"]))

        edge_positions = sorted(members_by_edge)
        position_to_index = {position:index for index, position in enumerate(edge_positions)}
        bundle_transfers = tuple(
            PieceTransfer(
                group = "EdgeBundle",
                part_code = "EAll",
                source_index = position_to_index[source],
                destination_index = position_to_index[destination],
                source = f"{source}@M",
                destination = f"{destination}@M",
                oriented_destination = self._oriented_mid_edge_name(destination, orientation),
                orientation = orientation,
            )
            for source, destination, orientation in bundle_rows
        )
        if not bundle_transfers:
            return components

        bundle_component = EffectComponent(
            group = "EdgeBundle",
            part_code = "EAll",
            piece_size = 2,
            transfers = bundle_transfers,
            cycles = self._cycles(bundle_transfers),
        )
        return [component for component in components if component.group != "Edge"] + [bundle_component]

    def _edge_base_position(self, piece):
        return self._rubiks_position_name("Edge", piece).split("@", 1)[0]

    def _oriented_mid_edge_name(self, destination, orientation):
        oriented_faces = destination[::-1] if orientation == "flip" else destination
        return f"{oriented_faces}@M"

    def _rotate_cycle_to_smallest(self, cycle):
        start = min(range(len(cycle)), key = lambda index:cycle[index])
        return cycle[start:] + cycle[:start]

    def _part_code(self, group, piece):
        if group == "Corner":
            return "C"
        if group == "Edge":
            if self._is_rubiks_cube():
                axis = self.cube._edge_axis_label(piece)
                if axis in {"M", "E", "S"}:
                    return "ME" if getattr(self.cube, "size", 3) > 3 else "E"
                layer = "".join(character for character in axis if character.isdigit())
                return f"W{layer}" if layer else "W"
            if "MasterPyraminx" in self.cube.__class__.__name__:
                return "OE"
            return "E"
        if group == "MidEdge":
            return "ME"
        if group.startswith("Center"):
            if self._is_rubiks_cube() and group == "Center":
                return self._rubiks_center_part_code(piece)
            return "Ctr" + group[len("Center"):]
        return group

    def _rubiks_center_part_code(self, piece):
        face, row, column = self.cube._index_to_face_row_col(piece[0])
        horizontal = self.cube._coordinate_axis_label(face, column, axis = "horizontal")
        vertical = self.cube._coordinate_axis_label(face, row, axis = "vertical")
        horizontal_middle = horizontal in {"M", "E", "S"}
        vertical_middle = vertical in {"M", "E", "S"}
        if horizontal_middle and vertical_middle:
            return "CtrCore"
        if horizontal_middle or vertical_middle:
            return "CtrPlus"
        if self._axis_depth(horizontal) == self._axis_depth(vertical):
            return "CtrX"
        return "CtrObl"

    def _axis_depth(self, label):
        digits = ""
        for character in label:
            if character.isdigit():
                digits += character
            else:
                break
        return int(digits or 1)

    def _position_name(self, group, piece):
        if self._is_rubiks_cube():
            return self._rubiks_position_name(group, piece)
        faces = tuple(self._sticker_face(index) for index in piece)
        if self._is_octahedral_puzzle() and len(piece) > 1:
            common = set(faces[0])
            for face in faces[1:]:
                common &= set(face)
            if common:
                return "".join(face for face in "UDFBLR" if face in common)
        if len(piece) == 1:
            local_index = int(piece[0] % getattr(self.cube, "face_sticker_count", self.cube.surface_num))
            return f"{faces[0]}{local_index}"
        separator = "" if all(len(face) == 1 for face in faces) else "."
        return separator.join(faces)

    def _rubiks_position_name(self, group, piece):
        faces = tuple(self.cube._move_face_label(index) for index in piece)
        if group == "Corner":
            return "".join(faces)
        if group == "Edge":
            axis = self.cube._edge_axis_label(piece)
            position = "".join(faces)
            return f"{position}@M" if axis in {"M", "E", "S"} else f"{position}@{axis}"
        face, row, column = self.cube._index_to_face_row_col(piece[0])
        horizontal = self.cube._coordinate_axis_label(face, column, axis = "horizontal")
        vertical = self.cube._coordinate_axis_label(face, row, axis = "vertical")
        if horizontal in {"M", "E", "S"} and vertical in {"M", "E", "S"}:
            return face
        return f"{face}@{horizontal}.{vertical}"

    def _oriented_destination_name(self, group, piece, destination, part_code, permutation):
        if permutation == tuple(range(len(piece))) or part_code.startswith("W"):
            return destination
        faces = tuple(self._sticker_face(index) for index in piece)
        oriented_faces = tuple(faces[index] for index in permutation)
        separator = "" if all(len(face) == 1 for face in oriented_faces) else "."
        oriented_name = separator.join(oriented_faces)
        if self._is_rubiks_cube() and group == "Edge":
            return f"{oriented_name}@M"
        return oriented_name

    def _sticker_face(self, index):
        if self._is_rubiks_cube():
            return self.cube._move_face_label(index)
        if self.cube.__class__.__module__ == "megaminx.cube":
            from megaminx.cube import DISPLAY_FACE_NAMES
            return DISPLAY_FACE_NAMES[str(self.cube.state_0[index])]
        if hasattr(self.cube, "index_to_face"):
            return str(self.cube.index_to_face[index])
        return str(self.cube.state_0[index])

    def _is_rubiks_cube(self):
        return self.cube.__class__.__module__ == "cube.rubiks_cube"

    def _is_octahedral_puzzle(self):
        return self.cube.__class__.__module__ in {"fto.cube", "cto.cube"}


def rename_myperms_by_effect(cube):
    """Replace registered myperm base names with effect-based names."""
    old_names = tuple(getattr(cube, "myperms2", ()))
    if not old_names:
        cube.myperm_name_aliases = {}
        cube.myperm_key_aliases = {}
        return {}

    analyzer = MypermEffectAnalyzer(cube)
    effect_names = {}
    grouped_names = defaultdict(list)
    for old_name in old_names:
        old_key = make_myperm_key(old_name, 0)
        if old_key not in cube.myperms:
            continue
        if _keeps_source_myperm_name(old_name):
            effect_name = old_name
        else:
            effect_name = analyzer.proposed_name(old_key)
        effect_names[old_name] = effect_name
        grouped_names[effect_name].append(old_name)

    name_aliases = {}
    for effect_name, matching_names in grouped_names.items():
        for variant_index, old_name in enumerate(sorted(matching_names), start = 1):
            new_name = effect_name
            if len(matching_names) > 1:
                new_name += f"~v{variant_index:02d}"
            if old_name != new_name:
                name_aliases[old_name] = new_name

    renamed_myperms = {}
    key_aliases = {}
    current_key_map = {}
    for old_key, moves in cube.myperms.items():
        if isinstance(old_key, tuple) and old_key[0] in name_aliases:
            new_key = make_myperm_key(name_aliases[old_key[0]], old_key[1])
            key_aliases[old_key] = new_key
        else:
            new_key = old_key
        if new_key in renamed_myperms:
            raise ValueError(f"effect-based myperm key collision: {new_key!r}")
        renamed_myperms[new_key] = moves
        current_key_map[old_key] = new_key

    transform_indices_by_name = defaultdict(set)
    for key in renamed_myperms:
        if isinstance(key, tuple) and len(key) == 2:
            transform_indices_by_name[key[0]].add(key[1])

    for legacy_name, current_name in name_aliases.items():
        for transform_index in transform_indices_by_name.get(current_name, ()):
            legacy_key = make_myperm_key(legacy_name, transform_index)
            current_key = make_myperm_key(current_name, transform_index)
            if current_key in renamed_myperms:
                key_aliases.setdefault(legacy_key, current_key)

    for original_key, reindexed_key in getattr(cube, "myperm_transform_key_aliases", {}).items():
        current_key = current_key_map.get(reindexed_key, reindexed_key)
        if current_key in renamed_myperms:
            key_aliases[original_key] = current_key

    cube.myperms = renamed_myperms
    cube.myperm_name_aliases = name_aliases
    cube.myperm_key_aliases = key_aliases
    cube.myperm_legacy_names = {new_name:old_name for old_name, new_name in name_aliases.items()}
    cube.myperms2 = {
        name_aliases.get(old_name, old_name):moves
        for old_name, moves in cube.myperms2.items()
    }

    if hasattr(cube, "myperms_sequence_group"):
        cube.myperms_sequence_group = {
            name_aliases.get(old_name, old_name):{
                current_key_map.get(key, key) for key in keys
            }
            for old_name, keys in cube.myperms_sequence_group.items()
        }
    if hasattr(cube, "myperm_transform_points"):
        cube.myperm_transform_points = {
            name_aliases.get(old_name, old_name):rows
            for old_name, rows in cube.myperm_transform_points.items()
        }
    if hasattr(cube, "single_and_rotate"):
        cube.single_and_rotate = [current_key_map.get(key, key) for key in cube.single_and_rotate]
    return name_aliases


def _keeps_source_myperm_name(name):
    """Keep intentionally compact source names that are clearer than effects."""
    return str(name).startswith(("OuterCenterBar", "MidCenterBar"))
