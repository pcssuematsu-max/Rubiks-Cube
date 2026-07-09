"""Corner Turning Octahedron state/move utilities."""

import itertools
import math

import numpy as np

from core.myperm_keys import make_myperm_key
from fto.cube import FTO_FACE_SIGNS, FtoCube


CTO_MOVE_AXES = {
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
    "U": (0, 1, 0),
    "D": (0, -1, 0),
    "F": (0, 0, 1),
    "B": (0, 0, -1),
}
CTO_AXIS_TO_MOVE = {axis:move for move, axis in CTO_MOVE_AXES.items()}
CTO_TIP_THRESHOLD = 2.0 / 3.0
CTO_WIDE_THRESHOLD = 1.0 / 3.0
CTO_LAYER_EPSILON = 1.0e-8


class CtoCube(FtoCube):
    """3-layer Corner Turning Octahedron model using the shared puzzle API."""

    def _init_move_tables(self):
        self.move = {}
        for face, axis_tuple in CTO_MOVE_AXES.items():
            axis = np.array(axis_tuple, dtype = "f")
            for base, layer in ((face.lower(), "tip"), (face, "wide")):
                clockwise = self._build_move_permutation(axis, layer = layer)
                self.move[base] = clockwise
                self.move[base + "'"] = np.argsort(clockwise)
                self.move[base + "2"] = clockwise[clockwise]

    def _build_move_permutation(self, axis, layer):
        perm = np.arange(self.sticker_num)
        normalized_axis = axis / np.linalg.norm(axis)
        angle = -math.pi / 2.0
        selected = self._selected_stickers(normalized_axis, layer)
        for source in selected:
            center = self._rotate_vector(self.stickers[source]["center"], normalized_axis, angle)
            normal = self._rotate_vector(self.stickers[source]["normal"], normalized_axis, angle)
            target = self._nearest_sticker(center, normal)
            perm[target] = source
        return perm

    def _selected_stickers(self, axis, layer):
        threshold = CTO_TIP_THRESHOLD if layer == "tip" else CTO_WIDE_THRESHOLD
        return [
            index for index, sticker in enumerate(self.stickers)
            if np.dot(axis, sticker["center"]) > (threshold + CTO_LAYER_EPSILON)
        ]

    def _init_move_keys(self):
        self.move_keys = tuple(self.move.keys())
        self.move_len = len(self.move_keys)
        self.key_to_num = {key:index for index, key in enumerate(self.move_keys)}
        self.inverse = {"":"'","'":"","2":"2"}
        self.mult = {
            ("", ""):"2",
            ("", "2"):"'",
            ("", "'"):0,
            ("2", ""):"'",
            ("2", "2"):0,
            ("2", "'"):"",
            ("'", ""):0,
            ("'", "2"):"",
            ("'", "'"):"2",
        }

    def _init_transformation_tables(self):
        self.transformation_keys = []
        axis_set = {tuple(axis) for axis in CTO_MOVE_AXES.values()}
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((-1, 1), repeat = 3):
                matrix = np.zeros((3, 3), dtype = int)
                for row, source in enumerate(perm):
                    matrix[row, source] = signs[row]
                mapped_axes = {tuple(matrix @ np.array(axis, dtype = int)) for axis in axis_set}
                if mapped_axes == axis_set:
                    self.transformation_keys.append({
                        "matrix": matrix,
                        "mirror": round(np.linalg.det(matrix)) < 0,
                    })
        self._move_identity_transformation_to_front()
        self.tf_invert = {
            index:self._inverse_transformation_index(index)
            for index in range(len(self.transformation_keys))
        }

    def _register_myperms2(self):
        self._add_myperm2("CTO-Commutator-A", ("U", "R", "U'", "R'"))
        self._add_myperm2("CTO-Commutator-B", ("U", "R'", "U'", "R"))
        self._add_myperm2("CTO-Commutator-C", ("U2", "R", "U2", "R'"))
        self._add_myperm2("CTO-Commutator-D", ("U2", "R'", "U2", "R"))



        self._add_myperm2("CTO-Flip-A", ("R'","F","R","F'","U","F'","U'","F"))
        self._add_myperm2("CTO-Flip-B", ('F', "R'", 'F', 'R', "F'", 'U', "F'", "U'"))
        self._add_myperm2("CTO-Flip-C", ('D2', 'F', "D'", 'L', 'D2', "L'", 'D', "F'"))
        self._add_myperm2("CTO-Flip-D", ('F', 'L', "D'", 'L2', 'D', "L'", 'F', 'L2', 'F2'))

        self._add_myperm2("CTO-Edge3Cycle-A", ("F","R","U","R'","U'","F'"))
        self._add_myperm2("CTO-Edge3Cycle-B", ('U2', "R'", "U'", 'R', "U'"))
        self._add_myperm2("CTO-Edge3Cycle-C", ('U2', "R", "U'", "R'", "U'"))
        self._add_myperm2("CTO-Edge3Cycle-D", ('U', "R'", 'U2', 'R', 'U'))
        self._add_myperm2("CTO-Edge3Cycle-E", ('R', 'U2', 'R2', 'U2', 'R'))
        self._add_myperm2("CTO-Edge3Cycle-F", ("U'", 'R2', "U'", 'R2', 'U2'))
        self._add_myperm2("CTO-Edge3Cycle-G", ('U2', 'R2', 'U2', 'R2'))
        self._add_myperm2("CTO-Edge3Cycle-H", ("R", "U", "R'", "U", "R", "U2", "R'"))
        self._add_myperm2("CTO-Edge3Cycle-I", ('F', "U'", 'R', 'U', "R'", "F'"))
        self._add_myperm2("CTO-Edge3Cycle-J", ("R'", 'U', 'F', "U'", "F'", 'R'))
        self._add_myperm2("CTO-Edge3Cycle-K", ('R2', "F'", 'U', 'F', "U'", 'R2'))
        self._add_myperm2("CTO-Edge3Cycle-L", ("R'", 'F2', 'U', 'F2', "U'", 'R'))
        self._add_myperm2("CTO-Edge3Cycle-M", ("R'", 'U2', "F'", 'U2', 'F', 'R'))


        self._add_myperm2("CTO-ParityFix-A", ('R', "U'", "R'", 'u', "U'", 'R', "U'", "R'", 'U2'))
        self._add_myperm2("CTO-ParityFix-B", ("u'", 'R2', 'U', 'R2', 'U2', 'R2', 'U', 'R2', 'U'))
        self._add_myperm2("CTO-ParityFix-C", ("u'", 'R', 'U', "R'", 'U', 'R', 'U', "R'", 'U2'))
        self._add_myperm2("CTO-ParityFix-D", ('U2', 'R', "U'", "R'", 'U', 'R', 'U', "R'", 'U', 'R', "U'", "R'", 'u'))
        self._add_myperm2("CTO-ParityFix-E", ("R'", 'U', 'R', "U'", 'R', 'U', 'R', "U'", "R'", "r'"))
        self._add_myperm2("CTO-ParityFix-F", ('R', "U'", 'R', 'U', 'R', "U'", 'R', 'U', 'R', "r'"))
        self._add_myperm2("CTO-ParityFix-G", ('R2', "U'", 'R2', 'u', "U'", 'R2', "U'", 'R2', 'U2'))
        self._add_myperm2("CTO-ParityFix-H", ("F'", 'f', "U'", 'R2', "F'", 'R2', 'U', 'F2', "U'", "F'", 'U'))
        self._add_myperm2("CTO-ParityFix-I", ('B', "D'", 'L', "B'", "L'", 'D', "B'", 'b', "D'", "B'", 'D', 'B'))



        self._add_myperm2("CTO-Center1-A", ('U2', 'R', 'U2', "R'", 'U', 'R', 'U2', "R'", 'U2', 'u2', 'R', 'U', "R'"))
        self._add_myperm2("CTO-Center2-A", ('D', 'B', 'D', "B'", 'D2', 'B', 'D', "B'", "d'", 'F', 'D', 'F', "D'", 'F', 'D', 'F', "D'", 'F', "f'"))




    def _init_groups(self):
        self.corner_index = self._piece_indices_at_vertices()
        self.edge_index = self._piece_indices_at_edges()
        self.center_index = self._center_piece_orbits()
        self.group_pieces = {
            "Corner":self.corner_index,
            "Edge":self.edge_index,
            "Center":self.center_index,
        }
        self.group_indices = {
            group_name:list(range(len(pieces)))
            for group_name, pieces in self.group_pieces.items()
        }

    def _center_piece_orbits(self):
        used = {
            index
            for pieces in (self.corner_index, self.edge_index)
            for piece in pieces
            for index in piece
        }
        center_indices = set(range(self.sticker_num)) - used
        adjacency = {index:set() for index in center_indices}
        for permutation in self.move.values():
            for target, source in enumerate(permutation):
                source = int(source)
                if source in adjacency and target in adjacency:
                    adjacency[source].add(target)
                    adjacency[target].add(source)

        seen = set()
        orbits = []
        for index in sorted(center_indices):
            if index in seen:
                continue
            orbit = []
            stack = [index]
            seen.add(index)
            while stack:
                current = stack.pop()
                orbit.append(current)
                for next_index in adjacency[current]:
                    if next_index not in seen:
                        seen.add(next_index)
                        stack.append(next_index)
            orbits.append(tuple(sorted(orbit)))
        return tuple(sorted(orbits, key = lambda orbit: orbit[0]))

    def invert_str(self, move):
        base, suffix = self._split_move(self.normalize_move_key(move))
        if suffix == "2":
            return base + "2"
        if suffix == "'":
            return base
        return base + "'"

    def _split_move(self, move):
        if move.endswith("'"):
            return move[:-1], "'"
        if move.endswith("2"):
            return move[:-1], "2"
        return move, ""

    def _transform_move(self, move, transformation):
        base, suffix = self._split_move(move)
        is_tip = base.islower()
        axis = self._move_axis(base)
        mapped_axis = tuple(int(value) for value in transformation["matrix"] @ axis.astype(int))
        mapped_base = CTO_AXIS_TO_MOVE[mapped_axis]
        if is_tip:
            mapped_base = mapped_base.lower()
        if transformation["mirror"] and suffix != "2":
            suffix = "" if suffix == "'" else "'"
        return self.normalize_move_key(mapped_base + suffix)

    def _move_axis(self, base):
        return np.array(CTO_MOVE_AXES[base.upper()], dtype = "f")

    def piece_display_name(self, piece_type, piece):
        labels = "-".join(self.index_to_face[index] + str(index % self.face_sticker_count) for index in piece)
        return f"{piece_type}-{labels}"
