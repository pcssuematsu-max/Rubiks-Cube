"""Face Turning Octahedron state/move utilities."""

import itertools
import math
import random
from collections import defaultdict
from functools import reduce

import numpy as np

from core.scramble_selector import ScrambleSelector
from core.myperm_effects import rename_myperms_by_effect
from core.myperm_keys import make_myperm_key, normalize_myperm_name, single_move_myperm_name


FTO_FACE_SIGNS = {
    "URF": (1, 1, 1),
    "UFL": (-1, 1, 1),
    "ULB": (-1, 1, -1),
    "UBR": (1, 1, -1),
    "DFR": (1, -1, 1),
    "DLF": (-1, -1, 1),
    "DBL": (-1, -1, -1),
    "DRB": (1, -1, -1),
}
FTO_FACE_NAMES = tuple(FTO_FACE_SIGNS.keys())
FTO_AXIS_REPRESENTATIVES = ("URF", "UFL", "ULB", "UBR")
FTO_FACE_COLORS = {
    "URF": "A",
    "UFL": "B",
    "ULB": "C",
    "UBR": "D",
    "DFR": "E",
    "DLF": "F",
    "DBL": "G",
    "DRB": "H",
}
FTO_SIGN_TO_FACE = {signs:face for face, signs in FTO_FACE_SIGNS.items()}
FTO_AXIS_TO_MIDDLE = {
    tuple(FTO_FACE_SIGNS[face]):"m" + face
    for face in FTO_AXIS_REPRESENTATIVES
}
FTO_MIDDLE_TO_AXIS = {
    middle:np.array(axis, dtype = "f")
    for axis, middle in FTO_AXIS_TO_MIDDLE.items()
}
FTO_LAYER_THRESHOLD = 1.0 / (3.0 * math.sqrt(3.0))
FTO_LAYER_EPSILON = 1.0e-8


class FtoCube:
    """3-layer Face Turning Octahedron model using the shared puzzle API."""

    def __init__(self, S = "", size = 3, F2L = False, OLL = False, Centers = False, Edges = False, Cross = False):
        self.size = 3
        self.order = 3
        self.F2L = F2L
        self.OLL = OLL
        self.Centers = Centers
        self.Edges = Edges
        self.Cross = Cross
        self.faces = FTO_FACE_NAMES
        self.colors = [FTO_FACE_COLORS[face] for face in self.faces]
        self.color_to_num = {color:index for index, color in enumerate(self.colors)}
        self.face_sticker_count = self.order ** 2
        self.surface_num = self.face_sticker_count
        self.sticker_num = len(self.faces) * self.face_sticker_count

        self._init_stickers()
        self._init_move_tables()
        self._init_move_keys()
        self._init_transformation_tables()
        self._init_myperms()
        self._init_scramble_registry()
        self._init_groups()
        self.state_0 = np.array([FTO_FACE_COLORS[self.index_to_face[i]] for i in range(self.sticker_num)])
        self.state = self.state_0.copy()
        rename_myperms_by_effect(self)
        self._init_feature_layout()
        self.perfect_data = self.makedata()
        self._init_group_values()
        self._init_myperms_index()
        self.scramble_selector = ScrambleSelector(self)
        if S:
            self.set_state(S)

    def _init_stickers(self):
        self.stickers = []
        self.index_to_face = []
        self.face_indices = {face:[] for face in self.faces}
        self.face_sticker_triangles = {face:[] for face in self.faces}
        for face in self.faces:
            vertices = self._face_vertices(face)
            for triangle in self._subtriangle_barycentrics(self.order):
                triangle_points = [self._barycentric_point(vertices, barycentric) for barycentric in triangle]
                center = sum(triangle_points) / 3.0
                self._add_sticker(face, center, self._face_normal(face), triangle_points)

    def _face_vertices(self, face):
        signs = FTO_FACE_SIGNS[face]
        return (
            np.array([signs[0], 0.0, 0.0], dtype = "f"),
            np.array([0.0, signs[1], 0.0], dtype = "f"),
            np.array([0.0, 0.0, signs[2]], dtype = "f"),
        )

    def _face_normal(self, face):
        normal = np.array(FTO_FACE_SIGNS[face], dtype = "f")
        return normal / np.linalg.norm(normal)

    def _subtriangle_barycentrics(self, order):
        triangles = []
        for i in range(order):
            for j in range(order - i):
                k = order - 1 - i - j
                triangles.append(((i + 1, j, k), (i, j + 1, k), (i, j, k + 1)))
        for i in range(order - 1):
            for j in range(order - 1 - i):
                k = order - 2 - i - j
                triangles.append(((i + 1, j + 1, k), (i + 1, j, k + 1), (i, j + 1, k + 1)))
        return triangles

    def _barycentric_point(self, vertices, barycentric):
        return sum(barycentric[index] * vertices[index] for index in range(3)) / float(self.order)

    def _add_sticker(self, face, center, normal, triangle_points):
        index = len(self.stickers)
        self.stickers.append({
            "face": face,
            "center": center,
            "normal": normal,
            "triangle": tuple(point.copy() for point in triangle_points),
        })
        self.index_to_face.append(face)
        self.face_indices[face].append(index)
        self.face_sticker_triangles[face].append(tuple(point.copy() for point in triangle_points))
        return index

    def _init_move_tables(self):
        self.move = {}
        for face in self.faces:
            axis = np.array(FTO_FACE_SIGNS[face], dtype = "f")
            clockwise = self._build_move_permutation(axis, layer = "face")
            self.move[face] = clockwise
            self.move[face + "'"] = np.argsort(clockwise)
        for face in FTO_AXIS_REPRESENTATIVES:
            axis = np.array(FTO_FACE_SIGNS[face], dtype = "f")
            move_key = "m" + face
            clockwise = self._build_move_permutation(axis, layer = "middle")
            self.move[move_key] = clockwise
            self.move[move_key + "'"] = np.argsort(clockwise)

    def _build_move_permutation(self, axis, layer):
        perm = np.arange(self.sticker_num)
        normalized_axis = axis / np.linalg.norm(axis)
        angle = -2.0 * math.pi / 3.0
        selected = self._selected_stickers(normalized_axis, layer)
        for source in selected:
            center = self._rotate_vector(self.stickers[source]["center"], normalized_axis, angle)
            normal = self._rotate_vector(self.stickers[source]["normal"], normalized_axis, angle)
            target = self._nearest_sticker(center, normal)
            perm[target] = source
        return perm

    def _selected_stickers(self, axis, layer):
        if layer == "face":
            return [
                index for index, sticker in enumerate(self.stickers)
                if np.dot(axis, sticker["center"]) > (FTO_LAYER_THRESHOLD + FTO_LAYER_EPSILON)
            ]
        return [
            index for index, sticker in enumerate(self.stickers)
            if abs(np.dot(axis, sticker["center"])) < (FTO_LAYER_THRESHOLD - FTO_LAYER_EPSILON)
        ]

    def _rotate_vector(self, vector, axis, angle):
        axis = axis / np.linalg.norm(axis)
        return (
            vector * math.cos(angle)
            + np.cross(axis, vector) * math.sin(angle)
            + axis * np.dot(axis, vector) * (1.0 - math.cos(angle))
        )

    def _nearest_sticker(self, center, normal):
        return min(
            range(self.sticker_num),
            key = lambda index: (
                np.linalg.norm(center - self.stickers[index]["center"])
                + 0.45 * np.linalg.norm(normal - self.stickers[index]["normal"])
            ),
        )

    def _init_move_keys(self):
        self.move_keys = tuple(self.move.keys())
        self.move_len = len(self.move_keys)
        self.key_to_num = {key:index for index, key in enumerate(self.move_keys)}
        self.inverse = {"":"'","'":""}
        self.mult = {
            ("", ""):"'",
            ("", "'"):0,
            ("'", ""):0,
            ("'", "'"):"",
        }

    def _init_transformation_tables(self):
        self.transformation_keys = []
        face_axes = {tuple(signs) for signs in FTO_FACE_SIGNS.values()}
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((-1, 1), repeat = 3):
                matrix = np.zeros((3, 3), dtype = int)
                for row, source in enumerate(perm):
                    matrix[row, source] = signs[row]
                mapped_axes = {tuple(matrix @ np.array(axis, dtype = int)) for axis in face_axes}
                if mapped_axes == face_axes:
                    self.transformation_keys.append({
                        "matrix": matrix,
                        "mirror": round(np.linalg.det(matrix)) < 0,
                    })
        self._move_identity_transformation_to_front()
        self.tf_invert = {
            index:self._inverse_transformation_index(index)
            for index in range(len(self.transformation_keys))
        }

    def _move_identity_transformation_to_front(self):
        identity = np.eye(3, dtype = int)
        for index, transformation in enumerate(self.transformation_keys):
            if np.array_equal(transformation["matrix"], identity):
                self.transformation_keys.insert(0, self.transformation_keys.pop(index))
                return

    def _inverse_transformation_index(self, transform_index):
        inverse = self.transformation_keys[transform_index]["matrix"].T
        for index, transformation in enumerate(self.transformation_keys):
            if np.array_equal(transformation["matrix"], inverse):
                return index
        return 0

    def _init_myperms(self):
        self.myperms = {}
        self.single_and_rotate = []
        for move_key in self.move_keys:
            key = make_myperm_key(single_move_myperm_name(move_key), 0)
            self.myperms[key] = (move_key,)
            self.single_and_rotate.append(key)
        self.myperms2 = {}
        self.myperms_sequence_group = {}
        self._register_myperms2()
        self._expand_myperms2()

    def _register_myperms2(self):
        self._add_myperm2("FTO-FaceCommutator-A", ("URF", "UFL", "URF'", "UFL'"))
        self._add_myperm2("FTO-FaceCommutator-B", ("URF", "UFL'", "URF'", "UFL"))
        self._add_myperm2("FTO-FaceCommutator-C", ("URF", "ULB", "URF'", "ULB'"))
        self._add_myperm2("FTO-FaceCommutator-D", ("URF", "ULB'", "URF'", "ULB"))

        self._add_myperm2("FTO-FaceOutPerm-A", ('URF', "ULB'", 'URF', 'ULB', 'URF'))

        self._add_myperm2("FTO-MiddleCommutator-A", ("URF", "mUFL", "URF'", "mUFL'"))
        self._add_myperm2("FTO-MiddleCommutator-B", ("URF", "mUFL'", "URF'", "mUFL"))
        self._add_myperm2("FTO-MiddleCommutator-C", ("URF", "mULB", "URF'", "mULB'"))
        self._add_myperm2("FTO-MiddleCommutator-D", ("URF", "mULB'", "URF'", "mULB"))

        self._add_myperm2("FTO-InsideCommutator-A", ("mURF", "mUFL", "mURF'", "mUFL'"))
        self._add_myperm2("FTO-InsideCommutator-B", ("mURF", "mUFL'", "mURF'", "mUFL"))
        self._add_myperm2("FTO-InsideCommutator-C", ("mURF", "mULB", "mURF'", "mULB'"))
        self._add_myperm2("FTO-InsideCommutator-D", ("mURF", "mULB'", "mURF'", "mULB"))


        self._add_myperm2("FTO-EdgeCycle3-A", ("URF","ULB","URF'","ULB","URF","ULB","URF'","ULB"))

        self._add_myperm2("FTO-CenterPerm-A", ("mUFL'","UBR","UFL'","UBR'","mUFL","UBR","UFL","UBR'"))
        self._add_myperm2("FTO-CenterPerm-B", ("mUBR'", "DFR'", 'UBR', 'DFR', 'mUBR', "DFR'", "UBR'", 'DFR'))
        self._add_myperm2("FTO-CenterPerm-C", ('mURF', 'DFR', "DBL'", "DFR'", "mURF'", 'DFR', 'DBL', "DFR'", "mURF'", "UBR'", 'DBL', 'UBR', 'mURF', "UBR'", "DBL'", 'UBR'))




    def _add_myperm2(self, name, moves, add_inverse = True):
        name = normalize_myperm_name(name)
        normalized_moves = self.normalize_move_sequence(moves)
        self.myperms2[name] = normalized_moves
        if add_inverse:
            self.myperms2[name + "-I"] = self.invert_moves(normalized_moves)

    def _expand_myperms2(self):
        for name, moves in self.myperms2.items():
            self.myperms_sequence_group[name] = set()
            for transform_index in range(len(self.transformation_keys)):
                transformed_moves = self.transform(moves, transform_index)
                myperm_key = make_myperm_key(name, transform_index)
                self.myperms[myperm_key] = transformed_moves
                self.myperms_sequence_group[name].add(myperm_key)

    def _init_scramble_registry(self):
        self.my_scrambles = []
        self.my_scrambles2 = {0:{}}
        self.my_scramble_changed_piece_keys = {0:{}}
        self.counter = {0:{}}
        for move_key in self.move_keys:
            self.my_scrambles2[0][move_key] = set()

    def _init_groups(self):
        self.corner_index = self._piece_indices_at_vertices()
        self.edge_index = self._piece_indices_at_edges()
        center_orbits = self._center_piece_orbits()
        self.center_a_index = [(index,) for index in center_orbits[0]]
        self.center_b_index = [(index,) for index in center_orbits[1]]
        self.group_pieces = {
            "Corner":self.corner_index,
            "Edge":self.edge_index,
            "CenterA":self.center_a_index,
            "CenterB":self.center_b_index,
        }
        self.group_indices = {
            group_name:list(range(len(pieces)))
            for group_name, pieces in self.group_pieces.items()
        }

    def _piece_indices_at_vertices(self):
        corner_pieces = []
        for axis_index in range(3):
            for sign in (-1, 1):
                piece = tuple(
                    index for index, sticker in enumerate(self.stickers)
                    if sign * sticker["center"][axis_index] > (2.0 / 3.0)
                )
                corner_pieces.append(tuple(sorted(piece)))
        return corner_pieces

    def _piece_indices_at_edges(self):
        edge_pieces = []
        for axis_a, axis_b in ((0, 1), (0, 2), (1, 2)):
            axis_c = ({0, 1, 2} - {axis_a, axis_b}).pop()
            for sign_a in (-1, 1):
                for sign_b in (-1, 1):
                    piece = tuple(
                        index for index, sticker in enumerate(self.stickers)
                        if (
                            sign_a * sticker["center"][axis_a] > (1.0 / 3.0)
                            and sign_b * sticker["center"][axis_b] > (1.0 / 3.0)
                            and abs(sticker["center"][axis_c]) < (1.0 / 3.0)
                        )
                    )
                    edge_pieces.append(tuple(sorted(piece)))
        return edge_pieces

    def _center_piece_orbits(self):
        center_indices = {
            index
            for face in self.faces
            for index in self.face_indices[face][6:9]
        }
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
        if len(orbits) != 2 or any(len(orbit) != 12 for orbit in orbits):
            raise ValueError("FTO center stickers must split into two 12-piece orbits")
        return tuple(sorted(orbits, key = lambda orbit: orbit[0]))

    def _init_feature_layout(self):
        self.piece_feature_offsets = {}
        self.feature_index_to_piece_color = {}
        self.piece_color_to_feature_index = {}
        self.piece_allowed_colors = self._collect_allowed_piece_colors()
        offset = 0
        for group_name in self.group_pieces:
            for piece in self.group_pieces[group_name]:
                allowed_colors = self.piece_allowed_colors[piece]
                feature_size = len(allowed_colors)
                self.piece_feature_offsets[piece] = (offset, feature_size)
                self.piece_color_to_feature_index[piece] = {}
                for color_offset, color in enumerate(allowed_colors):
                    feature_index = offset + color_offset
                    self.feature_index_to_piece_color[feature_index] = (piece, color)
                    self.piece_color_to_feature_index[piece][color] = feature_index
                offset += feature_size
        self.ips = offset

    def _collect_allowed_piece_colors(self):
        allowed_colors = {}
        for pieces in self.group_pieces.values():
            allowed_tuples = self._collect_allowed_source_tuples(pieces)
            for piece, source_tuples in allowed_tuples.items():
                colors = {
                    self._piece_color(self.state_0, source_tuple)
                    for source_tuple in source_tuples
                }
                allowed_colors[piece] = tuple(sorted(colors))
        return allowed_colors

    def _collect_allowed_source_tuples(self, pieces):
        piece_by_set = {frozenset(piece):piece for piece in pieces}
        allowed_tuples = {piece:{piece} for piece in pieces}
        changed = True
        while changed:
            changed = False
            for permutation in self.move.values():
                for target_piece in pieces:
                    source_order = tuple(int(permutation[index]) for index in target_piece)
                    source_piece = piece_by_set.get(frozenset(source_order))
                    if source_piece is None:
                        continue
                    source_positions = {
                        sticker:index
                        for index, sticker in enumerate(source_piece)
                    }
                    for old_tuple in tuple(allowed_tuples[source_piece]):
                        new_tuple = tuple(
                            old_tuple[source_positions[sticker]]
                            for sticker in source_order
                        )
                        if new_tuple not in allowed_tuples[target_piece]:
                            allowed_tuples[target_piece].add(new_tuple)
                            changed = True
        return allowed_tuples

    def _init_group_values(self):
        self.group_val = {}
        for group_name, pieces in self.group_pieces.items():
            group_vector = np.zeros((1, self.ips), dtype = "f")
            for piece in pieces:
                feature_index = self._piece_feature_index(piece, self._piece_color(self.state_0, piece))
                group_vector[0, feature_index] = 1.0
            self.group_val[group_name] = group_vector
        self.total_val = {key:np.sum(value) for key, value in self.group_val.items()}

    def _init_myperms_index(self):
        self.default_color = {}
        self.num_to_piece = {}
        self.myperms_dict = defaultdict(list)
        self.piece_color_counter = defaultdict(int)
        for pieces in self.group_pieces.values():
            for piece in pieces:
                self.default_color[piece] = self._piece_color(self.state_0, piece)
                for sticker_index in piece:
                    self.num_to_piece[sticker_index] = piece
        for key, moves in self.myperms.items():
            for move in self.invert_moves(moves):
                self.make_move(move)
            for pieces in self.group_pieces.values():
                for piece in pieces:
                    color = self._piece_color(self.state, piece)
                    if color != self.default_color[piece]:
                        dict_key = (piece, color)
                        self.myperms_dict[dict_key].append(key)
                        self.piece_color_counter[dict_key] += 0
            for move in moves:
                self.make_move(move)
        self.myperms_group = self.myperms_dict
        self.myperms_order = {
            group_name:[index for piece in pieces for index in piece]
            for group_name, pieces in self.group_pieces.items()
        }

    def collect_single_moves_and_rotate(self):
        return self.single_and_rotate

    def collect_single_move_and_rotate(self):
        return self.collect_single_moves_and_rotate()

    def create_new_set(self):
        index = len(self.my_scrambles2)
        self.my_scrambles2[index] = {move_key:set() for move_key in self.move_keys}
        self.my_scramble_changed_piece_keys[index] = {}
        self.counter[index] = {}

    def normalize_move_key(self, move):
        if isinstance(move, (tuple, list)):
            if len(move) == 1:
                move = move[0]
            else:
                raise KeyError(move)
        if move in self.move:
            return move
        raise KeyError(move)

    def normalize_move_sequence(self, moves):
        return tuple(self.normalize_move_key(move) for move in moves)

    def format_move(self, move):
        return self.normalize_move_key(move)

    def format_moves(self, moves):
        return tuple(self.format_move(move) for move in self.normalize_move_sequence(moves))

    def register_scramble_sequence(self, level, moves):
        normalized_moves = self.normalize_move_sequence(moves)
        if not normalized_moves:
            return
        while level not in self.my_scrambles2:
            self.create_new_set()
        self.my_scrambles2[level][normalized_moves[-1]].add(normalized_moves)
        self.my_scramble_changed_piece_keys[level][normalized_moves] = tuple(
            self.get_chenged_pieces_keys_from_moves(normalized_moves)
        )

    def get_registered_scramble_changed_piece_keys(self, level, moves):
        return self.my_scramble_changed_piece_keys[level].get(self.normalize_move_sequence(moves))

    def make_move(self, key):
        self.state = self.state[self.move[self.normalize_move_key(key)]]

    def scramble(self, N, Move = None, difficult_mode = False, scramble_mode = None, flip = None, rotate = None, swap = False, add_moves = False, transform_N = None, flip_inside = None, move_count_policy = "prefer_rare"):
        if Move is not None:
            return self._apply_moves_and_return(Move)
        if scramble_mode == "myperms":
            move_count_policy = self.scramble_selector.resolve_move_count_policy(move_count_policy, add_moves)
            return self._guided_scramble(N, move_count_policy = move_count_policy, transform_N = transform_N)
        return self._simple_scramble(N)

    def _apply_moves_and_return(self, moves):
        normalized_moves = self.normalize_move_sequence(moves)
        self._apply_scramble_moves(normalized_moves)
        return normalized_moves

    def _simple_scramble(self, N):
        moves = tuple(random.choice(self.move_keys) for _ in range(max(0, int(N))))
        self._apply_scramble_moves(moves)
        return moves

    def _apply_scramble_moves(self, moves):
        for move in moves:
            self.make_move(move)

    def _guided_scramble(self, level_count, move_count_policy = "prefer_rare", transform_N = None):
        move_count = self._init_scramble_count()
        moves = []
        transform_index = self._resolve_transform_index(transform_N)
        for level_index in range(max(0, int(level_count))):
            selected_moves = self._guided_scramble_moves(level_index, move_count, move_count_policy)
            if not selected_moves:
                continue
            transformed_moves = self.transform(selected_moves, transform_index)
            moves += list(transformed_moves)
            self._apply_scramble_moves(transformed_moves)
        if not moves:
            return self._simple_scramble(1)
        return tuple(moves)

    def _init_scramble_count(self):
        return self.scramble_selector.init_move_count()

    def _resolve_transform_index(self, transform_N):
        if transform_N is None:
            return random.randrange(len(self.transformation_keys))
        return int(transform_N) % len(self.transformation_keys)

    def _guided_scramble_moves(self, level, move_count, move_count_policy):
        return self.scramble_selector.select(level, move_count, move_count_policy = move_count_policy)

    def _collect_scramble_candidates(self, level):
        level = self.scramble_selector.resolve_level(level)
        return self.scramble_selector.collect_candidates(level)

    def _select_scramble_candidate(self, candidates, move_count, add_moves, level):
        if add_moves:
            return self.scramble_selector._select_candidate_max(candidates, move_count, level)
        return self.scramble_selector._select_candidate_min(candidates, move_count, level)

    def _select_candidate_max(self, candidates, move_count, level):
        return self.scramble_selector._select_candidate_max(candidates, move_count, level)

    def _select_candidate_min(self, candidates, move_count, level):
        return self.scramble_selector._select_candidate_min(candidates, move_count, level)

    def _evaluate_piece_color_value(self, changed_piece_keys):
        return sum(self.piece_color_counter[key] for key in changed_piece_keys)

    def _update_piece_color_counter(self, changed_piece_keys):
        self.scramble_selector.update_piece_color_counter(changed_piece_keys)

    def _update_count(self, move_count, moves):
        self.scramble_selector.update_count(move_count, moves)

    def _update_counter_stats(self, level, moves):
        self.scramble_selector.update_counter_stats(level, moves)

    def get_chenged_pieces_keys_from_moves(self, moves):
        old_state = self.state.copy()
        self.reset()
        for move in self.normalize_move_sequence(moves):
            self.make_move(move)
        changed = []
        for group_name, pieces in self.group_pieces.items():
            for piece in pieces:
                if any(self.state[index] != self.state_0[index] for index in piece):
                    changed.append((group_name, piece))
        self.state = old_state
        return changed

    def get_correct_group_count(self, group_name):
        return sum(
            all(self.state[index] == self.state_0[index] for index in piece)
            for piece in self.group_pieces.get(group_name, [])
        )

    def get_correct_group_index(self, group_name):
        return [
            index for index, piece in enumerate(self.group_pieces.get(group_name, []))
            if all(self.state[i] == self.state_0[i] for i in piece)
        ]

    def makedata(self):
        x = np.zeros(self.ips, dtype = "f")
        for group_name in self.group_pieces:
            for piece in self.group_pieces[group_name]:
                x[self._piece_feature_index(piece, self._piece_color(self.state, piece))] = 1.0
        return x

    def _piece_color(self, state, piece):
        return "".join(state[index] for index in piece)

    def _piece_color_index(self, color):
        color_index = 0
        for color_name in color:
            color_index = color_index * len(self.colors) + self.color_to_num[color_name]
        return color_index

    def _color_from_piece_color_index(self, color_index, length):
        values = []
        for _ in range(length):
            values.append(self.colors[color_index % len(self.colors)])
            color_index //= len(self.colors)
        return "".join(reversed(values))

    def _piece_feature_index(self, piece, color):
        return self.piece_color_to_feature_index[piece][color]

    def is_perfect(self):
        return bool((self.state == self.state_0).all())

    def reset(self):
        self.state = self.state_0.copy()

    def state_to_str(self):
        return reduce(lambda x, y: x + y, self.state)

    def set_state(self, S):
        if len(S) == self.sticker_num:
            self.state = np.array(list(S))

    def invert_str(self, move):
        move = self.normalize_move_key(move)
        if move.endswith("'"):
            return move[:-1]
        return move + "'"

    def invert_moves(self, Moves):
        return tuple(self.invert_str(move) for move in self.normalize_move_sequence(Moves)[::-1])

    def simplify(self, move_lis):
        simplified = ()
        for move in self.normalize_move_sequence(move_lis):
            base, suffix = self._split_move(move)
            if simplified:
                last_base, last_suffix = self._split_move(simplified[-1])
                if last_base == base:
                    new_suffix = self.mult[(last_suffix, suffix)]
                    simplified = simplified[:-1]
                    if new_suffix != 0:
                        simplified += (base + new_suffix,)
                    continue
            simplified += (move,)
        return simplified

    def reduce(self, move_lis):
        reduced_moves = []
        visited_states = ["".join(self.state)]
        kept_indices = []
        normalized_moves = self.normalize_move_sequence(move_lis)
        for original_index, move in enumerate(normalized_moves):
            self.make_move(move)
            state_key = "".join(self.state)
            if state_key in visited_states:
                trim_index = visited_states.index(state_key)
                reduced_moves = reduced_moves[:trim_index]
                visited_states = visited_states[:trim_index + 1]
                kept_indices = kept_indices[:trim_index]
            else:
                reduced_moves.append(move)
                visited_states.append(state_key)
                kept_indices.append(original_index)
        for move in self.invert_moves(normalized_moves):
            self.make_move(move)
        return (tuple(reduced_moves), kept_indices)

    def _split_move(self, move):
        if move.endswith("'"):
            return move[:-1], "'"
        return move, ""

    def conjugate(self, A, B):
        return self.simplify(A + B + self.invert_moves(A))

    def commutator(self, A, B):
        return self.simplify(A + B + self.invert_moves(A) + self.invert_moves(B))

    def transform(self, s, i, flip_inside = False, invert = False):
        transform_index = int(i) % len(self.transformation_keys)
        if invert:
            transform_index = self.tf_invert[transform_index]
        transformation = self.transformation_keys[transform_index]
        return tuple(self._transform_move(move, transformation) for move in self.normalize_move_sequence(s))

    def _transform_move(self, move, transformation):
        base, suffix = self._split_move(move)
        is_middle = base.startswith("m")
        axis = self._move_axis(base)
        mapped_axis = tuple(transformation["matrix"] @ axis.astype(int))
        mapped_base, axis_reversed = self._move_name_from_axis(mapped_axis, is_middle)
        if axis_reversed:
            suffix = "" if suffix == "'" else "'"
        if transformation["mirror"]:
            suffix = "" if suffix == "'" else "'"
        return self.normalize_move_key(mapped_base + suffix)

    def _move_axis(self, base):
        if base.startswith("m"):
            return FTO_MIDDLE_TO_AXIS[base]
        return np.array(FTO_FACE_SIGNS[base], dtype = "f")

    def _move_name_from_axis(self, axis, is_middle):
        axis = tuple(int(value) for value in axis)
        if is_middle:
            if axis in FTO_AXIS_TO_MIDDLE:
                return FTO_AXIS_TO_MIDDLE[axis], False
            opposite = tuple(-value for value in axis)
            return FTO_AXIS_TO_MIDDLE[opposite], True
        return FTO_SIGN_TO_FACE[axis], False

    def make_transformations(self, s, Moves):
        scramble_list = []
        move_list = []
        for transform_index in range(len(self.transformation_keys)):
            scramble_list.append(self.transform(s, transform_index, invert = True))
            move_list.append(self.transform(Moves, transform_index, invert = True))
        return scramble_list, move_list

    def flip_inside_moves(self, Moves):
        return self.normalize_move_sequence(Moves)

    def piece_display_name(self, piece_type, piece):
        labels = "-".join(self.index_to_face[index] + str(index % self.face_sticker_count) for index in piece)
        return f"{piece_type}-{labels}"
