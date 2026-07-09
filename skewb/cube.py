"""Skewb state/move utilities using the shared puzzle interface."""

import itertools
import math
import random
from collections import defaultdict
from functools import reduce

import numpy as np

from core.scramble_selector import ScrambleSelector
from core.myperm_effects import rename_myperms_by_effect
from core.myperm_keys import make_myperm_key, normalize_myperm_name, single_move_myperm_name


SKEWB_FACES = ("U", "R", "F", "D", "L", "B")
FACE_NORMALS = {
    "U": np.array([0.0, 1.0, 0.0]),
    "D": np.array([0.0, -1.0, 0.0]),
    "R": np.array([1.0, 0.0, 0.0]),
    "L": np.array([-1.0, 0.0, 0.0]),
    "F": np.array([0.0, 0.0, 1.0]),
    "B": np.array([0.0, 0.0, -1.0]),
}
FACE_AXIS = {
    "U": (1, 1),
    "D": (1, -1),
    "R": (0, 1),
    "L": (0, -1),
    "F": (2, 1),
    "B": (2, -1),
}
CORNER_SIGNS = tuple(itertools.product((-1, 1), repeat = 3))
MOVE_AXES = {
    "URF": np.array([1.0, 1.0, 1.0]),
    "ULB": np.array([-1.0, 1.0, -1.0]),
    "DFL": np.array([-1.0, -1.0, 1.0]),
    "DRB": np.array([1.0, -1.0, -1.0]),
    "UFL": np.array([-1.0, 1.0, 1.0]),
    "UBR": np.array([1.0, 1.0, -1.0]),
    "DBL": np.array([-1.0, -1.0, -1.0]),
    "DFR": np.array([1.0, -1.0, 1.0]),
}
AXIS_TO_MOVE = {tuple(axis.astype(int)):move for move, axis in MOVE_AXES.items()}


class SkewbCube:
    """Skewb puzzle model compatible with Rubiks/Megaminx/Pyraminx managers."""

    def __init__(self, S = "", size = 3, F2L = False, OLL = False, Centers = False, Edges = False, Cross = False):
        self.size = 3
        self.order = 3
        self.F2L = F2L
        self.OLL = OLL
        self.Centers = Centers
        self.Edges = Edges
        self.Cross = Cross
        self.faces = SKEWB_FACES
        self.colors = list(SKEWB_FACES)
        self.color_to_num = {color:index for index, color in enumerate(self.colors)}
        self.face_sticker_count = 5
        self.surface_num = self.face_sticker_count
        self.sticker_num = len(self.faces) * self.face_sticker_count
        self.ips = 0

        self._init_stickers()
        self._init_move_tables()
        self._init_move_keys()
        self._init_transformation_tables()
        self._init_myperms()
        self._init_scramble_registry()
        self._init_groups()
        self.state_0 = np.array([self.index_to_face[i] for i in range(self.sticker_num)])
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
        self.corner_lookup = {}
        for face in self.faces:
            normal = FACE_NORMALS[face]
            center_index = self._add_sticker(face, "center", None, normal, normal.copy())
            self.center_index_by_face = getattr(self, "center_index_by_face", {})
            self.center_index_by_face[face] = center_index
            for signs in self._face_corner_signs(face):
                center = self._corner_sticker_center(face, signs)
                index = self._add_sticker(face, "corner", signs, center, normal)
                self.corner_lookup[(face, signs)] = index

    def _add_sticker(self, face, kind, signs, center, normal):
        index = len(self.stickers)
        self.stickers.append({
            "face": face,
            "kind": kind,
            "signs": signs,
            "center": center,
            "normal": normal,
        })
        self.index_to_face.append(face)
        self.face_indices[face].append(index)
        return index

    def _face_corner_signs(self, face):
        axis, sign = FACE_AXIS[face]
        return tuple(signs for signs in CORNER_SIGNS if signs[axis] == sign)

    def _corner_sticker_center(self, face, signs):
        values = np.array(signs, dtype = "f")
        axis, sign = FACE_AXIS[face]
        values[axis] = float(sign)
        for index in range(3):
            if index != axis:
                values[index] *= 0.58
        return values

    def _init_move_tables(self):
        self.move = {}
        for move_key, axis in MOVE_AXES.items():
            # A Skewb turn is a 120-degree rotation around a body diagonal.
            # The axis name identifies the visible corner; the opposite corner
            # lies on the same diagonal.
            clockwise = self._build_move_permutation(axis, clockwise = True)
            self.move[move_key] = clockwise
            self.move[move_key + "'"] = np.argsort(clockwise)

    def _build_move_permutation(self, axis, clockwise = True):
        perm = np.arange(self.sticker_num)
        normalized_axis = axis / np.linalg.norm(axis)
        angle = (-2.0 if clockwise else 2.0) * math.pi / 3.0
        selected = self._selected_stickers(normalized_axis)
        for source in selected:
            center = self._rotate_vector(self.stickers[source]["center"], normalized_axis, angle)
            normal = self._rotate_vector(self.stickers[source]["normal"], normalized_axis, angle)
            target = self._nearest_sticker(center, normal)
            perm[target] = source
        return perm

    def _selected_stickers(self, axis):
        return [
            index for index, sticker in enumerate(self.stickers)
            if np.dot(axis, sticker["center"]) > 1.0e-8
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
                + 0.35 * np.linalg.norm(normal - self.stickers[index]["normal"])
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
        axes = set(AXIS_TO_MOVE.keys())
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((-1, 1), repeat = 3):
                matrix = np.zeros((3, 3), dtype = int)
                for row, source in enumerate(perm):
                    matrix[row, source] = signs[row]
                mapped_axes = {tuple(matrix @ np.array(axis, dtype = int)) for axis in axes}
                if mapped_axes == axes:
                    self.transformation_keys.append({
                        "matrix": matrix,
                        "mirror": round(np.linalg.det(matrix)) < 0,
                    })
        self.tf_invert = {
            index:self._inverse_transformation_index(index)
            for index in range(len(self.transformation_keys))
        }

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
        self._add_myperm2("Skewb-Sledgehammer-A", ("URF'", "ULB", "URF", "ULB'"))
        self._add_myperm2("Skewb-Sledgehammer-B", ("URF", "ULB'", "URF'", "ULB"))
        self._add_myperm2("Skewb-Sexy-A", ("URF", "DFL", "URF'", "DFL'"))
        self._add_myperm2("Skewb-Sexy-B", ("URF'", "DFL", "URF", "DFL'"))

        self._add_myperm2("Skewb-Corner2Twist", ("DRB'", 'DBL', 'DRB', 'DFR', 'DFL', 'DBL', "DFR'", "DBL'", "DFR'", "DFL'"))

        self._add_myperm2("Skewb-Center3Cycle-A", ("UBR","ULB'","UBR'","ULB","UFL","URF'","UFL'","URF"))
        self._add_myperm2("Skewb-Center3Cycle-B", ("UBR'","ULB'","UBR'","ULB","UFL","URF'","UFL'","URF","UBR'"))

        self._add_myperm2("Skewb-Corner3", ("DRB'", "DBL'", "ULB'", 'UFL', "URF'", "ULB'", "UBR'", 'ULB', "UBR'", "ULB'", "UFL'", 'ULB'))

        self._add_myperm2("Skewb-Corner4-A", ('DRB', 'UBR', "ULB'", 'DRB', "UBR'", "ULB'", "UBR'", "ULB'", 'UBR'))
        self._add_myperm2("Skewb-Corner4-B", ("DRB'", 'DBL', "DRB'", "DBL'", 'DFL', "DRB'", "DFR'", 'DRB', 'DFR', "DRB'"))
        self._add_myperm2("Skewb-Corner4-C", ('UBR', "ULB'", 'UBR', "ULB'", "UBR'", 'ULB', "UBR'", 'ULB'))

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
        self.corner_index = []
        for signs in CORNER_SIGNS:
            piece = tuple(
                self.corner_lookup[(face, signs)]
                for face in self.faces
                if signs[FACE_AXIS[face][0]] == FACE_AXIS[face][1]
            )
            self.corner_index.append(piece)
        self.center_index = [(self.center_index_by_face[face],) for face in self.faces]
        self.group_pieces = {
            "Corner": self.corner_index,
            "Center": self.center_index,
        }
        self.group_indices = {
            group_name:list(range(len(pieces)))
            for group_name, pieces in self.group_pieces.items()
        }

    def _init_myperms_index(self):
        self.default_color = {}
        self.num_to_piece = {}
        self.myperms_dict = {}
        self.piece_color_counter = {}
        for pieces in self.group_pieces.values():
            for piece in pieces:
                self.default_color[piece] = "".join(self.state_0[index] for index in piece)
                for sticker_index in piece:
                    self.num_to_piece[sticker_index] = piece
        self._init_empty_myperms_dict()
        self._register_myperms_dict_entries()
        self.myperms_group = self.myperms_dict
        self._init_myperms_order()

    def _init_empty_myperms_dict(self):
        self.myperms_dict = defaultdict(list)
        self.piece_color_counter = defaultdict(int)

    def _register_myperms_dict_entries(self):
        for key, moves in self.myperms.items():
            for move in self.invert_moves(moves):
                self.make_move(move)
            for pieces in self.group_pieces.values():
                for piece in pieces:
                    color = "".join(self.state[index] for index in piece)
                    if color != self.default_color[piece]:
                        dict_key = (piece, color)
                        if dict_key not in self.myperms_dict:
                            self.myperms_dict[dict_key] = []
                            self.piece_color_counter[dict_key] = 0
                        self.myperms_dict[dict_key].append(key)
            for move in moves:
                self.make_move(move)

    def _init_myperms_order(self):
        self.myperms_order = {
            group_name:[index for piece in pieces for index in piece]
            for group_name, pieces in self.group_pieces.items()
        }

    def _group_name_map(self):
        return {chr(ord("A") + index): group_name for index, group_name in enumerate(self.group_pieces)}

    def _init_feature_layout(self):
        self.piece_feature_offsets = {}
        self.feature_index_to_piece_color = {}
        offset = 0
        for group_name in self.group_pieces:
            for piece in self.group_pieces[group_name]:
                feature_size = len(self.colors) ** len(piece)
                self.piece_feature_offsets[piece] = (offset, feature_size)
                for color_index in range(feature_size):
                    color = self._color_from_piece_color_index(color_index, len(piece))
                    self.feature_index_to_piece_color[offset + color_index] = (piece, color)
                offset += feature_size
        self.ips = offset

    def _init_group_values(self):
        self.group_val = {}
        for group_name, pieces in self.group_pieces.items():
            group_vector = np.zeros((1, self.ips), dtype = "f")
            for piece in pieces:
                feature_index = self._piece_feature_index(piece, self._piece_color(self.state_0, piece))
                group_vector[0, feature_index] = 1.0
            self.group_val[group_name] = group_vector
        self.total_val = {key:np.sum(value) for key, value in self.group_val.items()}

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
        offset, _ = self.piece_feature_offsets[piece]
        return offset + self._piece_color_index(color)

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

    def scramble(self, N, Move = None, difficult_mode = False, scramble_mode = None, flip = None, rotate = None, swap = False, add_moves = False, transform_N = None, flip_inside = None, move_count_policy = 'prefer_rare'):
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

    def _guided_scramble(self, level_count, move_count_policy = 'prefer_rare', transform_N = None):
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
        if not changed_piece_keys:
            return 0
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
        mapped_axis = tuple(transformation["matrix"] @ MOVE_AXES[base].astype(int))
        mapped_base = AXIS_TO_MOVE[mapped_axis]
        if transformation["mirror"]:
            suffix = "" if suffix == "'" else "'"
        return self.normalize_move_key(mapped_base + suffix)

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
        labels = "-".join(self.index_to_face[index] for index in piece)
        return f"{piece_type}-{labels}"


Rubiks_3 = SkewbCube
