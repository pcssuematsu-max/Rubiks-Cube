"""Pyraminx and Master Pyraminx state/move utilities."""

import math
import random
from collections import defaultdict
from itertools import permutations
from functools import reduce

import numpy as np

from core.scramble_selector import ScrambleSelector
from core.myperm_effects import rename_myperms_by_effect
from core.myperm_keys import make_myperm_key, normalize_myperm_name, single_move_myperm_name


PYRAMINX_FACES = ("U", "L", "R", "B")
FACE_COLORS = {
    "U": "W",
    "L": "G",
    "R": "R",
    "B": "B",
}
MOVE_VERTICES = {
    "U": "U",
    "L": "L",
    "R": "R",
    "B": "B",
}


class PyraminxCube:
    """Tetrahedral twisty puzzle using the Rubiks/Megaminx cube interface."""

    def __init__(self, S = "", size = 3, F2L = False, OLL = False, Centers = False, Edges = False, Cross = False):
        self.size = 3 if size in (None, 0) else int(size)
        self.order = self.size
        self.myperm_point_puzzle = "masterpyraminx" if self.order >= 4 else "pyraminx"
        self.F2L = F2L
        self.OLL = OLL
        self.Centers = Centers
        self.Edges = Edges
        self.Cross = Cross
        self.faces = PYRAMINX_FACES
        self.colors = list(PYRAMINX_FACES)
        self.color_to_num = {color:index for index, color in enumerate(self.colors)}
        self.surface_num = self.order ** 2
        self.sticker_num = len(self.faces) * self.surface_num
        self.ips = 0

        self.vertices = self._build_vertices()
        self.face_vertices = {
            "U": ("L", "B", "R"),
            "L": ("U", "R", "B"),
            "R": ("U", "B", "L"),
            "B": ("U", "L", "R"),
        }
        self.face_opposite_vertex = {"U":"U", "L":"L", "R":"R", "B":"B"}
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

    def _build_vertices(self):
        points = {
            "U": np.array([1.0, 1.0, 1.0]),
            "L": np.array([1.0, -1.0, -1.0]),
            "R": np.array([-1.0, 1.0, -1.0]),
            "B": np.array([-1.0, -1.0, 1.0]),
        }
        return {key:value / np.linalg.norm(value) for key, value in points.items()}

    def _init_stickers(self):
        self.stickers = []
        self.index_to_face = []
        self.face_indices = {face:[] for face in self.faces}
        self.coord_lookup = {}
        for face in self.faces:
            vertices = self.face_vertices[face]
            for orientation, i, j, center in self._face_triangles(vertices):
                index = len(self.stickers)
                normal = self._face_normal(face)
                self.stickers.append({
                    "face": face,
                    "orientation": orientation,
                    "i": i,
                    "j": j,
                    "center": center,
                    "normal": normal,
                    "bary": self._tetra_barycentric(center),
                })
                self.index_to_face.append(face)
                self.face_indices[face].append(index)
                self.coord_lookup[(face, orientation, i, j)] = index

    def _face_triangles(self, vertices):
        a, b, c = [self.vertices[key] for key in vertices]
        n = self.order
        for i in range(n):
            for j in range(n - i):
                pts = [
                    self._face_point(a, b, c, i, j),
                    self._face_point(a, b, c, i + 1, j),
                    self._face_point(a, b, c, i, j + 1),
                ]
                yield "up", i, j, sum(pts) / 3.0
                if i + j < n - 1:
                    pts = [
                        self._face_point(a, b, c, i + 1, j),
                        self._face_point(a, b, c, i + 1, j + 1),
                        self._face_point(a, b, c, i, j + 1),
                    ]
                    yield "down", i, j, sum(pts) / 3.0

    def _face_point(self, a, b, c, i, j):
        u = i / self.order
        v = j / self.order
        w = 1.0 - u - v
        return w * a + u * b + v * c

    def _face_normal(self, face):
        opposite = self.vertices[self.face_opposite_vertex[face]]
        return -opposite / np.linalg.norm(opposite)

    def _tetra_barycentric(self, point):
        matrix = np.vstack([
            np.column_stack([self.vertices[key] for key in self.faces]),
            np.ones((1, len(self.faces))),
        ])
        bary = np.linalg.solve(matrix, np.append(point, 1.0))
        return {face:float(bary[index]) for index, face in enumerate(self.faces)}

    def _init_move_tables(self):
        self.move = {}
        for vertex in MOVE_VERTICES:
            for depth in self._allowed_depths():
                base = self._move_base_token(vertex, depth)
                clockwise = self._build_move_permutation(vertex, depth, clockwise = True)
                self.move[base] = clockwise
                self.move[base + "'"] = np.argsort(clockwise)

    def _allowed_depths(self):
        depths = [1, 2]
        if self.order >= 4:
            depths.append(3)
        return depths

    def _move_base_token(self, vertex, depth):
        if depth == 1:
            return vertex.lower()
        if depth == 2:
            return vertex
        return str(depth) + vertex

    def _build_move_permutation(self, vertex, depth, clockwise = True):
        perm = np.arange(self.sticker_num)
        selected = self._selected_stickers(vertex, depth)
        angle = (-2.0 if clockwise else 2.0) * math.pi / 3.0
        axis = self.vertices[vertex]
        for source in selected:
            center = self._rotate_vector(self.stickers[source]["center"], axis, angle)
            normal = self._rotate_vector(self.stickers[source]["normal"], axis, angle)
            target = self._nearest_sticker(center, normal)
            perm[target] = source
        return perm

    def _selected_stickers(self, vertex, depth):
        threshold = (self.order - depth) / self.order
        selected = []
        for index, sticker in enumerate(self.stickers):
            if sticker["face"] == vertex:
                continue
            if sticker["bary"][vertex] > threshold + 1.0e-8:
                selected.append(index)
        return selected

    def _rotate_vector(self, vector, axis, angle):
        axis = axis / np.linalg.norm(axis)
        return (
            vector * math.cos(angle)
            + np.cross(axis, vector) * math.sin(angle)
            + axis * np.dot(axis, vector) * (1.0 - math.cos(angle))
        )

    def _nearest_sticker(self, center, normal):
        best_index = None
        best_score = None
        for index, sticker in enumerate(self.stickers):
            score = np.linalg.norm(center - sticker["center"]) + 0.25 * np.linalg.norm(normal - sticker["normal"])
            if best_score is None or score < best_score:
                best_index = index
                best_score = score
        return best_index

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
        faces = tuple(self.faces)
        for perm in permutations(faces):
            face_map = {faces[index]:perm[index] for index in range(len(faces))}
            self.transformation_keys.append({
                "face_map": face_map,
                "mirror": self._permutation_parity(perm) == 1,
            })
        self.tf_invert = {
            index:self._inverse_transformation_index(index)
            for index in range(len(self.transformation_keys))
        }

    def _permutation_parity(self, perm):
        index_by_face = {face:index for index, face in enumerate(self.faces)}
        indices = [index_by_face[face] for face in perm]
        inversions = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if indices[i] > indices[j]:
                    inversions += 1
        return inversions % 2

    def _inverse_transformation_index(self, transform_index):
        face_map = self.transformation_keys[transform_index]["face_map"]
        inverse_map = {value:key for key, value in face_map.items()}
        for index, transformation in enumerate(self.transformation_keys):
            if transformation["face_map"] == inverse_map:
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
        if self.order < 4:
            self._add_myperm2("E3[RB>LU>UR]", ("B'", "L'", 'B', 'L'), add_inverse = False)
            self._add_myperm2("E3[RB>UR>LU]", ("L'", "B'", 'L', 'B'), add_inverse = False)
            self._add_myperm2("E3[UB>RU>LU]", ('R', "L'", "R'", 'L'), add_inverse = False)
            self._add_myperm2("E3[UB>LU>RU]", ("L'", 'R', 'L', "R'"), add_inverse = False)
            self._add_myperm2("E2[UL>LU;UR>RU]~v01", ('R', "L'", "R'", 'L', "B'", 'L', 'B', "L'"), add_inverse = False)
            self._add_myperm2("E2[UL>LU;UR>RU]~v02", ('L', "B'", "L'", 'B', "L'", 'R', 'L', "R'"), add_inverse = False)
            self._add_myperm2("E2[LB>BL;UR>RU]~v01", ("B'", "U'", 'B', "U'", 'L', 'U', "L'", 'U'), add_inverse = False)
            self._add_myperm2("E2[LB>BL;UR>RU]~v02", ("U'", 'L', "U'", "L'", 'U', "B'", 'U', 'B'), add_inverse = False)
            self._add_myperm2("E3[LR>UL>RU]", ("L'", 'B', 'L', 'B', "L'", 'B', 'L'), add_inverse = False)
            self._add_myperm2("E3[LR>RU>UL]", ("L'", "B'", 'L', "B'", "L'", "B'", 'L'), add_inverse = False)
            self._add_myperm2("E3[LR>UR>LU]", ("U'", "L'", "B'", 'L', 'B', 'U'), add_inverse = False)
            self._add_myperm2("E3[LR>LU>UR]", ("U'", "B'", "L'", 'B', 'L', 'U'), add_inverse = False)
            self._add_myperm2("E3[UB>UL>UR]", ('R', "L'", "R'", "L'", "B'", "L'", 'B'), add_inverse = False)
            self._add_myperm2("E3[UB>UR>UL]", ("B'", 'L', 'B', 'L', 'R', 'L', "R'"), add_inverse = False)
            self._add_myperm2("E3[RB>RU>LU]", ("B'", 'L', "U'", "L'", 'U', 'B'), add_inverse = False)
            self._add_myperm2("E3[RB>LU>RU]", ("B'", "U'", 'L', 'U', "L'", 'B'), add_inverse = False)
            self._add_myperm2("E3[RB>UL>UR]", ("L'", 'R', "L'", "R'", "L'"), add_inverse = False)
            self._add_myperm2("E3[RB>UR>UL]", ('L', 'R', 'L', "R'", 'L'), add_inverse = False)
            self._add_myperm2("Ctr3[B@U>R@U>L@U]", ('U', 'R', 'U', "R'", 'U', 'R', 'U', "R'", "u'"), add_inverse = False)
            self._add_myperm2("Ctr3[B@U>L@U>R@U]", ('u', 'R', "U'", "R'", "U'", 'R', "U'", "R'", "U'"), add_inverse = False)

        if self.order >= 4:
            self._add_myperm2("OE3[RB@L>LU@R;UL@R>RU@L;UR@L>RB@L]", ("3B'", "L'", '3B', 'L'), add_inverse = False)
            self._add_myperm2("OE3[RB@L>UR@L;UL@R>BR@L;UR@L>LU@R]", ("L'", "3B'", 'L', '3B'), add_inverse = False)
            self._add_myperm2("OE3[UB@L>RU@L;UL@B>BU@L;UR@L>UL@B]", ('3R', "L'", "3R'", 'L'), add_inverse = False)
            self._add_myperm2("OE3[UB@L>LU@B;UL@B>UR@L;UR@L>BU@L]", ("L'", '3R', 'L', "3R'"), add_inverse = False)
            self._add_myperm2("OE3[RB@U>LU@B;UL@B>RU@B;UR@B>RB@U]", ("B'", "3L'", 'B', '3L'), add_inverse = False)
            self._add_myperm2("OE3[RB@U>UR@B;UL@B>BR@U;UR@B>LU@B]", ("3L'", "B'", '3L', 'B'), add_inverse = False)
            self._add_myperm2("OE3[UB@R>RU@B;UL@R>BU@R;UR@B>UL@R]", ('R', "3L'", "R'", '3L'), add_inverse = False)
            self._add_myperm2("OE3[UB@R>LU@R;UL@R>UR@B;UR@B>BU@R]", ("3L'", 'R', '3L', "R'"), add_inverse = False)
            self._add_myperm2("OE3[UL@B>LU@R;UL@R>RU@L;UR@L>UL@B]", ('3R', "L'", "3R'", 'L', "3B'", 'L', '3B', "L'"), add_inverse = False)
            self._add_myperm2("OE3[UL@B>UR@L;UL@R>LU@B;UR@L>LU@R]", ('L', "3B'", "L'", '3B', "L'", '3R', 'L', "3R'"), add_inverse = False)
            self._add_myperm2("OE3[UL@B>RU@B;UL@R>LU@B;UR@B>UL@R]", ('R', "B'", '3L', 'B', "3L'", "R'"), add_inverse = False)
            self._add_myperm2("OE3[UL@B>LU@R;UL@R>UR@B;UR@B>LU@B]", ('R', '3L', "B'", "3L'", 'B', "R'"), add_inverse = False)
            self._add_myperm2("OE3[LB@U>UR@L;UR@B>BL@U;UR@L>RU@B]", ("L'", '3R', 'B', '3R', "B'", '3R', 'L'), add_inverse = False)
            self._add_myperm2("OE3[LB@U>RU@B;UR@B>RU@L;UR@L>LB@U]", ("L'", "3R'", 'B', "3R'", "B'", "3R'", 'L'), add_inverse = False)
            self._add_myperm2("OE3[LB@R>UR@B;UR@B>RU@L;UR@L>BL@R]", ("B'", "3U'", 'B', "3U'", 'L', '3U', "L'", '3U'), add_inverse = False)
            self._add_myperm2("OE3[LB@R>RU@L;UR@B>LB@R;UR@L>RU@B]", ("3U'", 'L', "3U'", "L'", '3U', "B'", '3U', 'B'), add_inverse = False)
            self._add_myperm2("OE3[RB@L>UR@L>UL@B]", ('L', '3R', 'L', "3R'", 'L'), add_inverse = False)
            self._add_myperm2("OE3[RB@L>UL@B>UR@L]", ("L'", '3R', "L'", "3R'", "L'"), add_inverse = False)
            self._add_myperm2("OE3[RB@U>UR@B>UL@R]", ('3L', 'R', '3L', "R'", '3L'), add_inverse = False)
            self._add_myperm2("OE3[RB@U>UL@R>UR@B]", ("3L'", 'R', "3L'", "R'", "3L'"), add_inverse = False)
            self._add_myperm2("OE3[LR@B>UR@L>UL@B]", ('L', 'B', '3U', "B'", "3U'", "L'"), add_inverse = False)
            self._add_myperm2("OE3[LR@B>UL@B>UR@L]", ('L', '3U', 'B', "3U'", "B'", "L'"), add_inverse = False)
            self._add_myperm2("OE3[LB@U>RU@B;UL@R>BL@U;UR@B>UL@R]", ('3B', 'U', "3R'", "U'", '3R', "3B'"), add_inverse = False)
            self._add_myperm2("OE3[LB@U>LU@R;UL@R>UR@B;UR@B>BL@U]", ('3B', "3R'", 'U', '3R', "U'", "3B'"), add_inverse = False)
            self._add_myperm2("OE3[LR@U>RU@L;UL@B>RL@U;UR@L>UL@B]", ("3R'", "U'", "3B'", 'U', '3B', '3R'), add_inverse = False)
            self._add_myperm2("OE3[LR@U>LU@B;UL@B>UR@L;UR@L>RL@U]", ("3R'", "3B'", "U'", '3B', 'U', '3R'), add_inverse = False)
            self._add_myperm2("OE3[UB@R>UR@L>UL@B]", ("3R'", '3B', "U'", '3B', 'U', '3B', '3R'), add_inverse = False)
            self._add_myperm2("OE3[UB@R>UL@B>UR@L]", ("3R'", "3B'", "U'", "3B'", 'U', "3B'", '3R'), add_inverse = False)
            self._add_myperm2("OE3[LR@B>UL@B;UL@B>RU@B;UR@B>RL@B]", ("3L'", 'B', '3L', 'B', "3L'", 'B', '3L'), add_inverse = False)
            self._add_myperm2("OE3[LR@B>RU@B;UL@B>LR@B;UR@B>LU@B]", ("3L'", "B'", '3L', "B'", "3L'", "B'", '3L'), add_inverse = False)
            self._add_myperm2("OE3[LR@U>UL@R;UL@R>RU@L;UR@L>RL@U]", ("L'", '3B', 'L', '3B', "L'", '3B', 'L'), add_inverse = False)
            self._add_myperm2("OE3[LR@U>RU@L;UL@R>LR@U;UR@L>LU@R]", ("L'", "3B'", 'L', "3B'", "L'", "3B'", 'L'), add_inverse = False)
            self._add_myperm2("ME3[UB>UR>UL]", ('3R', "3L'", "L'", '3U', "3L'", 'L', "3U'", 'U', '3L', "L'", '3U', 'U', "3L'", 'L', "3U'", 'U', "3L'", "3R'"), add_inverse = False)
            self._add_myperm2("ME3[UB>UL>UR]", ('3R', '3L', "U'", '3U', "L'", '3L', "U'", "3U'", 'L', "3L'", "U'", '3U', "L'", '3L', "3U'", 'L', '3L', "3R'"), add_inverse = False)
            self._add_myperm2("ME3[LR>RU>UL]", ("B'", "3R'", '3B', "3R'", '3B', 'R', '3B', "3R'", '3B', "3R'"), add_inverse = False)
            self._add_myperm2("ME3[LR>UL>RU]", ('3R', "3B'", '3R', "3B'", "R'", "3B'", '3R', "3B'", '3R', 'B'), add_inverse = False)
            self._add_myperm2("ME3[LB>UR>UL]", ("3R'", '3L', "3R'", "L'", "3R'", '3L', "3R'", '3L', 'R', '3L'), add_inverse = False)
            self._add_myperm2("ME3[LB>UL>UR]", ("3L'", "R'", "3L'", '3R', "3L'", '3R', 'L', '3R', "3L'", '3R'), add_inverse = False)
            self._add_myperm2("ME3[UB>RU>LU]", ("3R'", "3L'", "R'", '3R', "3L'", 'R', "3R'", "3L'", "3R'", '3L', 'R', "3R'", "3L'", "R'", "L'", '3R', 'L', "3R'"), add_inverse = False)
            self._add_myperm2("ME3[UB>LU>RU]", ('3R', "L'", "3R'", 'L', 'R', '3L', '3R', "R'", "3L'", '3R', '3L', '3R', "R'", '3L', "3R'", 'R', '3L', '3R'), add_inverse = False)
            self._add_myperm2("ME3[LB>RU>UL]", ("3R'", "3B'", "3R'", 'R', '3B', "3R'", "B'", "3R'", 'U', '3B', "U'", '3B', 'U', '3B', "U'", '3B'), add_inverse = False)
            self._add_myperm2("ME3[LB>UL>RU]", ("3B'", 'U', "3B'", "U'", "3B'", 'U', "3B'", "U'", '3R', 'B', '3R', "3B'", "R'", '3R', '3B', '3R'), add_inverse = False)
            self._add_myperm2("ME3[RB>LU>RU]", ("3U'", '3B', "3U'", "B'", "3U'", '3B', "3U'", '3B', 'U', '3B', "3L'", "R'", "3L'", '3R', "3L'", '3R', 'L', '3R', "3L'", '3R'), add_inverse = False)
            self._add_myperm2("ME3[RB>RU>LU]", ("3R'", '3L', "3R'", "L'", "3R'", '3L', "3R'", '3L', 'R', '3L', "3B'", "U'", "3B'", '3U', "3B'", '3U', 'B', '3U', "3B'", '3U'), add_inverse = False)
            self._add_myperm2("ME3[LR>LU>RU]", ("3R'", '3L', 'U', '3L', "3B'", "R'", "3U'", "3L'", "3R'", "3B'", 'U', "3L'", "3B'", '3U', '3B', '3U', "U'", "3B'", '3U', "3B'", 'U', '3B', "U'"), add_inverse = False)
            self._add_myperm2("ME3[LR>RU>LU]", ('U', "3B'", "U'", '3B', "3U'", '3B', 'U', "3U'", "3B'", "3U'", '3B', '3L', "U'", '3B', '3R', '3L', '3U', 'R', '3B', "3L'", "U'", "3L'", '3R'), add_inverse = False)
            self._add_myperm2("ME2[UL>LU;UR>RU]~v01", ('3R', "3U'", "L'", "3U'", '3B', 'R', '3L', '3U', '3R', "L'", '3B', '3U', '3L', 'B'), add_inverse = False)
            self._add_myperm2("ME2[UL>LU;UR>RU]~v02", ("B'", "3L'", "3U'", "3B'", 'L', "3R'", "3U'", "3L'", "R'", "3B'", '3U', 'L', '3U', "3R'"), add_inverse = False)
            self._add_myperm2("ME2[RB>BR;UL>LU]~v01", ('3R', '3B', 'L', '3B', "3U'", "R'", "3L'", "3B'", "3R'", 'L', "3U'", "3B'", "3L'", "U'", '3R'), add_inverse = False)
            self._add_myperm2("ME2[RB>BR;UL>LU]~v02", ("3R'", 'U', '3L', '3B', '3U', "L'", '3R', '3B', '3L', 'R', '3U', "3B'", "L'", "3B'", "3R'"), add_inverse = False)
            self._add_myperm2("Ctr3[B@U>R@U>L@U]", ('U', '3R', 'U', "3R'", 'U', '3R', 'U', "3R'", "u'"), add_inverse = False)
            self._add_myperm2("Ctr3[B@U>L@U>R@U]", ('u', '3R', "U'", "3R'", "U'", '3R', "U'", "3R'", "U'"), add_inverse = False)
            self._add_myperm2("Ctr3[B@C>L@C>R@C]", ("R'", "3U'", "3R'", 'R', '3U', "3R'", "3U'", "3R'", 'R', '3U', "3R'"), add_inverse = False)
            self._add_myperm2("Ctr3[B@C>R@C>L@C]", ('3R', "3U'", "R'", '3R', '3U', '3R', "3U'", "R'", '3R', '3U', 'R'), add_inverse = False)

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
        self.edge_index = []
        self.mid_edge_index = []
        self.center_index = []
        self.corner_index = self._piece_indices_at_vertices()
        self.edge_index = self._piece_indices_at_edges()
        self.mid_edge_index = self._piece_indices_at_mid_edges()
        used = {
            idx
            for piece in self.corner_index + self.edge_index + self.mid_edge_index
            for idx in piece
        }
        self.center_index = [(idx,) for idx in range(self.sticker_num) if idx not in used]
        self.group_pieces = {
            "Corner": self.corner_index,
            "Edge": self.edge_index,
        }
        if self.mid_edge_index:
            self.group_pieces["MidEdge"] = self.mid_edge_index
        self.group_pieces["Center"] = self.center_index
        self.group_indices = {
            group_name: list(range(len(pieces)))
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
            group_name: [index for piece in pieces for index in piece]
            for group_name, pieces in self.group_pieces.items()
        }

    def _group_name_map(self):
        return {chr(ord("A") + index): group_name for index, group_name in enumerate(self.group_pieces)}

    def _init_group_values(self):
        self.group_val = {}
        for group_name, pieces in self.group_pieces.items():
            group_vector = np.zeros((1, self.ips), dtype = "f")
            for piece in pieces:
                feature_index = self._piece_feature_index(piece, self._piece_color(self.state_0, piece))
                group_vector[0, feature_index] = 1.0
            self.group_val[group_name] = group_vector
        self.total_val = {key:np.sum(value) for key, value in self.group_val.items()}

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

    def _piece_indices_at_vertices(self):
        pieces = []
        for vertex in self.faces:
            indices = [
                index for index, sticker in enumerate(self.stickers)
                if sticker["face"] != vertex and sticker["bary"][vertex] > (self.order - 1) / self.order + 1.0e-8
            ]
            if indices:
                pieces.append(tuple(indices))
        return pieces

    def _piece_indices_at_edges(self):
        pieces = []
        for a_index, a in enumerate(self.faces):
            for b in self.faces[a_index + 1:]:
                adjacent_faces = [face for face in self.faces if face not in (a, b)]
                for edge_position in range(1, self.order - 1):
                    target_bary = {
                        a: (self.order - 1 - edge_position) / (self.order - 1),
                        b: edge_position / (self.order - 1),
                    }
                    edge_piece = tuple(
                        self._nearest_edge_sticker(face, target_bary)
                        for face in adjacent_faces
                    )
                    pieces.append(edge_piece)
        return pieces

    def _piece_indices_at_mid_edges(self):
        if self.order < 4:
            return []

        pieces = []
        reserved = {idx for piece in self.corner_index + self.edge_index for idx in piece}
        used_mid_edges = set()
        for a_index, a in enumerate(self.faces):
            for b in self.faces[a_index + 1:]:
                adjacent_faces = [face for face in self.faces if face not in (a, b)]
                target_bary = {a: 0.5, b: 0.5}
                piece = tuple(
                    self._nearest_mid_edge_sticker(
                        face,
                        target_bary,
                        reserved | used_mid_edges,
                    )
                    for face in adjacent_faces
                )
                used_mid_edges.update(piece)
                pieces.append(piece)
        return pieces

    def _nearest_mid_edge_sticker(self, face, target_bary, excluded):
        candidates = [
            index for index in self.face_indices[face]
            if index not in excluded and self.stickers[index]["orientation"] == "down"
        ]
        if not candidates:
            candidates = [
                index for index in self.face_indices[face]
                if index not in excluded
            ]
        return min(
            candidates,
            key = lambda index: self._edge_sticker_distance(self.stickers[index], target_bary),
        )

    def _nearest_edge_sticker(self, face, target_bary):
        candidates = self.face_indices[face]
        return min(
            candidates,
            key = lambda index: self._edge_sticker_distance(self.stickers[index], target_bary),
        )

    def _edge_sticker_distance(self, sticker, target_bary):
        distance = 0.0
        for face in self.faces:
            target = target_bary.get(face, 0.0)
            distance += (sticker["bary"][face] - target) ** 2
        return distance

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

    def scramble(self, N, Move = None, difficult_mode = False, scramble_mode = None, flip = None, rotate = None, swap = False, add_moves = None, transform_N = None, flip_inside = None, move_count_policy = 'prefer_rare'):
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

    def _select_scramble_candidate(self, candidates, move_count, move_count_policy, level):
        if move_count_policy == 'prefer_frequent':
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
        prefix, face, suffix = self._parse_move_token(move)
        mapped_face = transformation["face_map"][face.upper()]
        if face.islower():
            mapped_face = mapped_face.lower()
        if transformation["mirror"]:
            suffix = "" if suffix == "'" else "'"
        transformed_move = prefix + mapped_face + suffix
        return self.normalize_move_key(transformed_move)

    def _parse_move_token(self, move):
        suffix = "'" if move.endswith("'") else ""
        core = move[:-1] if suffix else move
        if core.startswith("3"):
            return "3", core[1], suffix
        return "", core[0], suffix

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


class MasterPyraminxCube(PyraminxCube):
    def __init__(self, S = "", size = 4, F2L = False, OLL = False, Centers = False, Edges = False, Cross = False):
        super().__init__(S = S, size = 4 if size in (None, 0, 3) else size, F2L = F2L, OLL = OLL, Centers = Centers, Edges = Edges, Cross = Cross)
