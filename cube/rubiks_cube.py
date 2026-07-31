"""Rubiks cube model and move/state utilities."""

import random
from functools import reduce
from pathlib import Path

import numpy as np

from core.cube_constants import AB, R_Nums
from core.myperm_keys import (
    format_myperm_key,
    make_myperm_key,
    myperm_base_key,
    myperm_transform_index,
    normalize_myperm_registry,
    single_move_myperm_name,
)
from core.myperm_effects import rename_myperms_by_effect
from core.myperm_points import load_myperm_points, reindex_myperms_by_points
from core.scramble_selector import ScrambleSelector
from cube.move_sequence_ops import MoveSequenceOps


RUBIKS_MOVE_FACE_LABELS_BY_INDEX = ('U', 'D', 'F', 'B', 'L', 'R')
RUBIKS_SOLVED_COLORS_BY_FACE_INDEX = ('R', 'O', 'Y', 'W', 'G', 'B')
RUBIKS_COLOR_NAMES = {
    'R': 'Red',
    'O': 'Orange',
    'Y': 'Yellow',
    'W': 'White',
    'G': 'Green',
    'B': 'Blue',
    'X': 'Masked',
}
RUBIKS_AXIS_INFO = {
    'U': {'horizontal': ('R', 'L', True),  'vertical': ('F', 'B', True)},
    'D': {'horizontal': ('R', 'L', False), 'vertical': ('F', 'B', True)},
    'F': {'horizontal': ('R', 'L', True),  'vertical': ('U', 'D', False)},
    'B': {'horizontal': ('L', 'R', False), 'vertical': ('U', 'D', True)},
    'R': {'horizontal': ('B', 'F', True),  'vertical': ('U', 'D', False)},
    'L': {'horizontal': ('F', 'B', True),  'vertical': ('U', 'D', False)},
}
RUBIKS_MIDDLE_AXIS_LABEL = {
    frozenset({'R', 'L'}): 'M',
    frozenset({'F', 'B'}): 'S',
    frozenset({'U', 'D'}): 'E',
}
RUBIKS_AXIS_FAMILY = {
    'R': 'RL',
    'L': 'RL',
    'M': 'RL',
    'F': 'FB',
    'B': 'FB',
    'S': 'FB',
    'U': 'UD',
    'D': 'UD',
    'E': 'UD',
}


def _build_group_indices_by_size():
    """cube size ごとの group index 定義を返す。"""
    return {
        2: {'A':list(range(4)),'B':[],'C':[],'c':[],'D':[],'d':[],'E':[],'e':[],'F':[],'f':[],'G':[]},
        3: {'A':list(range(4)),'B':list(range(4,8)),'C':[],'c':[],'D':[],'d':[],'E':[],'e':[],'F':[],'f':[],'G':[8]},
        4: {'A':list(range(4)),'B':[],'C':list(range(4,12)),'c':[],'D':list(range(12,16)),'d':[],'E':[],'e':[],'F':[],'f':[],'G':[]},
        5: {'A':list(range(4)),'B':list(range(4,8)),'C':list(range(8,16)),'c':[],'D':list(range(16,20)),'d':[],'E':list(range(20,24)),'e':[],'F':[],'f':[],'G':[24]},
        6: {'A':list(range(4)),'B':[],'C':[4,5,6,7,8,9,10,11],'c':[12,13,14,15,16,17,18,19],'D':[20,21,22,23],'d':[32,33,34,35],'E':[],'e':[],'F':[24,25,26,27],'f':[28,29,30,31],'G':[]},
        7: {'A':list(range(4)),'B':list(range(4,8)),'C':[8,9,10,11,12,13,14,15],'c':[16,17,18,19,20,21,22,23],'D':[24,25,26,27],'d':[40,41,42,43],'E':[28,29,30,31],'e':[44,45,46,47],'F':[32,33,34,35],'f':[36,37,38,39],'G':[48]},
    }


class Rubiks_3:
    def __init__(self,S = '',size = 3,F2L = False,OLL = False,Centers = False,Edges = False,Cross = False,PointReindex = False,RegisterMyperms = True):        
        
        self.size = size
        self.F2L = F2L and (size == 3)
        self.OLL = OLL and (size == 3)
        self.Centers = Centers
        self.Edges = Edges
        self.Cross = Cross
        self.PointReindex = PointReindex
        self.RegisterMyperms = RegisterMyperms
        if self.PointReindex and not self.RegisterMyperms:
            raise ValueError("PointReindex requires RegisterMyperms = True")
        if self.F2L:
            self.colors = ['X','O','Y','W','G','B']
        elif self.Centers:
            self.colors = ['R','O','Y','W','G','B']
        else:
            self.colors = ['R','O','Y','W','G','B']
        
        self.move = {}
        self._init_move_keys()
        self._init_move_symbol_tables()
        self._init_symmetry_tables()
        self._init_transformation_tables()

        self.move_ops = MoveSequenceOps(self)
        
        self._init_myperm_containers()
        if self.RegisterMyperms:
            self._register_myperms2()
            self._expand_registered_myperms()

        self._init_cube_state_and_moves()
        self._init_color_keys_and_groups()
        if self.PointReindex:
            point_reindex_names = None if self.PointReindex is True else tuple(self.PointReindex)
            self._reindex_myperms_by_points(names = point_reindex_names)
        if self.RegisterMyperms:
            rename_myperms_by_effect(self)
        self._init_myperms_index()
        self._init_single_move_and_rotate()
        self.scramble_selector = ScrambleSelector(self)


    def _init_move_symbol_tables(self):
        """手の反対面・逆回転・合成結果などの基本表を初期化する。"""
        self.opposite = {"U":"D","D":"U","F":"B","B":"F","L":"R","R":"L","M":"M","S":"S","E":"E","x":"x","y":"y","z":"z"}
        self.inverse = {" ":"'","'":" ","2":"2"}
        self.mult = {(" "," "):"2",(" ","2"):"'",(" ","'"):0,
                     ("2"," "):"'",("2","2"):0,("2","'"):" ",
                     ("'"," "):0,("'","2"):" ",("'","'"):"2"}
        self.axis = {"L":"x","R":"x","M":"x","U":"y","D":"y","E":"y","F":"z","B":"z","S":"z"}

    def _init_symmetry_tables(self):
        """鏡映・回転・対角反転の move table を初期化する。"""
        self.flip = {}
        self.flip['UD'] = {"U ":"D'","D ":"U'","F ":"F'","B ":"B'","L ":"L'","R ":"R'",
                           "U'":"D ","D'":"U ","F'":"F ","B'":"B ","L'":"L ","R'":"R ",
                           "M ":"M'","S ":"S'","E ":"E ","M'":"M ","S'":"S ","E'":"E'",
                           "U2":"D2","D2":"U2","F2":"F2","B2":"B2","L2":"L2","R2":"R2",
                           "M2":"M2","S2":"S2","E2":"E2",
                           "x ":"x'","y ":"y ","z ":"z'","x'":"x ","y'":"y'","z'":"z ",
                           "x2":"x2","y2":"y2","z2":"z2"}
        
        self.flip['FB'] = {"U ":"U'","D ":"D'","F ":"B'","B ":"F'","L ":"L'","R ":"R'",
                           "U'":"U ","D'":"D ","F'":"B ","B'":"F ","L'":"L ","R'":"R ",
                           "M ":"M'","S ":"S ","E ":"E'","M'":"M ","S'":"S'","E'":"E ",
                           "U2":"U2","D2":"D2","F2":"B2","B2":"F2","L2":"L2","R2":"R2",
                           "M2":"M2","S2":"S2","E2":"E2",
                           "x ":"x'","y ":"y'","z ":"z ","x'":"x ","y'":"y ","z'":"z'",
                           "x2":"x2","y2":"y2","z2":"z2"}

        self.flip['LR'] = {"U ":"U'","D ":"D'","F ":"F'","B ":"B'","L ":"R'","R ":"L'",
                           "U'":"U ","D'":"D ","F'":"F ","B'":"B ","L'":"R ","R'":"L ",
                           "M ":"M ","S ":"S'","E ":"E'","M'":"M'","S'":"S ","E'":"E ",
                           "U2":"U2","D2":"D2","F2":"F2","B2":"B2","L2":"R2","R2":"L2",
                           "M2":"M2","S2":"S2","E2":"E2",
                           "x ":"x ","y ":"y'","z ":"z'","x'":"x'","y'":"y ","z'":"z ",
                           "x2":"x2","y2":"y2","z2":"z2"}

        self.rotate = {}
        self.rotate['UD'] = {"U ":"U ","D ":"D ","F ":"L ","B ":"R ","L ":"B ","R ":"F ",
                             "U'":"U'","D'":"D'","F'":"L'","B'":"R'","L'":"B'","R'":"F'",
                             "M ":"S'","S ":"M ","E ":"E ","M'":"S ","S'":"M'","E'":"E'",
                             "U2":"U2","D2":"D2","F2":"L2","B2":"R2","L2":"B2","R2":"F2",
                             "M2":"S2","S2":"M2","E2":"E2",
                             "x ":"z ","y ":"y ","z ":"x'","x'":"z'","y'":"y'","z'":"x ",
                             "x2":"z2","y2":"y2","z2":"x2"}

        self.rotate['FB']= {"U ":"R ","D ":"L ","F ":"F ","B ":"B ","L ":"U ","R ":"D ",
                            "U'":"R'","D'":"L'","F'":"F'","B'":"B'","L'":"U'","R'":"D'",
                            "M ":"E'","S ":"S ","E ":"M ","M'":"E ","S'":"S'","E'":"M'",
                            "U2":"R2","D2":"L2","F2":"F2","B2":"B2","L2":"U2","R2":"D2",
                            "M2":"E2","S2":"S2","E2":"M2",
                            "x ":"y'","y ":"x ","z ":"z ","x'":"y ","y'":"x'","z'":"z'",
                            "x2":"y2","y2":"x2","z2":"z2"}

        self.rotate['LR'] = {"U ":"B ","D ":"F ","F ":"U ","B ":"D ","L ":"L ","R ":"R ",
                             "U'":"B'","D'":"F'","F'":"U'","B'":"D'","L'":"L'","R'":"R'",
                             "M ":"M ","S ":"E'","E ":"S ","M'":"M'","S'":"E ","E'":"S'",
                             "U2":"B2","D2":"F2","F2":"U2","B2":"D2","L2":"L2","R2":"R2",
                             "M2":"M2","S2":"E2","E2":"S2",
                             "x ":"x ","y ":"z'","z ":"y ","x'":"x'","y'":"z ","z'":"y'",
                             "x2":"x2","y2":"z2","z2":"y2"}

        self.rotate['RL'] = {self.rotate['LR'][k]:k for k in self.rotate['LR']}

        self.rotate['120'] = {"U ":"R ","D ":"L ","F ":"U ","B ":"D ","L ":"B ","R ":"F ",
                              "U'":"R'","D'":"L'","F'":"U'","B'":"D'","L'":"B'","R'":"F'",
                              "M ":"S'","S ":"E'","E ":"M ","M'":"S ","S'":"E ","E'":"M'",
                              "U2":"R2","D2":"L2","F2":"U2","B2":"D2","L2":"B2","R2":"F2",
                              "M2":"S2","S2":"E2","E2":"M2",
                              "x ":"z ","y ":"x ","z ":"y ","x'":"z'","y'":"x'","z'":"y'",
                              "x2":"z2","y2":"x2","z2":"y2"}

        self.rotate['240'] = {"U ":"F ","D ":"B ","F ":"R ","B ":"L ","L ":"D ","R ":"U ",
                              "U'":"F'","D'":"B'","F'":"R'","B'":"L'","L'":"D'","R'":"U'",
                              "M ":"E ","S ":"M'","E ":"S'","M'":"E'","S'":"M ","E'":"S ",
                              "U2":"F2","D2":"B2","F2":"R2","B2":"L2","L2":"D2","R2":"U2",
                              "M2":"E2","S2":"M2","E2":"S2",
                              "x ":"y ","z ":"x ","y ":"z ","x'":"y'","z'":"x'","y'":"z'",
                              "x2":"y2","z2":"x2","y2":"z2"}

        self.diag_flip = {"U ":"U'","D ":"D'","F ":"R'","B ":"L'","L ":"B'","R ":"F'",
                          "U'":"U ","D'":"D ","F'":"R ","B'":"L ","L'":"B ","R'":"F ",
                          "M ":"S ","S ":"M ","E ":"E'","M'":"S'","S'":"M'","E'":"E ",
                          "U2":"U2","D2":"D2","F2":"R2","B2":"L2","L2":"B2","R2":"F2",
                          "M2":"S2","S2":"M2","E2":"E2",
                          "x ":"z'","z ":"x'","y ":"y'","x'":"z ","z'":"x ","y'":"y ",
                          "x2":"z2","z2":"x2","y2":"y2"}

    def _init_transformation_tables(self):
        """対称変換の列挙と逆変換表を初期化する。"""
        self.transformation_keys = [(),("UD","FB","LR"),("UD","LR"),("FB",),("FB","LR"),("UD",),("UD","FB"),("LR",),
                                    ('120',),("UD","FB","LR",'120'),("UD","LR",'120'),("FB",'120'),("FB","LR",'120'),("UD",'120'),("UD","FB",'120'),("LR",'120'),
                                    ('240',),("UD","FB","LR",'240'),("UD","LR",'240'),("FB",'240'),("FB","LR",'240'),("UD",'240'),("UD","FB",'240'),("LR",'240'),
                                    ("XX",),("UD","FB","LR","XX"),("UD","LR","XX"),("FB","XX"),("FB","LR","XX"),("UD","XX"),("UD","FB","XX"),("LR","XX"),
                                    ('120',"XX"),("UD","FB","LR",'120',"XX"),("UD","LR",'120',"XX"),("FB",'120',"XX"),("FB","LR",'120',"XX"),("UD",'120',"XX"),("UD","FB",'120',"XX"),("LR",'120',"XX"),
                                    ('240',"XX"),("UD","FB","LR",'240',"XX"),("UD","LR",'240',"XX"),("FB",'240',"XX"),("FB","LR",'240',"XX"),("UD",'240',"XX"),("UD","FB",'240',"XX"),("LR",'240',"XX"),
                                    ]

        if self.size >= 6:
            self.transformation_keys = [x + y for y in [(),('S',)] for x in self.transformation_keys]

        self.tf_invert = {"UD":"UD","FB":"FB","LR":"LR","120":"240","240":"120","XX":"XX","S":"S"}

    def _init_myperm_containers(self):
        """myperm登録用の辞書とグループ情報を初期化する。"""
        self.myperms = {}
        self._add_single_moves_to_myperms()
        self.myperms2 = {}
        self._init_group_indices()

    def _add_myperm2(self, name, moves):
        """Register one source myperm by its canonical source name."""
        self.myperms2[name] = moves
        return name

    def _moves_available_for_size(self, moves):
        """Drop inner-layer moves that do not exist for the current cube size."""
        return self.simplify(
            tuple(
                move
                for move in moves
                if self._move_available_for_size(move)
            )
        )

    def _move_available_for_size(self, move):
        token = str(move).strip()
        layer_digits = []
        for character in token:
            if not character.isdigit():
                break
            layer_digits.append(character)
        if not layer_digits:
            return True
        layer = int("".join(layer_digits))
        face_index = len(layer_digits)
        if face_index < len(token) and token[face_index] in "UDFBLR":
            return layer <= self.size // 2
        return True

    def _add_single_moves_to_myperms(self):
        for m in self.move_keys:
            self.myperms[make_myperm_key(single_move_myperm_name(m), 0)] = (m,)

        self.myperms[make_myperm_key('Rotate6A-00', 0)] = (" x "," z ")
        self.myperms[make_myperm_key('Rotate6A-01', 0)] = (" x "," z'")
        self.myperms[make_myperm_key('Rotate6A-02', 0)] = (" x'"," z ")
        self.myperms[make_myperm_key('Rotate6A-03', 0)] = (" x'"," z'")
        self.myperms[make_myperm_key('Rotate6A-04', 0)] = (" z "," x ")
        self.myperms[make_myperm_key('Rotate6A-05', 0)] = (" z "," x'")
        self.myperms[make_myperm_key('Rotate6A-06', 0)] = (" z'"," x ")
        self.myperms[make_myperm_key('Rotate6A-07', 0)] = (" z'"," x'")

        self.myperms[make_myperm_key('Rotate6B-00', 0)] = (" y "," x2")
        self.myperms[make_myperm_key('Rotate6B-01', 0)] = (" y "," z2")
        self.myperms[make_myperm_key('Rotate6B-02', 0)] = (" x "," y2")
        self.myperms[make_myperm_key('Rotate6B-03', 0)] = (" x'"," y2")
        self.myperms[make_myperm_key('Rotate6B-04', 0)] = (" z "," y2")
        self.myperms[make_myperm_key('Rotate6B-05', 0)] = (" z'"," y2")

    def _init_group_indices(self):
        """group index 定義を読み込み、意味名ベースの受け口を作る。"""
        short_group_indices = _build_group_indices_by_size()[self.size]
        group_names = self._group_name_map()
        self.group_indices = {}
        for short_key, indices in short_group_indices.items():
            index_list = list(indices)
            self.group_indices[short_key] = index_list
            self.group_indices[group_names[short_key]] = index_list


    def _register_myperms2(self):
        """myperms2へ固定手順と派生手順を登録する。"""
        self._register_myperms2_base()
        self._register_myperms2_x_perms()
        self._register_myperms2_odd_size()
        if hasattr(self, '_parity_swap_basis_moves'):
            del self._parity_swap_basis_moves
        self._register_myperms2_general()
        self._register_myperms2_f2l_oll()

    def _register_myperms2_base(self):
        """基本パターンと大分類の手順を登録する。"""
        # 命名メモ:
        # - X-Center / Plus-Center / Oblique-Center は動かす center の配置族。
        # - 4 / 6 は見た目上で動く center 数、末尾の英字は variant を表す。

        self._add_myperm2('EAll12s', (' U2', ' D2', ' F2', ' B2', ' R2', ' L2'))
        self._add_myperm2('EAll12[2x6][XY>YX]', (' U ', ' R2', ' F ', ' B ', ' R ', ' B2', ' R ', ' U2', ' L ', ' B2', ' R ', " U'", " D'", ' R2', ' F ', " L'", ' R ', ' U2', ' D2', ' B2', ' D2', ' B2'))
        self._add_myperm2('EAll12[XY>YX]', (' U ', ' R2', ' F ', ' B ', ' R ', ' B2', ' R ', ' U2', ' L ', ' B2', ' R ', " U'", " D'", ' R2', ' F ', " R'", ' L ', ' B2', ' U2', ' F2'))
        self._add_myperm2('C8~v01', (' U2', ' B2', ' D ', ' L2', " F'", " B'", ' R2', ' D ', ' F2', ' D2', ' B ', ' R2', " U'", " D'", ' L2', ' B '))
        self._add_myperm2('C8~v02', (" D'", ' L ', ' D ', ' R2', " D'", " L'", ' D ', ' R2', ' U2', " B'", ' D ', ' B ', ' U2', " B'", " D'", ' B ', " R'", ' U ', ' R ', ' D2', " R'", " U'", ' R ', ' D2', ' L2', " F'", ' R ', ' F ', ' L2', " F'", " R'", ' F ', ' R ', ' B2', " R'", ' D ', ' F2', " D'", ' R ', ' B2', " R'", ' D ', ' F2', " D'"))
        self._add_myperm2('C8[3x2]+EAll12[3x4]', (" L'"," R "," U "," D'"," F'"," B "," L'"," R "))
        self._add_myperm2('C8s+EAll12[2x6]', (" L'"," R "," U2"," D2"," L'"," R "," F2"," B2"))

        self._add_myperm2('EAll12[3x4]', (' F ', ' B2', " R'", ' D2', ' B ', ' R ', ' U ', " D'", ' R ', " L'", " D'", " F'", ' R2', ' D ', ' F2', " B'"))
        self._add_myperm2('EAll12[6x2]', (' F ', ' B2', " R'", ' D2', ' B ', ' R ', ' U ', " D'", ' R ', " L'", " D'", " F'", ' R2', ' D ', ' F2', " B'", ' L2', ' R2', ' U2', ' D2', ' F2', ' B2'))
        self._add_myperm2('C8[2x4]+EAll8[2x2]', (' L ', ' U ', ' F2', ' R ', " L'", ' U2', " B'", ' U ', ' D ', ' B2', ' L ', ' F ', " B'", " R'", ' L ', " F'", ' R '))
        self._add_myperm2('C8p[4x2]', (' R2', ' L2', " U'", ' R2', ' L2', ' U2', ' B2', ' F2', ' D ', ' B2', ' F2', ' U2'))
        self._add_myperm2('C8s~v01', (' L2', ' U2', ' D2', ' F2', ' U2', ' D2', ' L2', ' R2', ' B2', ' R2'))
        self._add_myperm2('C8[3x2]~v01', (" R'", ' F2', ' B2', ' R ', ' D ', ' F2', ' B2', " D'", " U'", ' F2', ' B2', ' D ', ' B2', ' F2', ' R ', ' B2', ' F2', " L'"))
        self._add_myperm2('C8s~v02', (' U ', ' R2', ' U2', ' D2', ' B2', ' F2', ' L2', ' B2', ' F2', ' U ', ' D2'))
        self._add_myperm2('C8[4x2]', (' R2', ' L2', " U'", ' R2', ' L2', ' U2', ' B2', ' F2', ' D ', ' B2', ' F2', ' U2', ' U2', ' B2', ' D ', ' L2', " F'", " B'", ' R2', ' D ', ' F2', ' D2', ' B ', ' R2', " U'", " D'", ' L2', ' B '))
        self._add_myperm2('C8[2x4]~v01', (' L2', ' U2', ' D2', ' F2', ' U2', ' D2', ' L2', ' R2', ' B2', ' R2', ' U2', ' B2', ' D ', ' L2', " F'", " B'", ' R2', ' D ', ' F2', ' D2', ' B ', ' R2', " U'", " D'", ' L2', ' B '))
        self._add_myperm2('C8[3x2]~v02', (" R'", ' F2', ' B2', ' R ', ' D ', ' F2', ' B2', " D'", " U'", ' F2', ' B2', ' D ', ' B2', ' F2', ' R ', ' B2', ' F2', " L'", ' U2', ' B2', ' D ', ' L2', " F'", " B'", ' R2', ' D ', ' F2', ' D2', ' B ', ' R2', " U'", " D'", ' L2', ' B '))
        self._add_myperm2('C8[2x4]~v02', (' U ', ' R2', ' U2', ' D2', ' B2', ' F2', ' L2', ' B2', ' F2', ' U ', ' D2', ' U2', ' B2', ' D ', ' L2', " F'", " B'", ' R2', ' D ', ' F2', ' D2', ' B ', ' R2', " U'", " D'", ' L2', ' B '))
        self._add_myperm2('C6[DBL>FLU>RDF;DRB>BUL>RFU]+EAll6[DB>LU>FR;DR>LB>FU]', (' F ', ' R ', ' F ', " D'", ' L ', ' D ', ' F2', ' R2', " D'", " R'", ' B ', " U'", " B'", ' R2', ' D '))
        self._add_myperm2('C8s+EAll4s[BR<>RF;FL<>LB]', (' F ', ' U ', ' F ', ' R ', ' L2', ' B ', " D'", ' R ', ' D2', ' L ', " D'", ' B ', ' R2', ' L ', ' F ', ' U ', ' F '))


        self._add_myperm2('CtrCore4[B>R>F>L]+CtrObl32p[4x8]+CtrPlus32p[4x8]+CtrX32p[4x8]+EAll2[FL>FR]', (' y ', ' L2', " U'", ' D ', ' F ', " D'", " B'", " D'", ' B ', " U'", ' D ', " L'", ' D ', ' L ', ' U ', " R'", " U'", ' R ', ' U2', ' R2', " U'", " D'", " F'", ' U ', ' F ', ' D ', ' R2', " U'", ' F '))
        self._add_myperm2('CtrCore4[B>R>F>L]+CtrObl32p[4x8]+CtrPlus32p[4x8]+CtrX32p[4x8]+EAll2s[FL<>RF]', (' R ', ' U ', ' B ', ' D ', " B'", ' D ', " U'", ' L ', " D'", " L'", " D'", " R'", ' D ', " U'", ' y '))
        self._add_myperm2('CtrCore4[B>L>F>R]+CtrObl32p[4x8]+CtrPlus32p[4x8]+CtrX32p[4x8]+EAll2s[LB<>RF]', (' U ', " B'", " R'", ' U ', ' R ', ' U2', " F'", ' U ', " L'", " D'", ' L ', " D'", ' U ', " B'", ' D ', ' B ', ' D ', ' U2', ' F ', " U'", " D'", ' U2', " y'", ' B ', ' L ', " U'", " L'", ' U '))
        self._add_myperm2('CtrCore4[B>L>F>R]+CtrObl32p[4x8]+CtrPlus32p[4x8]+CtrX32p[4x8]+EAll2[LB>FR]', (' B2', " F'", " U'", " L'", " D'", ' L ', " D'", ' U ', " B'", ' D ', ' B ', ' D ', ' F ', " D'", ' U ', " y'", ' B2'))
        self._add_myperm2('CtrCore4[B>L>F>R]+CtrObl32p[4x8]+CtrPlus32p[4x8]+CtrX32p[4x8]+EAll2[RF>RU]', (" U'", " B'", " F'", " U'", " L'", " D'", ' L ', " D'", ' U ', " B'", ' D ', ' B ', ' D ', ' F ', " D'", ' U ', " y'", ' B ', ' U '))
        self._add_myperm2('CtrCore4[B>L>F>R]+CtrObl32p[4x8]+CtrPlus32p[4x8]+CtrX32p[4x8]+EAll2[RF>FU]', (' U2', " B'", " F'", " U'", " L'", " D'", ' L ', " D'", ' U ', " B'", ' D ', ' B ', ' D ', ' F ', " D'", ' U ', " y'", ' B ', ' U2'))
        self._add_myperm2('CtrCore4[B>L>F>R]+CtrObl32p[4x8]+CtrPlus32p[4x8]+CtrX32p[4x8]+EAll2[RF>LU]', (' U ', " B'", " F'", " U'", " L'", " D'", ' L ', " D'", ' U ', " B'", ' D ', ' B ', ' D ', ' F ', " D'", ' U ', " y'", ' B ', " U'"))
        self._add_myperm2('CtrCore4[B>L>F>R]+CtrObl32p[4x8]+CtrPlus32p[4x8]+CtrX32p[4x8]+EAll2[RF>BU]', (" B'", " F'", " U'", " L'", " D'", ' L ', " D'", ' U ', " B'", ' D ', ' B ', ' D ', ' F ', " D'", ' U ', " y'", ' B '))
        self._add_myperm2('CtrCore4[B>R>F>L]+CtrObl32p[4x8]+CtrPlus32p[4x8]+CtrX32p[4x8]+EAll2[RF>RU]', (" U'", " B'", ' y ', " U'", ' D ', " F'", " D'", " B'", " D'", ' B ', " U'", ' D ', " L'", ' D ', ' L ', ' U ', ' F ', ' B ', ' U '))
        self._add_myperm2('CtrCore4[B>R>F>L]+CtrObl32p[4x8]+CtrPlus32p[4x8]+CtrX32p[4x8]+EAll2[RF>FU]', (' U2', " B'", ' y ', " U'", ' D ', " F'", " D'", " B'", " D'", ' B ', " U'", ' D ', " L'", ' D ', ' L ', ' U ', ' F ', ' B ', ' U2'))
        self._add_myperm2('CtrCore4[B>R>F>L]+CtrObl32p[4x8]+CtrPlus32p[4x8]+CtrX32p[4x8]+EAll2[RF>LU]', (' U ', " B'", ' y ', " U'", ' D ', " F'", " D'", " B'", " D'", ' B ', " U'", ' D ', " L'", ' D ', ' L ', ' U ', ' F ', ' B ', " U'"))
        self._add_myperm2('CtrCore4[B>R>F>L]+CtrObl32p[4x8]+CtrPlus32p[4x8]+CtrX32p[4x8]+EAll2[RF>BU]', (" B'", ' y ', " U'", ' D ', " F'", " D'", " B'", " D'", ' B ', " U'", ' D ', " L'", ' D ', ' L ', ' U ', ' F ', ' B '))
        self._add_myperm2('C2[DFR>FUR]+CtrCore4[B>L>F>R]+CtrObl32p[16x2]+CtrPlus32p[16x2]+CtrX32p[16x2]', (" R'", " U'", " R'", ' U ', " y'", " D'", ' R ', ' U ', ' R ', " F'", ' U ', " D'", ' L ', ' D ', ' L ', " D'", " L'", ' D ', " U'", ' F ', ' D ', " F'", " D'", ' R2', ' B ', ' R ', " B'", ' R ', ' D2', " L'", ' F ', ' L ', ' D2'))
        self._add_myperm2('C2s[DFR<>UFL]+CtrCore4[B>R>F>L]+CtrObl32p[16x2]+CtrPlus32p[16x2]+CtrX32p[16x2]', (' F ', ' y ', ' L ', ' U ', ' D ', ' U2', ' F ', ' U ', " D'", ' L ', ' B2', ' L2', ' D ', " U'", ' F ', " L'", ' U ', " L'", ' U2', ' R ', " U'", " R'", " U'", " L'", ' U ', ' R2', " U'", ' L ', ' U ', ' R2', ' U2'))
        self._add_myperm2('C2[DFR>LUF]+CtrCore4[B>L>F>R]+CtrObl32p[16x2]+CtrPlus32p[16x2]+CtrX32p[16x2]', (" R'", " U'", " R'", " y'", ' U ', " D'", ' R ', ' U ', ' R ', " F'", " D'", ' U ', ' L ', " U'", ' D ', ' L ', " D'", " L'", ' D ', ' F ', ' D ', " F'", " D'", ' R2', ' B ', ' R ', " B'", ' R ', ' D2', " L'", ' F ', ' L ', ' D2'))
        self._add_myperm2('C2[DBL>FUR]+CtrCore4[B>R>F>L]+CtrObl32p[16x2]+CtrPlus32p[16x2]+CtrX32p[16x2]', (" B'", " D'", " L'", " U'", ' L ', ' D ', " U'", " F'", ' U ', ' F ', ' U ', ' B ', " U'", ' D ', ' y ', ' R ', ' B ', " R'", " B'", " D'", " R'", " D'", ' R2', " F'", ' R2', ' F ', ' D2', ' F2', ' L ', " B'", " L'", ' F2', ' L ', ' B ', " L'"))
        self._add_myperm2('C2[DRB>LUF]+CtrCore4[B>R>F>L]+CtrObl32p[16x2]+CtrPlus32p[16x2]+CtrX32p[16x2]', (' D2', " F'", ' D ', ' R2', " U'", ' B ', ' U ', ' R2', ' D ', " B'", ' D ', ' F ', " D'", ' B ', ' y ', ' B2', ' D ', " U'", " L'", ' U ', ' R ', ' U ', " R'", ' D ', " U'", ' B ', " U'", " B'", " D'", ' F ', ' D ', " F'", ' D2', ' F2', ' D ', ' U ', ' L ', " D'", " L'", " U'", ' F2', ' D ', " L'"))
        self._add_myperm2('C2s[DBL<>URF]+CtrCore4[B>R>F>L]+CtrObl32p[16x2]+CtrPlus32p[16x2]+CtrX32p[16x2]', (' L ', ' U ', ' F ', ' D ', " F'", ' D ', " U'", ' R ', " D'", " R'", " D'", " L'", " U'", ' D ', " L'", ' y ', " L'", ' B ', ' L ', ' U ', " B'", " U'", " L'", ' D ', " L'", " D'", ' L2', " B'", ' L ', " F'", " L'", ' B ', ' L ', ' F ', " L'"))
        self._add_myperm2('C2[UFL>RFU]+CtrCore4[B>R>F>L]+CtrObl32p[16x2]+CtrPlus32p[16x2]+CtrX32p[16x2]', (" F'", " D'", " R'", " U'", ' R ', ' D ', " U'", " B'", ' U ', ' B ', ' U ', ' F ', " U'", ' D ', ' y ', " L'", " F'", ' L ', ' F ', ' U ', " L'", " U'", " F'", ' D ', " F'", " D'", ' F2', ' R2', ' D2', " R'", ' U2', ' R ', ' D2', " R'", ' U2', " R'"))

    
    def _register_myperms2_x_perms(self):
        """ParitySwap系とその派生手順を登録する。"""
        # 命名メモ:
        # - ParitySwap-* は corner 2つ + midedge 2つの同時 swap。
        # - ParityCycle-* は corner 4つ + edge 2つの置換。
        # - A/B/F/J/K は corner 配置 family、末尾番号は family 内 variant。
        PLLParity = self._moves_available_for_size(
            ("2F2", "3F2", " R2", " U2", "2F2", "3F2", " U2", " R2", "2F2", "3F2")
        )


        parity_swap_moves = {}
        parity_swap_moves['ParitySwap-A0-'] = (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F2', ' R ') + PLLParity
        parity_swap_moves['ParitySwap-A1-'] = (" R'", ' F2', ' D2', " B'", " L'", ' B ', ' D2', " F'", ' R ', " F'") + PLLParity
        parity_swap_moves['ParitySwap-A2-'] = PLLParity + (" R'", ' F2', ' D2', " B'", " L'", ' B ', ' L ', ' D2', " L'", " F'", ' R ', ' F2', " L'", " F'", ' L ', ' F2')
        parity_swap_moves['ParitySwap-A3-'] = PLLParity + (' F2', " R'", ' F2', ' D2', " B'", " L'", ' B ', ' L ', ' D2', " L'", " F'", ' R ', ' F2', " L'", " F'", ' L ')
        parity_swap_moves['ParitySwap-A4-'] = PLLParity + (' F2', ' U2', ' F2', ' U2', ' F ', ' R ', " L'", ' U2', " R'", ' L ', " F'", ' B ', ' U ', " B'", ' U ', ' R2', " D'", ' F ', ' D ', ' R2')
        parity_swap_moves['ParitySwap-A5-'] = PLLParity + (' U2', ' F2', ' U2', ' F ', ' R ', " L'", ' U2', " R'", ' L ', " F'", ' B ', ' U ', " B'", ' U ', ' R2', " D'", ' F ', ' D ', ' R2', " F2")
        
        
        parity_swap_moves['ParitySwap-B0-'] = PLLParity + (" L2"," F2"," U2"," L'"," U2"," L2"," F2"," L'"," U2"," L2"," U2"," F2"," L'"," F2")
        parity_swap_moves['ParitySwap-B1-'] = PLLParity + (" F2"," L "," F2"," U2"," L2"," U2"," L "," F2"," L2"," U2"," L "," U2"," F2"," L2")
        
        parity_swap_moves['ParitySwap-B2-'] = (" R2", " U2", " B2", " R'", " B2", " R2", " U2", " R ", " B2", " R2", " B2", " U2", " R ", " U2") + PLLParity
        parity_swap_moves['ParitySwap-B3-'] = (" U2", " R'", " U2", " B2", " R2", " B2", " R'", " U2", " R2", " B2", " R ", " B2", " U2", " R2") + PLLParity

        parity_swap_moves['ParitySwap-F0-'] = PLLParity + (" R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F2', ' R ', " F ")
        parity_swap_moves['ParitySwap-F1-'] = PLLParity + (" F'", " R'", " F2", " D2", " B'", " L'", " B ", " D2", " F'", " R ")      
        parity_swap_moves['ParitySwap-F2-'] = (" B ", " L'", ' B2', ' D2', " F'", " R'", ' F ', ' D2', " B'", ' L ', " B2") + PLLParity
        parity_swap_moves['ParitySwap-F3-'] = (' B2', " L'", ' B ', ' D2', " F'", ' R ', ' F ', ' D2', ' B2', ' L ', " B'") + PLLParity
        parity_swap_moves['ParitySwap-F4-'] = (" U2", " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F2', ' R ', " F ", " U2") + PLLParity
        parity_swap_moves['ParitySwap-F5-'] = (" U2", " F'", " R'", " F2", " D2", " B'", " L'", " B ", " D2", " F'", " R ", " U2") + PLLParity




        
        parity_swap_moves['ParitySwap-J0-'] = (' B2', " L'", ' B2', ' D2', " F'", " R'", ' F ', ' D2', " B'", ' L ', ' B ') + PLLParity
        parity_swap_moves['ParitySwap-J1-'] = (" B'", " L'", ' B ', ' D2', " F'", ' R ', ' F ', ' D2', ' B2', ' L ', ' B2') + PLLParity

        parity_swap_moves['ParitySwap-J2-'] = (' F2', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F ', " L'", ' F ', ' R ', " F'", ' L ') + PLLParity
        parity_swap_moves['ParitySwap-J3-'] = (" L'", ' F ', " R'", " F'", ' L ', " F'", ' D2', " B'", " L'", ' B ', ' D2', " F'", ' R ', ' F2') + PLLParity

        parity_swap_moves['ParitySwap-J4-'] = (' B2', ' L2', ' D2', ' F2', ' D2', ' L2', " B'", ' U2', ' L2', ' D ', ' F ', " D'", ' L2', ' U ', " B'", " U'") + PLLParity
        parity_swap_moves['ParitySwap-J5-'] = (' U ', ' B ', " U'", ' L2', ' D ', " F'", " D'", ' L2', ' U2', ' B ', ' L2', ' D2', ' F2', ' D2', ' L2', ' B2') + PLLParity
    
        parity_swap_moves['ParitySwap-K0-'] = (" R'", ' U2', ' L ', ' F2', " L'", ' F2', ' R2', ' U2', ' R ', ' U2', " R'", ' U2', ' F2', ' R2', ' F2') + PLLParity
        parity_swap_moves['ParitySwap-K1-'] = (' R2', ' F2', ' U2', ' R ', ' U2', " R'", ' U2', ' R2', ' F2', ' L ', ' F2', " L'", ' U2', ' R ', ' F2') + PLLParity

        ParityCycleU = self._moves_available_for_size(
            ('2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2')
        )
        ParityCycleD = self._moves_available_for_size(
            ('2D2', '3D2', ' L2', ' F2', '2D2', '3D2', ' F2', ' L2', '2D2', '3D2')
        )

        self._add_myperm2('C4[DFR>FUR>LFD>LUF]+ME2[FL>FR]', (" U'", " R'", " F'", ' R ', ' F ', " R'", ' F ', ' R ', ' F ', " R'", " F'", ' R ', ' U ') + ParityCycleU)
        self._add_myperm2('C4[DFR>LUF>LFD>FUR]+ME2[FL>FR]', (" U'", " R'", ' F ', ' R ', " F'", " R'", " F'", ' R ', " F'", " R'", ' F ', ' R ', ' U ') + ParityCycleU)
        self._add_myperm2('C4[DBL>FLU>DRB>FUR]+ME2[FL>FR]', (' D ', " L'", " F'", ' L ', ' F ', " L'", ' F ', ' L ', ' F ', " L'", " F'", ' L ', " D'") + ParityCycleD)
        self._add_myperm2('C4[DBL>FUR>DRB>FLU]+ME2[FL>FR]', (' D ', " L'", ' F ', ' L ', " F'", " L'", " F'", ' L ', " F'", " L'", ' F ', ' L ', " D'") + ParityCycleD)
        self._add_myperm2('C4[DBL>FUR>DLF>FLU]+ME2[FL>FR]', (" L'", " F'", ' L ', ' F ', " L'", ' F ', ' L ', ' F ', " L'", " F'", ' L ') + ParityCycleD)
        self._add_myperm2('C4[DBL>FLU>DLF>FUR]+ME2[FL>FR]', (" L'", ' F ', ' L ', " F'", " L'", " F'", ' L ', " F'", " L'", ' F ', ' L ') + ParityCycleD)

 


        if self.size % 2 == 1:
            self._add_myperm2('CtrCore6p[3x2][B>R>D;F>L>U]', (" M "," E "," M'"," E'"))
            self._add_myperm2('CtrCore4s[B<>F;L<>R]', (' E ', ' S2', " E'", ' S2'))


        parity_swap_moves['ParitySwap-XB-'] = (' U ', " F'", ' R ') + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', " F'", " U'")
        parity_swap_moves['ParitySwap-XC-'] = (' F2', ' R ') + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2')
        parity_swap_moves['ParitySwap-XD-'] = (' F ', ' R ') + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F ')
        parity_swap_moves['ParitySwap-XE-'] = (' R ',) + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', " F2")
        parity_swap_moves['ParitySwap-XF-'] = (" F'", ' R ') + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', " F'")
        parity_swap_moves['ParitySwap-XG-'] = self.conjugate((" R2",),parity_swap_moves['ParitySwap-A0-'])
        parity_swap_moves['ParitySwap-XH-'] = self.conjugate((" U'"," F'"," R "),parity_swap_moves['ParitySwap-A0-'])
        

        parity_swap_moves['ParitySwap-YA-'] = PLLParity + (" R "," U "," R'"," U'"," R'"," F "," R2"," U'"," R'"," U'"," R "," U "," R'"," F'")
        parity_swap_moves['ParitySwap-YB-'] = self.conjugate((" U "," F'"," R "),parity_swap_moves['ParitySwap-YA-'])
        parity_swap_moves['ParitySwap-YC-'] = self.conjugate((" F2"," R "),parity_swap_moves['ParitySwap-YA-'])
        parity_swap_moves['ParitySwap-YD-'] = self.conjugate((" F "," R "),parity_swap_moves['ParitySwap-YA-'])
        parity_swap_moves['ParitySwap-YE-'] = self.conjugate((" R ",),parity_swap_moves['ParitySwap-YA-'])
        parity_swap_moves['ParitySwap-YF-'] = self.conjugate((" F'"," R "),parity_swap_moves['ParitySwap-YA-'])
        parity_swap_moves['ParitySwap-YG-'] = self.conjugate((" R2",),parity_swap_moves['ParitySwap-YA-'])
        parity_swap_moves['ParitySwap-YH-'] = self.conjugate((" R'"," U'"," F "," U "),parity_swap_moves['ParitySwap-YA-'])


        
        parity_swap_moves['ParitySwap-ZA-'] = PLLParity + (' U2', " B'", ' U2', ' B ', ' U2',' D2', " R'", " B'", ' R ', ' D2', " L'", ' F ', " L'", " F'", ' L2')
        parity_swap_moves['ParitySwap-ZB-'] = self.conjugate((" F'"," U "," L'"),parity_swap_moves['ParitySwap-ZA-'])
        parity_swap_moves['ParitySwap-ZC-'] = self.conjugate((" U2"," L "),parity_swap_moves['ParitySwap-ZA-'])
        parity_swap_moves['ParitySwap-ZD-'] = self.conjugate((" U "," L "),parity_swap_moves['ParitySwap-ZA-'])
        parity_swap_moves['ParitySwap-ZE-'] = self.conjugate((" L ",),parity_swap_moves['ParitySwap-ZA-'])
        parity_swap_moves['ParitySwap-ZF-'] = self.conjugate((" U'"," L "),parity_swap_moves['ParitySwap-ZA-'])
        parity_swap_moves['ParitySwap-ZG-'] = self.conjugate((" L2",),parity_swap_moves['ParitySwap-ZA-'])
        parity_swap_moves['ParitySwap-ZH-'] = self.conjugate((" F "," U "," L'"),parity_swap_moves['ParitySwap-ZA-'])

        
        parity_swap_moves['ParitySwap-JXB-'] = (" R2", ' U ', " F'", ' R ') + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', " F'", " U'", " R2")
        parity_swap_moves['ParitySwap-JYB-'] = (" R2", ' U ', " F'", ' R ') + PLLParity + (' R ', ' U ', " R'", " U'", " R'", ' F ', ' R2', " U'", " R'", " U'", ' R ', ' U ', " R'", " F'", " R'", ' F ', " U'", " R2")
        parity_swap_moves['ParitySwap-JZB-'] = (" U'", ' B ', " R'") + PLLParity + (" B'", ' R ', " B'", ' D2', ' F ', " L'", " F'", ' D2', ' B ', ' U ', ' L ', ' U2', " L'", ' D ', ' L ', ' U2', " L'", " D'")

        self._parity_swap_basis_moves = parity_swap_moves
        def add_parity_swap(name, moves):
            self._add_myperm2(name, self._moves_available_for_size(moves))

        add_parity_swap('C2[DFR>RFU]+ME2[FL>FR]~v01', (" U'", ' R ', " U'", ' B2', ' D ', " L'", " D'", ' B2', ' U2', " R'", '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2'))
        add_parity_swap('C2[DFR>RFU]+ME2[FL>FR]~v02', (' R ', ' U2', ' B2', ' D ', ' L ', " D'", ' B2', ' U ', " R'", ' U ', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2'))
        add_parity_swap('C2[UFL>FUR]+ME2[FL>FR]', ('2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' R ', ' U2', ' B2', ' D ', ' L ', " D'", " L'", ' B2', ' L ', ' U ', " R'", ' U2', ' L ', ' U ', " L'", ' U2'))
        add_parity_swap('C2[UBR>BUL]+ME2[FL>FR]', ('2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' U2', ' R ', ' U2', ' B2', ' D ', ' L ', " D'", " L'", ' B2', ' L ', ' U ', " R'", ' U2', ' L ', ' U ', " L'"))
        add_parity_swap('C2s[UFL<>URF]+ME2[FL>FR]', ('2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' U2', ' F2', ' U2', ' F2', " U'", " R'", ' L ', ' F2', ' R ', " L'", ' U ', " D'", " F'", ' D ', " F'", ' R2', ' B ', " U'", " B'", ' R2'))
        add_parity_swap('C2s[UBR<>ULB]+ME2[FL>FR]', ('2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' F2', ' U2', ' F2', " U'", " R'", ' L ', ' F2', ' R ', " L'", ' U ', " D'", " F'", ' D ', " F'", ' R2', ' B ', " U'", " B'", ' R2', ' U2'))
        add_parity_swap('C2s[DFR<>UFL]+ME2[FL>FR]~v01', ('2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' L2', ' U2', ' F2', ' L ', ' F2', ' L2', ' U2', ' L ', ' F2', ' L2', ' F2', ' U2', ' L ', ' U2'))
        add_parity_swap('C2s[DFR<>UFL]+ME2[FL>FR]~v02', ('2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' U2', " L'", ' U2', ' F2', ' L2', ' F2', " L'", ' U2', ' L2', ' F2', " L'", ' F2', ' U2', ' L2'))
        add_parity_swap('C2s[ULB<>URF]+ME2[FL>FR]~v01', (' L2', ' F2', ' U2', ' L ', ' U2', ' L2', ' F2', " L'", ' U2', ' L2', ' U2', ' F2', " L'", ' F2', '2D2', '3D2', ' L2', ' F2', '2D2', '3D2', ' F2', ' L2', '2D2', '3D2'))
        add_parity_swap('C2s[ULB<>URF]+ME2[FL>FR]~v02', (' F2', ' L ', ' F2', ' U2', ' L2', ' U2', ' L ', ' F2', ' L2', ' U2', " L'", ' U2', ' F2', ' L2', '2D2', '3D2', ' L2', ' F2', '2D2', '3D2', ' F2', ' L2', '2D2', '3D2'))
        add_parity_swap('C2[DBL>LUF]+ME2[FL>FR]~v01', ('2D2', '3D2', ' L2', ' F2', '2D2', '3D2', ' F2', ' L2', '2D2', '3D2', ' L ', " D'", ' B2', ' U ', " R'", " U'", ' B2', ' D2', " L'", " D'"))
        add_parity_swap('C2[DBL>LUF]+ME2[FL>FR]~v02', ('2D2', '3D2', ' L2', ' F2', '2D2', '3D2', ' F2', ' L2', '2D2', '3D2', ' D ', ' L ', ' D2', ' B2', ' U ', ' R ', " U'", ' B2', ' D ', " L'"))
        add_parity_swap('C2[DFR>LUF]+ME2[FL>FR]~v01', (" D'", ' L ', ' D2', ' B2', ' U ', ' R ', " U'", ' B2', ' D ', " L'", ' D2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2'))
        add_parity_swap('C2[DFR>LUF]+ME2[FL>FR]~v02', (' D2', ' L ', " D'", ' B2', ' U ', " R'", " U'", ' B2', ' D2', " L'", ' D ', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2'))
        add_parity_swap('C2[UBR>LUF]+ME2[FL>FR]~v01', (' F2', ' R ', " U'", ' B2', ' D ', " L'", " D'", ' B2', ' U2', " R'", " U'", ' F2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2'))
        add_parity_swap('C2[UBR>LUF]+ME2[FL>FR]~v02', (' F2', ' U ', ' R ', ' U2', ' B2', ' D ', ' L ', " D'", ' B2', ' U ', " R'", ' F2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2'))
        add_parity_swap('C2[DRB>LUF]+ME2[FL>FR]~v01', (' D2', ' L ', ' D2', ' B2', ' U ', ' R ', " U'", ' B2', ' D ', " L'", " D'", '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2'))
        add_parity_swap('C2[DRB>LUF]+ME2[FL>FR]~v02', (' D ', ' L ', " D'", ' B2', ' U ', " R'", " U'", ' B2', ' D2', " L'", ' D2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2'))
        add_parity_swap('C2[DRB>FLU]+ME2[FL>FR]~v01', (' D2', ' L ', " D'", ' B2', ' U ', " R'", " U'", ' B2', " D'", ' R ', " D'", " L'", ' D ', " R'", '2D2', '3D2', ' L2', ' F2', '2D2', '3D2', ' F2', ' L2', '2D2', '3D2'))
        add_parity_swap('C2[DRB>FLU]+ME2[FL>FR]~v02', (' R ', " D'", ' L ', ' D ', " R'", ' D ', ' B2', ' U ', ' R ', " U'", ' B2', ' D ', " L'", ' D2', '2D2', '3D2', ' L2', ' F2', '2D2', '3D2', ' F2', ' L2', '2D2', '3D2'))
        add_parity_swap('C2s[DRB<>UFL]+ME2[FL>FR]~v01', (' D2', ' L2', ' B2', ' U2', ' B2', ' L2', ' D ', ' F2', ' L2', " B'", " U'", ' B ', ' L2', " F'", ' D ', ' F ', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2'))
        add_parity_swap('C2s[DRB<>UFL]+ME2[FL>FR]~v02', (" F'", " D'", ' F ', ' L2', " B'", ' U ', ' B ', ' L2', ' F2', " D'", ' L2', ' B2', ' U2', ' B2', ' L2', ' D2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2'))
        add_parity_swap('C2[DBL>RFU]+ME2s[FL<>RF]', (' D2', " L'", ' F ', " D'", '2F2', '3F2', ' D2', ' L2', '2F2', '3F2', ' L2', ' D2', '2F2', '3F2', " F'", ' D ', " F'", ' R2', ' B ', " U'", " B'", ' R2', ' F ', ' L ', ' D2'))
        add_parity_swap('C2[DBL>FUR]+ME2s[FL<>RF]', (' D2', " L'", ' F ', " D'", '2F2', '3F2', ' D2', ' L2', '2F2', '3F2', ' L2', ' D2', '2F2', '3F2', " D'", " L'", ' D ', ' L ', ' D ', " F'", ' D2', ' L ', ' D ', ' L ', " D'", " L'", ' D ', ' F ', ' D ', " F'", ' L ', ' D2'))
        add_parity_swap('C2s[DBL<>URF]+ME2s[FL<>RF]', (" L'", ' F ', " D'", '2B2', '3B2', ' D2', ' L2', '2B2', '3B2', ' L2', ' D2', '2B2', '3B2', " F'", ' D ', " F'", ' R2', ' B ', " U'", " B'", ' R2', ' F ', ' L ', ' U ', ' L2', " U'", ' R ', ' U ', ' L2', " U'", " R'"))
        add_parity_swap('C2[UFL>RFU]+ME2[FL>FR]', (' R ', ' F2', " L'", ' U2', ' L ', ' U2', ' R2', ' F2', " R'", ' F2', ' R ', ' F2', ' U2', ' R2', ' U2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2'))
        add_parity_swap('C2[UBR>LBU]+ME2[FL>FR]', (' R2', ' U2', ' F2', " R'", ' F2', ' R ', ' F2', ' R2', ' U2', " L'", ' U2', ' L ', ' F2', " R'", ' U2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2'))
        add_parity_swap('C2[DLF>LUF]+ME2s[FL<>RF]', (" R'", ' F ', " U'", '2F2', '3F2', ' U2', ' R2', '2F2', '3F2', ' R2', ' U2', '2F2', '3F2', " F'", ' U ', " F'", ' L2', ' B ', " D'", " B'", ' L2', ' F ', ' R '))
        add_parity_swap('C2[UFL>FUR]+ME2s[RF<>UF]', (' U2', " B'", '2U2', '3U2', ' B2', ' R2', '2U2', '3U2', ' R2', ' B2', '2U2', '3U2', " U'", ' B ', " U'", ' L2', ' D ', " F'", " D'", ' L2'))
        add_parity_swap('C2[UFL>FUR]+ME2s[FL<>UF]', (" U'", " R'", '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', " U'", ' R ', " U'", ' B2', ' D ', " L'", " D'", ' B2', " U'"))
        add_parity_swap('C2[DFR>RFU]+ME2[RF>LU]', (' R ', '2F2', '3F2', ' R2', ' U2', '2F2', '3F2', ' U2', ' R2', '2F2', '3F2', ' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F2'))
        add_parity_swap('C2[DLF>FLU]+ME2s[FL<>UB]', (' L ', " D'", '2L2', '3L2', ' D2', ' B2', '2L2', '3L2', ' B2', ' D2', '2L2', '3L2', " L'", ' D ', " L'", ' F2', ' R ', " U'", " R'", ' F2', ' L '))
        add_parity_swap('C2[DLF>FLU]+ME2s[BR<>FL]', (' F2', ' U ', " F'", ' U ', ' L2', " D'", ' B ', ' D ', ' L2', ' U2', ' F ', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2'))
        add_parity_swap('C2[DLF>FLU]+ME2[BR>LF]~v01', (' B ', ' L ', " D'", " L'", ' D ', " L'", ' F2', ' R ', " U'", " R'", ' F2', ' L2', " D'", '2L2', '3L2', ' D2', ' B2', '2L2', '3L2', ' B2', ' D2', '2L2', '3L2', ' D ', " L'", " B'"))
        add_parity_swap('C2[DFR>FUR]+ME2[FL>FR]', ('2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', " R'", " F'", ' R ', ' F ', ' R ', " U'", ' R2', ' F ', ' R ', ' F ', " R'", " F'", ' R ', ' U '))
        add_parity_swap('C2[DLF>FLU]+ME2s[FL<>RF]', (" R'", ' F ', " U'", '2F2', '3F2', ' U2', ' R2', '2F2', '3F2', ' R2', ' U2', '2F2', '3F2', " U'", " R'", ' U ', ' R ', ' U ', " F'", ' U2', ' R ', ' U ', ' R ', " U'", " R'", ' U ', ' F ', ' U ', " F'", ' R '))
        add_parity_swap('C2s[UFL<>URF]+ME2s[RF<>UF]', (' U2', " B'", '2U2', '3U2', ' B2', ' R2', '2U2', '3U2', ' R2', ' B2', '2U2', '3U2', " B'", " R'", ' B ', ' R ', ' B ', " U'", ' B2', ' R ', ' B ', ' R ', " B'", " R'", ' B ', ' U ', ' B ', ' U2'))
        add_parity_swap('C2s[UFL<>URF]+ME2s[FL<>UF]', (" U'", " R'", '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', " R'", " F'", ' R ', ' F ', ' R ', " U'", ' R2', ' F ', ' R ', ' F ', " R'", " F'", ' R ', ' U ', ' R ', ' U '))
        add_parity_swap('C2[DFR>FUR]+ME2[RF>LU]', (' R ', '2F2', '3F2', ' R2', ' U2', '2F2', '3F2', ' U2', ' R2', '2F2', '3F2', ' R ', ' U ', " R'", " U'", " R'", ' F ', ' R2', " U'", " R'", " U'", ' R ', ' U ', " R'", " F'", " R'"))
        add_parity_swap('C2[DLF>LUF]+ME2s[FL<>UB]', (' L ', " D'", '2L2', '3L2', ' D2', ' B2', '2L2', '3L2', ' B2', ' D2', '2L2', '3L2', " D'", " B'", ' D ', ' B ', ' D ', " L'", ' D2', ' B ', ' D ', ' B ', " D'", " B'", ' D ', ' L ', ' D ', " L'"))
        add_parity_swap('C2[DLF>LUF]+ME2s[BR<>FL]', (' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F ', ' R ', " F'", " R'", " F'", ' U ', ' F2', " R'", " F'", " R'", ' F ', ' R ', " F'", " U'", ' F2'))
        add_parity_swap('C2[DLF>FLU]+ME2[BR>LF]~v02', (' L ', ' D ', " B'", " D'", '2B2', '3B2', ' L2', ' D2', '2B2', '3B2', ' D2', ' L2', '2B2', '3B2', " L'", " D'", ' L ', ' D ', ' L ', " B'", ' L2', ' D ', ' L ', ' D ', " L'", " D'", ' L ', ' B ', ' D ', ' B ', " D'", " L'"))
        add_parity_swap('C2s[DFR<>URF]+ME2[FL>FR]', ('2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' F2', ' D ', ' F2', " D'", ' F2', ' B2', ' R ', ' D ', " R'", ' B2', ' L ', " U'", ' L ', ' U ', ' L2'))
        add_parity_swap('C2s[DLF<>UFL]+ME2s[FL<>RF]', (" R'", ' F ', " D'", '2R2', '3R2', ' U2', ' F2', '2R2', '3R2', ' F2', ' U2', '2R2', '3R2', ' F2', " L'", ' F2', ' L ', ' F2', ' B2', " U'", " L'", ' U ', ' B2', " D'", ' R ', " D'", " R'", " D'", " F'", ' R '))
        add_parity_swap('C2[UFL>RFU]+ME2[FL>FU]', (' U2', ' F ', '2R2', '3R2', ' B2', ' U2', '2R2', '3R2', ' U2', ' B2', '2R2', '3R2', ' U2', " L'", ' U2', ' L ', ' U2', ' D2', " B'", " L'", ' B ', ' D2', " F'", ' R ', " F'", " R'", ' F ', ' U2'))
        add_parity_swap('C2[UFL>RFU]+ME2s[FL<>UF]', (" U'", " L'", '2B2', '3B2', ' R2', ' U2', '2B2', '3B2', ' U2', ' R2', '2B2', '3B2', ' U2', ' F ', ' U2', " F'", ' U2', ' D2', ' R ', ' F ', " R'", ' D2', ' L ', " B'", ' L ', ' B ', " L'", ' U '))
        add_parity_swap('C2[UBR>FUR]+ME2s[FL<>UR]', (" L'", '2B2', '3B2', ' R2', ' U2', '2B2', '3B2', ' U2', ' R2', '2B2', '3B2', ' U2', ' F ', ' U2', " F'", ' U2', ' D2', ' R ', ' F ', " R'", ' D2', ' L ', " B'", ' L ', ' B ', " L'"))
        add_parity_swap('C2s[DLF<>UFL]+ME2[FL>RU]', (" F'", ' U ', '2L2', '3L2', ' D2', ' F2', '2L2', '3L2', ' F2', ' D2', '2L2', '3L2', ' F2', " R'", ' F2', ' R ', ' F2', ' B2', " D'", " R'", ' D ', ' B2', " U'", ' L ', " U'", " L'", ' U ', ' F '))
        add_parity_swap('C2s[DFR<>URF]+ME2s[LB<>RF]', (' B2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' R2', " D'", ' R2', ' D ', ' R2', ' L2', " F'", " D'", ' F ', ' L2', " B'", ' U ', " B'", " U'"))
        add_parity_swap('C2s[DLF<>UFL]+ME2[BR>LF]', (' R ', ' F ', " D'", '2R2', '3R2', ' U2', ' F2', '2R2', '3R2', ' F2', ' U2', '2R2', '3R2', ' F2', " L'", ' F2', ' L ', ' F2', ' B2', " U'", " L'", ' U ', ' B2', " D'", ' R ', " D'", " R'", " D'", " F'", " R'"))

        if self.size >= 6:
            self._add_myperm2('C2[DRB>LUF]+ME2s[FL<>UL]', (' D2', ' F2', ' R ', '2F2', '3F2', ' R2', ' U2', '2F2', '3F2', ' U2', ' R2', '2F2', '3F2', ' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B '))
            self._add_myperm2('C2[DRB>LUF]+ME2[RF>RU]', (" z'", ' F ', " B'", ' y ', " U'", " D'", ' F2', ' R ', '2F2', '3F2', ' R2', ' U2', '2F2', '3F2', ' U2', ' R2', '2F2', '3F2', ' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', " D'", ' U ', " y'", ' B ', " F'", ' z '))
            self._add_myperm2('C2[DRB>FLU]+ME2s[FL<>UF]', (' D2', ' L ', ' D ', '2L2', '3L2', ' D2', ' F2', '2L2', '3L2', ' F2', ' D2', '2L2', '3L2', ' L ', " D'", ' L ', ' B2', " R'", ' U ', ' R ', ' B2', ' L ', ' D2'))
            self._add_myperm2('C2[DBL>RFU]+ME2[FL>LU]', (" y'", ' U ', " D'", ' z ', " F'", " B'", " U'", " B'", '2U2', '3U2', ' B2', ' R2', '2U2', '3U2', ' R2', ' B2', '2U2', '3U2', " U'", ' B ', " U'", ' L2', ' D ', " F'", " D'", ' L2', " U'", ' B ', ' F ', " z'", ' D ', " U'", ' y '))
            self._add_myperm2('C2[DLF>RUB]+ME2s[FL<>UR]', (' U2', ' F2', ' R ', '2F2', '3F2', ' R2', ' U2', '2F2', '3F2', ' U2', ' R2', '2F2', '3F2', ' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' U2'))
            self._add_myperm2('C2[DRB>FLU]+ME2s[RF<>UB]', (" y'", ' U ', " D'", " x'", ' R ', " L'", ' R2', ' U2', " B'", '2U2', '3U2', ' B2', ' R2', '2U2', '3U2', ' R2', ' B2', '2U2', '3U2', " U'", ' B ', " U'", ' L2', ' D ', " F'", " D'", ' L2', ' R2', ' L ', " R'", ' x ', ' D ', " U'", ' y '))
            self._add_myperm2('C2[DLF>BRU]+ME2s[FL<>UB]', (' U2', ' L ', ' D ', '2L2', '3L2', ' D2', ' F2', '2L2', '3L2', ' F2', ' D2', '2L2', '3L2', ' L ', " D'", ' L ', ' B2', " R'", ' U ', ' R ', ' B2', ' L ', ' U2'))
            self._add_myperm2('C2[DBL>FUR]+ME2s[FL<>UB]', (" x'", " L'", ' R ', ' y ', " U'", ' D ', ' U2', " L'", " D'", '2L2', '3L2', ' D2', ' B2', '2L2', '3L2', ' B2', ' D2', '2L2', '3L2', " L'", ' D ', " L'", ' F2', ' R ', " U'", " R'", ' F2', " L'", ' U2', " D'", ' U ', " y'", " R'", ' L ', ' x '))
        else:
            self.myperms2['SuperParitySwap-JC00-'] = (" D2",' F2', ' R ') + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ')
            self.myperms2['SuperParitySwap-JE00-'] = (" U2",' F2', ' R ') + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B '," D2"," U2")
            self.myperms2['SuperParitySwap-JD00-'] = (" R2",' F ', ' R ') + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F '," R2")
            self.myperms2['SuperParitySwap-JF00-'] = (" L2",' F ', ' R ') + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F '," L2")

            self.myperms2['SuperParitySwap-JC01-'] = self.conjugate((" z'"," F "," B'"," y "," U'"," D "),self.myperms2['SuperParitySwap-JC00-'])
            self.myperms2['SuperParitySwap-JD01-'] = self.conjugate((" z "," F'"," B "," x "," L "," R'"),self.myperms2['SuperParitySwap-JD00-'])
            self.myperms2['SuperParitySwap-JE01-'] = self.conjugate((" z "," F'"," B "," y "," U'"," D "),self.myperms2['SuperParitySwap-JE00-'])
            self.myperms2['SuperParitySwap-JF01-'] = self.conjugate((" z'"," F "," B'"," x "," L "," R'"),self.myperms2['SuperParitySwap-JF00-'])


    def _register_myperms2_odd_size(self):
        """奇数サイズで使うQ/P/R系の手順を登録する。"""
        # 命名メモ:
        # - CenterMidEdgeSwap-P,Q* は center 4つの cycle と midedge 2つの swap。
        # - CenterMidEdgeSwap-R,S* は center 6つ((2,2,2)-cycle)と midedge 2つの swap。
        # - CenterCornerSwap-* は center 4つの cycle と corner 2つの swap。
        # - Q/P/S/R は配置 family、末尾の英字や番号は向き違い・variant。
        if self.size % 2 == 1:
            center_midedge_swap_qa_basis = (' S ', ' D ', ' S ', " D'", ' S ', " D'", ' S ', ' D ', ' S2', " D'", ' S ', ' D2', ' L2', " S'", " D'", " S'", ' D ', ' L2', " D'")
            self._add_myperm2('CtrCore4[B>U>F>D]+ME2[FL>FR]', (' M ', " F'", ' M ', ' F ', ' M ', ' F ', ' M ', " F'", ' M2', ' F ', ' U2', " F'", " M'", ' F ', ' M ', ' U2', ' F2', " M'", ' F '))
            self._add_myperm2('CtrCore4[B>U>F>D]+ME2s[FL<>RF]', (' S ', " E'", " S'", ' B ', " E'", ' B2', ' R2', ' B ', " E'", ' B ', " E'", ' B2', ' R2', ' B ', " E'"))
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2s[RF<>UF]', (' R ', " S'", ' U ', " S'", " U'", ' S ', ' U2', " S'", ' U2', " S'", " U'", " S'", ' U ', ' S2', ' R2', ' S ', ' R '))
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2[RF>FU]', (" F'", ' U2', ' F ', ' M ', ' F ', ' M ', " F'", ' M ', " F'", ' M ', ' U2', ' F ', ' E ', ' M ', " E'", ' M ', ' U2', " M'", ' U2'))
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2[FL>BU]', (' M ', " E'", " M'", " B'", " E'", ' B ', " E'", ' B ', " E'", " B'", " E'", " M'", ' F ', " M'", ' F2', ' M ', ' F ', ' M '))
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2s[FL<>UB]', (' M ', " E'", " M'", " B'", " E'", ' B ', " E'", ' B ', " E'", " B'", " E'", " F'", ' M ', ' F2', " M'", " F'"))
            self._add_myperm2('CtrCore4[B>U>F>D]+ME2s[LB<>RF]', (" B'", ' M ', ' F ', ' M ', " F'", ' M ', " F'", ' M ', ' F ', ' M ', " B'", " L'", ' E ', ' L ', ' B2', " L'", " E'", ' L '))
            self._add_myperm2('CtrCore4[B>U>F>D]+ME2[LB>FR]', (" B'", ' M ', ' F ', ' M ', " F'", ' M ', " F'", ' M ', ' F ', ' M ', ' B ', ' E ', ' B2', " E'", ' B2'))
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2[FL>FR]', (" E'", ' B ', " E'", " B'", " E'", " B'", " E'", ' B ', ' M ', " E'", " F'", ' M ', ' F2', ' D2', " M'", " F'", " M'", ' F ', ' D2', " F'"))
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2s[FL<>RF]', (' E ', ' R2', " E'", ' R ', " S'", ' L ', " S'", " L'", " S'", " L'", " S'", ' L ', " S'", ' R '))
            self._add_myperm2('CtrCore4[B>L>F>R]+ME2[FL>FR]', (' E ', " B'", ' E ', ' B ', ' E ', ' B ', ' E ', " B'", ' E2', ' B ', ' E ', ' B2', ' L2', " E'", ' B ', " E'", " B'", ' L2', ' B '))
            self._add_myperm2('CtrCore4[B>L>F>R]+ME2s[FL<>RF]', (' E ', ' B ', ' E ', ' B2', ' L2', ' B ', ' E ', ' B ', ' E ', ' B2', ' L2', ' B ', ' E '))
            self._add_myperm2('CtrCore4[B>R>F>L]+ME2[RF>RU]', (' F2', ' E ', ' F2', ' E2', " R'", " E'", ' R ', " E'", ' R ', " E'", " R'", ' E2', ' L ', " S'", ' L2', ' S ', ' L ', ' E '))
            self._add_myperm2('CtrCore4[B>L>F>R]+ME2[RF>RU]', (" E'", " L'", " S'", ' L2', ' S ', " L'", ' E2', ' R ', ' E ', " R'", ' E ', " R'", ' E ', ' R ', ' E2', ' F2', " E'", ' F2'))
            self._add_myperm2('CtrCore4[B>R>F>L]+ME2s[RF<>UR]', (' F2', ' E ', ' F2', ' E2', " R'", ' F2', " E'", ' R ', " E'", ' R ', " E'", " R'", " E'", " R'", ' F2', ' R '))
            self._add_myperm2('CtrCore4[B>L>F>R]+ME2s[RF<>UR]', (" R'", ' F2', ' R ', ' E ', ' R ', ' E ', " R'", ' E ', " R'", ' E ', ' F2', ' R ', ' E2', ' F2', " E'", ' F2'))
            self._add_myperm2('CtrCore4[B>R>F>L]+ME2[RF>LU]', (" E'", " L'", " E'", ' L ', " E'", ' L ', " E'", " L'", " E'", " S'", ' R ', " S'", ' R2', ' S ', ' R ', ' S '))
            self._add_myperm2('CtrCore4[B>L>F>R]+ME2[RF>LU]', (" S'", " R'", " S'", ' R2', ' S ', " R'", ' S ', ' E ', ' L ', ' E ', " L'", ' E ', " L'", ' E ', ' L ', ' E '))
            self._add_myperm2('CtrCore4[B>R>F>L]+ME2s[RF<>UL]', (" E'", ' L ', " E'", ' L2', ' B2', ' L ', " E'", ' L ', " E'", " L'", " E'", " L'", ' B2', ' L '))
            self._add_myperm2('CtrCore4[B>L>F>R]+ME2s[RF<>UL]', (" L'", ' B2', ' L ', ' E ', ' L ', ' E ', " L'", ' E ', " L'", ' B2', ' L2', ' E ', " L'", ' E '))
            self._add_myperm2('CtrCore4[B>R>F>L]+ME2s[LB<>RF]', (" E'", ' F ', " E'", " F'", " E'", " F'", ' B2', " E'", ' F ', ' B2', ' E2', " F'", ' R2', ' F ', ' E ', " F'", ' R2', ' F '))
            self._add_myperm2('CtrCore4[B>R>F>L]+ME2[LB>FR]', (' L ', " E'", " L'", " E'", ' L ', " E'", ' L ', " E'", " L'", " E'", " L'", " E'", ' R2', ' E ', ' R2'))
            self._add_myperm2('CtrCore6s[B<>U;D<>F;L<>R]+ME2[FL>FR]', (' M ', " F'", ' M ', ' F ', ' M ', ' F ', ' M ', " F'", ' M2', ' F ', ' U2', " F'", " M'", ' F ', ' M ', ' U2', ' F2', " M'", ' F ', " E'", ' M2', ' E ', ' M2'))
            self._add_myperm2('CtrCore6s[B<>U;D<>F;L<>R]+ME2s[FL<>RF]', (' S ', " E'", " S'", ' B ', " E'", ' B2', ' R2', ' B ', " E'", ' B ', " E'", ' B2', ' R2', ' B ', ' E2', ' M2', ' E ', ' M2'))
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2s[RF<>UF]', (' R ', " S'", ' U ', " S'", " U'", ' S ', ' U2', " S'", ' U2', " S'", " U'", " S'", ' U ', ' S2', ' R2', ' S ', ' R ', " M'", ' S2', ' M ', ' S2'))
            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2s[RF<>UF]', (' R ', " S'", ' U ', " S'", " U'", ' S ', ' U2', " S'", ' U2', " S'", " U'", " S'", ' U ', ' S2', ' R2', ' S ', ' R ', ' E ', ' S2', " E'", ' S2'))
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2[RF>FU]', (" F'", ' U2', ' F ', ' M ', ' F ', ' M ', " F'", ' M ', " F'", ' M ', ' U2', ' F ', ' E ', ' M ', " E'", ' M ', ' U2', " M'", ' U2', " M'", ' S2', ' M ', ' S2'))
            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2[RF>FU]', (" F'", ' U2', ' F ', ' M ', ' F ', ' M ', " F'", ' M ', " F'", ' M ', ' U2', ' F ', ' E ', ' M ', " E'", ' M ', ' U2', " M'", ' U2', ' E ', ' S2', " E'", ' S2'))
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2[FL>BU]', (' M ', " E'", " M'", " B'", " E'", ' B ', " E'", ' B ', " E'", " B'", " E'", " M'", ' F ', " M'", ' F2', ' M ', ' F ', ' S2', ' M ', ' S2'))
            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2[FL>BU]', (' M ', " E'", " M'", " B'", " E'", ' B ', " E'", ' B ', " E'", " B'", " E'", " M'", ' F ', " M'", ' F2', ' M ', ' F ', ' M ', ' E ', ' S2', " E'", ' S2'))
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2s[FL<>UB]', (' M ', " E'", " M'", " B'", " E'", ' B ', " E'", ' B ', " E'", " B'", " E'", " F'", ' M ', ' F2', " M'", " F'", " M'", ' S2', ' M ', ' S2'))
            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2s[FL<>UB]', (' M ', " E'", " M'", " B'", " E'", ' B ', " E'", ' B ', " E'", " B'", " E'", " F'", ' M ', ' F2', " M'", " F'", ' E ', ' S2', " E'", ' S2'))
            self._add_myperm2('CtrCore6s[B<>U;D<>F;L<>R]+ME2s[LB<>RF]', (" B'", ' M ', ' F ', ' M ', " F'", ' M ', " F'", ' M ', ' F ', ' M ', " B'", " E'", " R'", " E'", ' R ', ' B2', " R'", ' E ', ' R ', ' M2', ' E ', ' M2'))
            self._add_myperm2('CtrCore6s[B<>U;D<>F;L<>R]+ME2[LB>FR]', (" B'", ' M ', ' F ', ' M ', " F'", ' M ', " F'", ' M ', ' F ', ' M ', ' B ', ' E ', ' B2', " E'", ' B2', " E'", ' M2', ' E ', ' M2'))
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2[FL>FR]', (" E'", ' B ', " E'", " B'", " E'", " B'", " E'", ' B ', ' M ', " E'", " F'", ' M ', ' F2', ' D2', " M'", " F'", " M'", ' F ', ' D2', " F'", " M'", ' S2', ' M ', ' S2'))
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2s[FL<>RF]', (' E ', ' R2', " E'", ' R ', " S'", ' L ', " S'", " L'", " S'", " L'", " S'", ' L ', " S'", ' R ', " M'", ' S2', ' M ', ' S2'))
            self._add_myperm2('CtrCore6s[B<>L;D<>U;F<>R]+ME2[FL>FR]', (' E ', " B'", ' E ', ' B ', ' E ', ' B ', ' E ', " B'", ' E2', ' B ', ' E ', ' B2', ' L2', " E'", ' B ', " E'", " B'", ' L2', ' B ', ' M ', ' E2', " M'", ' E2'))
            self._add_myperm2('CtrCore6s[B<>L;D<>U;F<>R]+ME2s[FL<>RF]', (' M ', ' E2', " M'", ' E ', " B'", " E'", ' B2', ' R2', " B'", " E'", " B'", " E'", ' B2', ' R2', " B'", " E'"))
            self._add_myperm2('CtrCore6s[B<>D;F<>U;L<>R]+ME2s[RF<>UF]', (' U2', ' M ', ' U2', ' M2', " F'", " M'", ' F ', " M'", ' F ', " M'", " F'", ' M2', ' B ', ' E ', ' B2', " E'", ' B ', ' M ', " E'", ' M2', ' E ', ' M2'))
            self._add_myperm2('CtrCore6s[B<>U;D<>F;L<>R]+ME2s[RF<>UF]', (' U2', ' M ', ' U2', ' M2', " F'", " M'", ' F ', " M'", ' F ', " M'", " F'", ' M2', ' B ', ' E ', ' B2', " E'", ' B ', ' M ', " S'", ' M2', ' S ', ' M2'))
            self._add_myperm2('CtrCore6s[B<>D;F<>U;L<>R]+ME2[RF>FU]', (' U2', ' M ', ' U2', ' M2', " F'", ' U2', " M'", ' F ', " M'", ' F ', " M'", " F'", " M'", " F'", ' U2', ' F ', " E'", ' M2', ' E ', ' M2'))
            self._add_myperm2('CtrCore6s[B<>U;D<>F;L<>R]+ME2[RF>FU]', (' U2', ' M ', ' U2', ' M2', " F'", ' U2', " M'", ' F ', " M'", ' F ', " M'", " F'", " M'", " F'", ' U2', ' F ', " S'", ' M2', ' S ', ' M2'))
            self._add_myperm2('CtrCore6s[B<>U;D<>F;L<>R]+ME2s[RF<>UB]', (' M ', ' F ', ' M ', " F'", ' M ', " F'", ' M ', ' F ', ' M ', " E'", " B'", " E'", ' B2', ' E ', " B'", ' E2', ' M2', " E'", ' M2'))
            self._add_myperm2('CtrCore6s[B<>D;F<>U;L<>R]+ME2s[RF<>UB]', (' M ', ' F ', ' M ', " F'", ' M ', " F'", ' M ', ' F ', ' M ', " E'", " B'", " E'", ' B2', ' E ', " B'", ' E ', " S'", ' M2', ' S ', ' M2'))
            self._add_myperm2('CtrCore6s[B<>U;D<>F;L<>R]+ME2[RF>BU]', (' M ', " F'", ' M ', ' F2', ' D2', " F'", ' M ', " F'", ' M ', ' F ', ' M ', ' F ', ' D2', " F'", ' E ', ' M2', " E'", ' M2'))
            self._add_myperm2('CtrCore6s[B<>D;F<>U;L<>R]+ME2[RF>BU]', (' M ', " F'", ' M ', ' F2', ' D2', " F'", ' M ', " F'", ' M ', ' F ', ' M ', ' F ', ' D2', " F'", " S'", ' M2', ' S ', ' M2'))
            self._add_myperm2('CtrCore6s[B<>L;D<>U;F<>R]+ME2s[LB<>RF]', (" E'", ' F ', " E'", " F'", " E'", " F'", ' B2', " E'", ' F ', ' B2', ' E2', " F'", ' R2', ' F ', ' E ', " F'", ' R2', ' F ', ' S ', ' E2', " S'", ' E2'))
            self._add_myperm2('CtrCore6s[B<>R;D<>U;F<>L]+ME2s[LB<>RF]', (" E'", ' F ', " E'", " F'", " E'", " F'", ' B2', " E'", ' F ', ' B2', ' E2', " F'", ' R2', ' F ', ' E ', " F'", ' R2', ' F ', ' M ', ' E2', " M'", ' E2'))
            self._add_myperm2('CtrCore6s[B<>L;D<>U;F<>R]+ME2[LB>FR]', (" B'", ' E ', ' B ', ' E ', " B'", ' E ', " B'", ' E ', ' B ', ' E ', ' B ', " M'", ' E2', ' M ', " E'", ' F2', " E'", ' F2'))
            self._add_myperm2('CtrCore6s[B<>R;D<>U;F<>L]+ME2[LB>FR]', (" B'", ' E ', ' B ', ' E ', " B'", ' E ', " B'", ' E ', ' B ', ' E ', ' B ', ' S ', ' E2', " S'", " E'", ' F2', " E'", ' F2'))




            if self.size >= 6:
                self._add_myperm2('C2[DFR>FUR]+CtrCore4[B>R>F>L]~v01', (' U ', " F'", ' U ', ' L2', " D'", ' B ', ' D ', ' L2', ' U2', ' F ', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', " E'", ' L ', " E'", " L'", " E'", " L'", " E'", ' L ', ' E2', " L'", " E'", ' L2', ' B2', ' E ', " L'", ' E ', ' L ', ' B2', " L'"))
                self._add_myperm2('C2[DFR>FUR]+CtrCore4[B>R>F>L]~v02', (" F'", ' U2', ' L2', " D'", " B'", ' D ', ' L2', " U'", ' F ', " U'", '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', " E'", ' L ', " E'", " L'", " E'", " L'", " E'", ' L ', ' E2', " L'", " E'", ' L2', ' B2', ' E ', " L'", ' E ', ' L ', ' B2', " L'"))
                self._add_myperm2('C2s[DFR<>UFL]+CtrCore4[B>L>F>R]~v01', ('2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' L2', ' U2', ' F2', ' L ', ' F2', ' L2', ' U2', ' L ', ' F2', ' L2', ' F2', ' U2', ' L ', ' U2', ' E ', " B'", ' E ', ' B ', ' E ', ' B ', ' E ', " B'", ' E2', ' B ', ' E ', ' B2', ' L2', " E'", ' B ', " E'", " B'", ' L2', ' B '))
                self._add_myperm2('C2s[DFR<>UFL]+CtrCore4[B>L>F>R]~v02', ('2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' U2', " L'", ' U2', ' F2', ' L2', ' F2', " L'", ' U2', ' L2', ' F2', " L'", ' F2', ' U2', ' L2', ' E ', " B'", ' E ', ' B ', ' E ', ' B ', ' E ', " B'", ' E2', ' B ', ' E ', ' B2', ' L2', " E'", ' B ', " E'", " B'", ' L2', ' B '))
                self._add_myperm2('C2[DFR>LUF]+CtrCore4[B>R>F>L]~v01', ('2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', " F'", ' U ', ' L2', " D'", ' B ', ' D ', ' L2', ' U2', ' F ', ' U ', " E'", ' L ', " E'", " L'", " E'", " L'", " E'", ' L ', ' E2', " L'", " E'", ' L2', ' B2', ' E ', " L'", ' E ', ' L ', ' B2', " L'"))
                self._add_myperm2('C2[DFR>LUF]+CtrCore4[B>R>F>L]~v02', ('2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', " U'", " F'", ' U2', ' L2', " D'", " B'", ' D ', ' L2', " U'", ' F ', " E'", ' L ', " E'", " L'", " E'", " L'", " E'", ' L ', ' E2', " L'", " E'", ' L2', ' B2', ' E ', " L'", ' E ', ' L ', ' B2', " L'"))
                self._add_myperm2('C2[DBL>FUR]+CtrCore4[B>L>F>R]~v01', (' U2', " B'", ' U2', ' R2', " D'", " F'", ' D ', ' R2', " U'", ' B ', ' U ', '2D2', '3D2', ' F2', ' L2', '2D2', '3D2', ' L2', ' F2', '2D2', '3D2', ' E ', ' R ', ' E ', " R'", ' E ', " R'", ' E ', ' R ', ' E2', " R'", ' E ', ' R2', ' B2', " E'", " R'", " E'", ' R ', ' B2', " R'"))
                self._add_myperm2('C2[DBL>FUR]+CtrCore4[B>L>F>R]~v02', (" U'", " B'", ' U ', ' R2', " D'", ' F ', ' D ', ' R2', ' U2', ' B ', ' U2', '2D2', '3D2', ' F2', ' L2', '2D2', '3D2', ' L2', ' F2', '2D2', '3D2', ' E ', ' R ', ' E ', " R'", ' E ', " R'", ' E ', ' R ', ' E2', " R'", ' E ', ' R2', ' B2', " E'", " R'", " E'", ' R ', ' B2', " R'"))
                self._add_myperm2('C2[DRB>LUF]+CtrCore4[B>L>F>R]~v01', (' D2', " F'", ' D ', ' R2', " U'", ' B ', ' U ', ' R2', ' D ', " B'", ' D ', ' F ', " D'", ' B ', '2D2', '3D2', ' F2', ' L2', '2D2', '3D2', ' L2', ' F2', '2D2', '3D2', ' E ', ' R ', ' E ', " R'", ' E ', " R'", ' E ', ' R ', ' E2', " R'", ' E ', ' R2', ' B2', " E'", " R'", " E'", ' R ', ' B2', " R'"))
                self._add_myperm2('C2[DRB>LUF]+CtrCore4[B>L>F>R]~v02', (" B'", ' D ', " F'", " D'", ' B ', " D'", ' R2', " U'", " B'", ' U ', ' R2', " D'", ' F ', ' D2', '2D2', '3D2', ' F2', ' L2', '2D2', '3D2', ' L2', ' F2', '2D2', '3D2', ' E ', ' R ', ' E ', " R'", ' E ', " R'", ' E ', ' R ', ' E2', " R'", ' E ', ' R2', ' B2', " E'", " R'", " E'", ' R ', ' B2', " R'"))
                self._add_myperm2('C2s[DBL<>URF]+CtrCore4[B>L>F>R]~v01', (' E ', ' F ', ' E ', ' F2', ' R2', ' F ', ' E ', ' F ', ' E ', ' F2', ' R2', ' F ', ' E ', " R'", ' B ', " D'", '3F2', '2F2', ' D2', ' R2', '3F2', '2F2', ' R2', ' D2', '3F2', '2F2', " B'", ' D ', " B'", ' L2', ' F ', " U'", " F'", ' L2', ' B ', ' R ', " D'", ' R2', ' D ', " L'", " D'", ' R2', ' D ', ' L '))
                self._add_myperm2('C2s[DFR<>URF]+CtrCore4[B>R>F>L]~v01', ('2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' R2', " D'", ' R2', ' D ', ' R2', ' L2', " F'", " D'", ' F ', ' L2', " B'", ' U ', " B'", " U'", ' B2', " E'", ' L ', " E'", " L'", " E'", " L'", " E'", ' L ', ' E2', " L'", " E'", ' L2', ' B2', ' E ', " L'", ' E ', ' L ', ' B2', " L'"))
                self._add_myperm2('C2s[DFR<>URF]+CtrCore4[B>R>F>L]~v02', ('2U2', '3U2', ' F2', ' R2', '2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' R2', " D'", ' R2', ' D ', ' R2', ' L2', " F'", " D'", ' F ', ' L2', " B'", ' U ', " B'", " U'", ' B2', " E'", ' L ', " E'", " L'", " E'", " L'", " E'", ' L ', ' E2', " L'", " E'", ' L2', ' B2', ' E ', " L'", ' E ', ' L ', ' B2', " L'"))
            else:
                parity_swap_basis = self._parity_swap_basis_moves
                self._add_myperm2('C2[UBR>RFU]+CtrCore4[D>L>U>R]~v01', parity_swap_basis['ParitySwap-A0-'] + center_midedge_swap_qa_basis)
                self._add_myperm2('C2[UBR>RFU]+CtrCore4[D>L>U>R]~v02', parity_swap_basis['ParitySwap-A1-'] + center_midedge_swap_qa_basis)
                self._add_myperm2('C2s[UBR<>UFL]+CtrCore4[D>L>U>R]~v01', parity_swap_basis['ParitySwap-B0-'] + center_midedge_swap_qa_basis)
                self._add_myperm2('C2s[UBR<>UFL]+CtrCore4[D>L>U>R]~v02', parity_swap_basis['ParitySwap-B1-'] + center_midedge_swap_qa_basis)
                self._add_myperm2('C2[DFR>BRU]+CtrCore4[D>L>U>R]~v01', parity_swap_basis['ParitySwap-F0-'] + center_midedge_swap_qa_basis)
                self._add_myperm2('C2[DFR>BRU]+CtrCore4[D>L>U>R]~v02', parity_swap_basis['ParitySwap-F1-'] + center_midedge_swap_qa_basis)
                self._add_myperm2('C2[DRB>LUF]+CtrCore4[D>L>U>R]~v01', parity_swap_basis['ParitySwap-J0-'] + center_midedge_swap_qa_basis)
                self._add_myperm2('C2[DRB>LUF]+CtrCore4[D>L>U>R]~v02', parity_swap_basis['ParitySwap-J1-'] + center_midedge_swap_qa_basis)
                self._add_myperm2('C2s[DLF<>UBR]+CtrCore4[D>L>U>R]~v01', parity_swap_basis['ParitySwap-J2-'] + center_midedge_swap_qa_basis)
                self._add_myperm2('C2s[DLF<>UBR]+CtrCore4[D>L>U>R]~v02', parity_swap_basis['ParitySwap-J3-'] + center_midedge_swap_qa_basis)
                self._add_myperm2('C2[DRB>FLU]+CtrCore4[D>L>U>R]~v01', parity_swap_basis['ParitySwap-J4-'] + center_midedge_swap_qa_basis)
                self._add_myperm2('C2[DRB>FLU]+CtrCore4[D>L>U>R]~v02', parity_swap_basis['ParitySwap-J5-'] + center_midedge_swap_qa_basis)
                self._add_myperm2('C2[UBR>FUR]+CtrCore4[D>L>U>R]~v01', parity_swap_basis['ParitySwap-ZA-'] + center_midedge_swap_qa_basis)
                self._add_myperm2('C2[UBR>FUR]+CtrCore4[D>L>U>R]~v02', parity_swap_basis['ParitySwap-ZA-'] + center_midedge_swap_qa_basis)

    def _register_myperms2_general(self):
        """通常モードで使う汎用手順群を登録する。"""
        self._register_myperms2_classic_perms()
        self._register_myperms2_midedge_general()
        self._register_myperms2_edge_general()
        self._register_myperms2_center_general()

    def _register_myperms2_classic_perms(self):
        """小サイズで使うPLL系の基本手順を登録する。"""
        if self.size <= 1:
            self.myperms2['G-Perm-A'] = (" R2"," U'"," R "," U'"," R "," U "," R'"," U "," R2"," U "," D'"," R "," U'"," R'"," D ")
            self.myperms2['G-Perm-B'] = (" D'"," R "," U "," R'"," D "," U'"," R2"," U'"," R "," U'"," R'"," U "," R'"," U "," R2")
            
            self.myperms2['T-Perm'] = (" R "," U "," R'"," U'"," R'"," F "," R2"," U'"," R'"," U'"," R "," U "," R'"," F'")
            self.myperms2['N-Perm'] = (" R'"," U "," R "," U'"," R'"," F'"," U'"," F "," R "," U "," R'"," F "," R'"," F'"," R "," U'"," R ")
            self.myperms2['F-Perm'] = (" R'"," U'"," F'"," R "," U "," R'"," U'"," R'"," F "," R2"," U'"," R'"," U'"," R "," U "," R'"," U "," R ")

            #self.myperms2['J-Perm'] = (" R "," U "," R'"," F'"," R "," U "," R'"," U'"," R'"," F "," R2"," U'"," R'"," U'")
            self.myperms2['J-Perm'] = (" R "," U2"," R'"," U'"," R "," U2"," L'"," U "," R'"," U'"," L ",)
            self.myperms2['Y-Perm'] = (" F "," R "," U'"," R'"," U'"," R "," U "," R'"," F'"," R "," U "," R'"," U'"," R'"," F "," R "," F'")
            self.myperms2['R-Perm'] = (" U "," R2"," F "," R "," U "," R "," U'"," R'"," F'"," R "," U2"," R'"," U2"," R ")
            self.myperms2['V-Perm'] = (" R "," U'"," R "," U "," R'"," D "," R "," D'"," R "," U'"," D "," R2"," U "," R2"," D'"," R2")

        
        self._add_myperm2('C4[DLF>FLU;UBR>RFU;UFL>LFD;URF>UBR]+EAll3[FL>FU>RU]', (" F "," U "," F'"," U'"))
        self._add_myperm2('C4[DLF>BUL;UFL>FUR;ULB>FDL;URF>UFL]+EAll3[FL>FU>LU]', (" F "," U'"," F'"," U "))
        self._add_myperm2('C4[DFR>RUB;UBR>RDF;UFL>ULB;ULB>LUF]+EAll3[RF>LU>BU]', (' U2', ' R ', " U'", " R'", " U'"))
        self._add_myperm2('C4[DFR>LBU;UFL>URF;ULB>FRD;URF>FLU]+EAll3[RF>LU>FU]', (' U2', ' R ', ' U ', " R'", ' U '))
        self._add_myperm2('C4[DFR>BUL;UFL>FUR;ULB>RDF;URF>UFL]+EAll3[RF>FU>LU]', (" U'", ' R ', " U'", " R'", ' U2'))
        self._add_myperm2('C4[DFR>BRU;UBR>FRD;UFL>LBU;ULB>UFL]+EAll3[RF>BU>LU]', (' U ', ' R ', ' U ', " R'", ' U2'))

        self._add_myperm2('C5[DFR>RFU>UBR>LFD>UFL]+EAll3[FL>FR>UR]', (' F2', " R'", ' F2', ' R '))
        self._add_myperm2('C5[DFR>UFL>LFD>UBR>RFU]+EAll3[FL>UR>FR]', (" R'", ' F2', ' R ', ' F2'))
        self._add_myperm2('C5[DFR>RUB>UFL>LFD>RFU]+EAll3[FL>FR>RU]', (" F "," U "," F2"," U'"," F ") )
        self._add_myperm2('C5[DFR>RFU>BUL>LFD>UFL]+EAll3[FL>FR>LU]', (" F "," U'"," F2"," U "," F ")     )
        self._add_myperm2('C5[DFR>RFU>LFD>UFL>RUB]+EAll3[FL>RU>FR]', (" F'"," U "," F2"," U'"," F'") )
        self._add_myperm2('C5[DFR>UFL>LFD>BUL>RFU]+EAll3[FL>LU>FR]', (" F'"," U'"," F2"," U "," F'") )

        if self.size >= 4:
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v01', (" F ","2U "," F'","2U'"))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v02', (" F'", '2U ', ' F ', "2U'"))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v03', ("2U "," F'","2U'"," F "))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v04', ('2U ', ' F ', "2U'", " F'"))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v05', (" F2","2U "," F'","2U'"," F'"))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v06', (' F2', '2U ', ' F ', "2U'", ' F '))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v07', (' R ', "2U'", ' R ', '2U ', ' R2'))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v08', (" F'", "2D'", " F'", '2D ', ' F2'))

            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v01', (" F2","2U "," F2","2U'") )
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v02', (' F2', '2U ', ' F2', "2U'"))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v03', ("2D'", ' F2', '2D ', ' F2'))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v04', ("2D'", ' F2', '2D ', ' F2'))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v05', (" F ","2U "," F2","2U'"," F ") )
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v06', (' R ', "2U'", ' R2', '2U ', ' R '))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v07', (' R ', "2U'", ' R2', '2U ', ' R '))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v08', (' F ', '2U ', ' F2', "2U'", ' F '))

            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-6p[3x2]~v01', (' R ', '2U2', " R'", '2U2'))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-6p[3x2]~v02', ('2U2', ' L ', '2U2', " L'"))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-6p[3x2]~v03', (' R2', '2D2', " R'", '2D2', " R'"))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-6p[3x2]~v04', (' L ', '2U2', ' L ', '2U2', ' L2'))

            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-6p[3x2]~v01', (' R2', '2U2', ' R2', '2U2'))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-6p[3x2]~v02', ('2U2', ' L2', '2U2', ' L2'))
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-6p[3x2]~v03', (" L'", '2B2', ' L2', '2B2', " L'"))

        if self.size % 2 == 1:
            self._add_myperm2('CtrPlus12p[3x4]+ME5[BR>UF>FD>FL>RF]', (' F ', " E'", " F'", ' E '))
            self._add_myperm2('CtrPlus12p[3x4]+ME5[BR>DR>RU>FL>RF]', (' E ', " R'", " E'", ' R '))
            self._add_myperm2('CtrPlus12p[3x4]+ME5[BR>RF>RU>DR>LF]', (' R2', ' E ', " R'", " E'", " R'"))
            self._add_myperm2('CtrPlus12p[3x4]+ME5[BR>DF>FU>FR>LF]', (" F'", " E'", " F'", ' E ', ' F2'))

            self._add_myperm2('CtrPlus12p[3x4]+ME6[BR>DR>UL;DL>FR>UR]', (' R ', ' S2', " R'", ' S2'))
            self._add_myperm2('CtrPlus12p[3x4]+ME6[BR>UR>FL;DR>BL>FR]', (' E2', ' R ', ' E2', " R'"))
            self._add_myperm2('CtrPlus12p[3x4]+ME6[BR>UR>BL;DR>FL>FR]', (' R2', ' E2', ' R ', ' E2', ' R '))
            self._add_myperm2('CtrPlus12p[3x4]+ME6[BR>DR>DL;RF>RU>LU]', (" R'", ' S2', " R'", ' S2', ' R2'))



    def _register_myperms2_midedge_general(self):
        """奇数サイズ向けのMidEdge系手順を登録する。"""
        # 命名メモ:
        # - MidEdge3-* は midedge 3個の cycle。
        # - MidEdge4-* は midedge 4個の cycle / 2-2 swap 型。
        # - family 文字は位置関係、末尾 A/B/C... は向き違い。
        # - MidEdgeFlip2/4-* は midedge の flip 用 family。
        if self.size % 2 == 1:           
            self._add_myperm2('ME3[FL>BL>FR]', (' E ', ' F2', " E'", ' F2'))
            self._add_myperm2('ME3[FL>BL>RF]', (' B ', " E'", " B'", ' L2', ' B ', ' E ', " B'", ' L2'))
            self._add_myperm2('ME3[FL>LB>RF]', (' L ', ' S ', ' L ', ' F2', " L'", " S'", ' L ', ' F2', ' L2'))
            self._add_myperm2('ME3[FL>LB>FR]', (' F2', ' R ', ' E ', " R'", ' F2', ' R ', " E'", " R'"))

            self._add_myperm2('ME4[BR>RF;FL>LF;LB>RB;RF>LB]', (' E2', ' B2', ' E2', " B'", " E'", " B'", ' L2', ' B ', ' E ', " B'", ' L2'))
            self._add_myperm2('ME4[BR>RF;FL>LF;LB>BR;RF>BL]', (" F'", " E'", ' F ', ' R2', " F'", ' E ', ' F ', " E'", " F'", " E'", ' F ', ' R2', " F'", ' E ', " F'", ' E ', ' F2'))
            self._add_myperm2('ME4[BR>FR;FL>LF;LB>BR;RF>LB]', (' F2', ' E ', ' F2', ' B2', " L'", ' E ', ' L ', ' B2', " L'", " E'", ' L ', " E'"))
            self._add_myperm2('ME4[BR>FR;FL>LF;LB>RB;RF>BL]', (' E ', ' B2', ' L ', ' E ', " L'", ' B2', ' L ', " E'", ' L ', " E'", ' L2'))


            self._add_myperm2('ME4[DF>UF;FL>FR]', (' E2', ' F ', ' E2', ' F2', ' E2', ' F ', ' E2'))
            self._add_myperm2('ME4s[DF<>UF;FL<>RF]', (' E2', " F'", ' E ', ' F2', " E'", " F'", ' E ', ' F ', ' L2', " F'", " E'", ' F ', ' L2', " F'", ' E2'))
            self._add_myperm2('ME4[DF>UF;FL>RF;RF>LF;UF>FD]', (' M2', " F'", " M'", ' F2', ' D2', ' M ', " F'", " M'", ' F ', ' D2', " F'", " M'"))
            self._add_myperm2('ME4[DF>FU;FL>RF]', (" E'", ' F ', ' L2', " F'", " E'", ' F ', ' E2', ' L2', ' E2', ' L2', " F'", ' E ', ' F ', ' L2', " F'", ' E '))

            self._add_myperm2('ME4[DB>UB;FL>FR]', (" F'", ' M2', ' F2', ' M2', " F'"))
            self._add_myperm2('ME4s[DB<>UB;FL<>RF]', (' B ', ' E ', ' B2', " E'", ' B ', ' E ', " B'", ' R2', ' B ', " E'", " B'", ' R2', ' B '))
            self._add_myperm2('ME4[DB>UB;FL>RF;RF>LF;UB>BD]', (" M'", " F'", ' D2', ' F ', ' M ', " F'", " M'", ' D2', ' F2', ' M ', " F'"))
            self._add_myperm2('ME4[DB>BU;FL>RF]', (" M'", ' F ', ' D2', " F'", ' M2', ' F ', ' U2', " F'", " M'", ' F ', ' D2', ' U2', " F'"))



            self._add_myperm2('ME3[FL>FR>FU]', (' M2', ' F ', " M'", ' F2', ' M ', ' F ', ' M2'))
            self._add_myperm2('ME3[FL>FR>UF]', (" M'", " F'", ' M ', ' F2', " M'", " F'", ' M '))
            self._add_myperm2('ME3[FL>RF>FU]', (' E2', ' F ', ' R2', " F'", " E'", ' F ', ' R2', " F'", " E'"))
            self._add_myperm2('ME3[FL>RF>UF]', (" E'", " F'", ' L2', ' F ', " E'", " F'", ' L2', ' F ', ' E2'))

            self._add_myperm2('ME3[FL>FR>BU]', (" F'", ' M ', ' F2', " M'", " F'"))
            self._add_myperm2('ME3[FL>FR>UB]', (" M'", ' F ', " M'", ' F2', ' M ', ' F ', ' M '))
            self._add_myperm2('ME3[FL>RF>BU]', (' B ', ' L2', " B'", " E'", ' B ', ' L2', " B'", ' E '))
            self._add_myperm2('ME3[FL>RF>UB]', (' E ', " B'", ' R2', ' B ', " E'", " B'", ' R2', ' B '))


            self._add_myperm2('ME4[DF>UF;FL>LF;RF>DF;UF>FR]', (" E'", " F'", ' L2', ' F ', " E'", " F'", ' L2', ' E2', ' F2', ' E2', " F'", ' E2'))
            self._add_myperm2('ME4[DF>UF;FL>LF;RF>FD;UF>RF]', (' E2', " F'", ' E2', ' F2', ' E2', ' L2', " F'", ' E ', ' F ', ' L2', " F'", ' E '))
            self._add_myperm2('ME4[DF>FU;FL>LF;RF>DF;UF>RF]', (' M ', " F'", ' D2', ' F ', ' M ', " F'", ' D2', ' M ', ' F2', " M'", " F'", ' M2'))
            self._add_myperm2('ME4[DF>FU;FL>LF;RF>FD;UF>FR]', (' M2', ' F ', ' U2', " F'", ' M ', ' F ', ' U2', " F'", ' M2', ' F ', " M'", ' F2', ' M ', ' F ', " M'"))


            self._add_myperm2('ME4[DB>UB;FL>LF;RF>DB;UB>FR]', (' E ', " B'", ' R2', ' B ', " E'", " B'", ' R2', ' E2', ' B2', ' E2', " B'"))
            self._add_myperm2('ME4[DB>UB;FL>LF;RF>BD;UB>RF]', (" B'", ' E2', ' B2', ' E2', ' R2', " B'", ' E ', ' B ', ' R2', " B'", " E'"))
            self._add_myperm2('ME4[DB>BU;FL>LF;RF>DB;UB>RF]', (' M ', ' F ', ' U2', " F'", " M'", ' F ', ' U2', " M'", ' F2', ' M ', ' F '))
            self._add_myperm2('ME4[DB>BU;FL>LF;RF>BD;UB>FR]', (" M'", " F'", " M'", ' F2', ' M ', " F'", ' M2', ' F ', ' U2', " F'", " M'", ' F ', ' U2', " F'"))

            self._add_myperm2('ME3[FL>FR>LU]', (' E ', " R'", ' S ', ' R2', " S'", " R'", " E'"))
            self._add_myperm2('ME3[FL>FR>UL]', (' R ', ' S ', ' R2', " S'", " R'", ' E ', ' R2', " E'"))
            self._add_myperm2('ME3[FL>RF>LU]', (' E ', ' L ', ' F2', " L'", " E'", ' L ', ' F2', " L'"))
            self._add_myperm2('ME3[FL>RF>UL]', (' L ', " F'", ' M ', ' F ', " L'", " F'", " M'", ' F '))
            self._add_myperm2('ME3[FL>LU>FR]', (' E ', ' R ', ' S ', ' R2', " S'", ' R ', " E'"))
            self._add_myperm2('ME3[FL>UL>FR]', (' E ', ' R2', " E'", ' R ', ' S ', ' R2', " S'", " R'"))
            self._add_myperm2('ME3[FL>LU>RF]', (' L ', ' F2', " L'", ' E ', ' L ', ' F2', " L'", " E'"))
            self._add_myperm2('ME3[FL>UL>RF]', (" F'", ' M ', ' F ', ' L ', " F'", " M'", ' F ', " L'"))

            self._add_myperm2('ME3[BR>FL>UR]', (' B2', " L'", " S'", ' L2', ' S ', " L'", ' B2'))
            self._add_myperm2('ME3[BR>FL>RU]', (' E2', ' F ', ' U ', " F'", ' E2', ' F ', " U'", " F'"))
            self._add_myperm2('ME3[BR>LF>UR]', (' B2', ' R ', ' B2', " R'", " E'", ' R ', ' B2', " R'", ' E ', ' B2'))
            self._add_myperm2('ME3[BR>LF>RU]', (" L'", " S'", " L'", ' E ', ' L2', " E'", " L'", ' S ', ' L '))
            self._add_myperm2('ME3[BR>UR>FL]', (' B2', ' L ', " S'", ' L2', ' S ', ' L ', ' B2'))
            self._add_myperm2('ME3[BR>RU>FL]', (' F ', ' U ', " F'", ' E2', ' F ', " U'", " F'", ' E2'))
            self._add_myperm2('ME3[BR>UR>LF]', (' B2', " E'", ' R ', ' B2', " R'", ' E ', ' R ', ' B2', " R'", ' B2'))
            self._add_myperm2('ME3[BR>RU>LF]', (" L'", " S'", ' L ', ' E ', ' L2', " E'", ' L ', ' S ', ' L '))


            self._add_myperm2('ME3[RF>FU>LU]', (' U ', " L'", ' E2', ' L ', " U'", " L'", ' E2', ' L '))
            self._add_myperm2('ME3[RF>UF>LU]', (" R'", ' S2', ' R ', ' F ', " R'", ' S2', ' R ', " F'"))
            self._add_myperm2('ME3[RF>UF>UL]', (' R ', ' S ', " R'", ' F ', ' R ', " S'", " R'", " F'"))
            self._add_myperm2('ME3[RF>FU>UL]', (' E ', " L'", ' B ', " M'", ' B2', ' M ', ' B ', ' L ', " E'"))

            
            self._add_myperm2('ME3[RF>BU>LU]', (" U'", " L'", ' E2', ' L ', ' U ', " L'", ' E2', ' L '))
            self._add_myperm2('ME3[RF>UB>LU]', (' L ', " F'", ' M ', ' F2', " M'", " F'", " L'"))
            self._add_myperm2('ME3[RF>UB>UL]', (' B ', ' E2', " B'", ' U ', ' B ', ' E2', " B'", " U'"))
            self._add_myperm2('ME3[RF>BU>UL]', (" S'", " R'", ' D ', " M'", ' D2', ' M ', ' D ', ' R ', ' S '))


            self._add_myperm2('ME3[RF>FU>UR]', (" S2"," L'"," E "," R "," U'"," R'"," E'"," R "," U "," R'"," L "," S2"))
            self._add_myperm2('ME3[RF>UF>RU]', (' E2', ' R ', " B'", " M'", ' B2', ' M ', " B'", " R'", ' E2'))
            self._add_myperm2('ME3[DR>LF>BU]', (' L ', ' S ', " U'", ' L2', ' U ', " S'", " U'", ' L2', ' U ', " L'"))
            self._add_myperm2('ME3[DR>FL>BU]', (' B ', " L'", ' S ', ' L2', " S'", " L'", " B'"))


            self._add_myperm2('ME4[DF>LF;RF>UF]', (' E2', " F'", ' R2', ' E2', ' R2', ' E2', ' F ', ' E2'))
            self._add_myperm2('ME4[DF>FL;FL>FD;RF>UF;UF>FR]', (' E2', ' F ', ' E ', ' F2', " E'", ' F ', " E'", " F'", ' R2', ' F ', ' E ', " F'", ' R2', ' F ', ' E2'))
            self._add_myperm2('ME4[DF>LF;FL>DF;RF>UF;UF>FR]', (' E2', " F'", " E'", ' R2', ' F2', ' E ', " F'", " E'", " F'", ' R2', ' F ', " E'"))
            self._add_myperm2('ME4[DF>LF;RF>FU]', (' E2', " F'", " E'", ' F2', ' E ', " F'", " E'", " F'", " E'", ' F2', ' E ', " F'", " E'"))
            self._add_myperm2('ME4[DF>FL;RF>FU]', (' M ', " F'", " M'", ' F2', ' M ', " F'", ' M2', ' F ', ' M ', ' F2', " M'", ' F ', ' M '))


            self._add_myperm2('ME4[DB>LF;RF>UB]', (" B'", ' R2', ' E2', ' R2', ' E2', ' B '))
            self._add_myperm2('ME4[DB>FL;FL>BD;RF>UB;UB>FR]', (' B ', ' E ', ' B2', " E'", ' B ', " E'", " B'", ' L2', ' B ', ' E ', " B'", ' L2', ' B '))
            self._add_myperm2('ME4[DB>LF;FL>DB;RF>UB;UB>FR]', (' F ', " M'", ' F2', ' M ', ' F ', " E'", ' B ', " E'", ' B2', ' E ', ' B ', ' E '))
            self._add_myperm2('ME4[DB>LF;RF>BU]', (" E'", ' B ', " E'", ' B2', ' E2', ' L2', " E'", ' B ', ' E ', " B'", ' L2', ' B '))
            self._add_myperm2('ME4[DB>FL;RF>BU]', (" E'", ' B ', " E'", ' B2', ' E ', ' B ', ' E2', " B'", ' E ', ' B2', " E'", " B'", " E'"))


            self._add_myperm2('ME4[BR>BL;FL>FR]', (' E2', ' F2', ' E2', ' F2'))
            self._add_myperm2('ME4[BR>BL;FL>RF]', (' E ', ' B2', ' L ', ' E ', " L'", ' B2', ' L ', " E'", " L'", ' B2', " E'", ' B2'))
            self._add_myperm2('ME4[BR>BL;FL>RF;LB>BR;RF>LF]', (' B2', ' E ', " L'", ' E ', ' L ', ' B2', " L'", " E'", ' L ', ' R2', " E'", ' R2'))
            self._add_myperm2('ME4[BR>BL;FL>FR;LB>BR;RF>FL]', (" E'", " L'", " E'", ' L ', ' F2', " L'", ' E ', ' L ', ' R2', ' E ', ' R2', ' F2'))
            self._add_myperm2('ME4s[BR<>LB;FL<>RF]', (' R2', " B'", ' E ', ' B ', ' R2', " B'", " E'", ' B ', ' F ', " E'", " F'", ' R2', ' F ', ' E ', " F'", ' R2'))

            self._add_myperm2('ME4s[BR<>FL;LB<>RF]', (' R2', ' E2', ' F2', ' E2', ' F2', ' R2'))
            self._add_myperm2('ME4[BR>LF;LB>RF]', (" R'", " E'", ' R ', ' B2', " R'", ' E ', ' R ', ' B2', ' E ', ' B2', " E'", ' B2'))
            self._add_myperm2('ME4[BR>FL;FL>RB;LB>FR;RF>LB]', (" E'", ' R2', ' F ', " E'", " F'", ' R2', ' F ', ' E ', " F'", ' L2', ' E ', ' L2'))
            self._add_myperm2('ME4[BR>LF;LB>FR]', (' L ', ' E ', " L'", ' B2', ' L ', " E'", " L'", ' R ', " E'", " R'", ' B2', ' R ', ' E ', " R'"))
        

            self._add_myperm2('ME4[BR>BU;DF>LF]', (' D2', " B'", " E'", ' B2', ' E ', " B'", ' D2', ' U2', " F'", " E'", ' F2', ' E ', " F'", ' U2'))
            self._add_myperm2('ME4[BR>BU;DF>FL;FL>FD;UB>BR]', (" F'", " E'", ' F2', ' E ', " F'", ' M ', ' F2', " M'", ' F2', ' E ', ' F ', ' L2', " F'", ' E ', ' F ', " E'", ' L2', ' F2', ' E ', ' F ', ' E2'))
            self._add_myperm2('ME4[BR>BU;DF>LF;FL>DF;UB>BR]', (' E ', " F'", ' M ', ' F2', " M'", " F'", ' L2', " E'", ' L2', ' D2', " B'", " E'", ' B2', ' E ', " B'", ' D2'))
            self._add_myperm2('ME4[BR>UB;DF>LF]', (" E'", " F'", ' M ', ' F2', " M'", ' F ', ' E ', ' F2', ' M2', " F'", ' M ', ' F2', " M'", " F'", ' M2'))
            self._add_myperm2('ME4s[BR<>UB;DF<>FL]', (" E'", " B'", ' E ', ' B2', " E'", " B'", ' E ', ' M ', " B'", " E'", ' B2', ' E ', ' B ', " M'", ' B2'))
            
            self._add_myperm2('ME4[BR>FU;DB>LF]', (' F ', " E'", ' F2', ' E ', ' F ', ' M2', ' B2', ' M2', " B'", " E'", ' B2', ' E ', ' B '))
            self._add_myperm2('ME4[BR>FU;DB>FL;FL>BD;UF>BR]', (' D2', ' B ', ' U2', " B'", " M'", ' B ', ' U2', " B'", ' M ', ' D2', " F'", " M'", ' F2', ' M ', " F'", " E'", ' F2', ' E ', ' F2'))
            self._add_myperm2('ME4[BR>FU;DB>LF;FL>DB;UF>BR]', (" M'", ' F ', ' M ', ' F2', " M'", ' F ', ' M ', " E'", ' F2', ' E ', ' F2', ' U2', " B'", " E'", ' B2', ' E ', " B'", ' U2'))
            self._add_myperm2('ME4[BR>UF;DB>LF]', (' E ', " B'", " M'", ' B2', ' M ', " B'", ' R2', " E'", ' F ', " M'", ' F2', ' M ', ' F ', ' R2'))
            self._add_myperm2('ME4s[BR<>UF;DB<>FL]', (' M ', " B'", ' M ', ' B2', " M'", " B'", ' F ', ' M ', ' F2', " M'", ' F ', " M'", ' E2', ' B2', ' E2', ' B2'))

            self._add_myperm2('ME4[FL>FR;UL>UR]', (' E ', ' R2', " E'", ' R2', " S'", " R'", ' S2', ' R2', ' S2', " R'", ' S '))
            self._add_myperm2('ME4[FL>RF;RF>LF;UL>RU;UR>UL]', (' E2', " L'", ' B2', ' L ', " E'", " L'", ' B2', ' L ', " E'", ' S ', " R'", ' E ', ' R2', " E'", " R'", " S'"))
            self._add_myperm2('ME4[FL>FR;RF>FL;UL>RU;UR>UL]', (' L2', ' B2', ' D2', ' E2', " R'", " E'", ' R2', ' B2', ' E ', " R'", " E'", ' R ', ' B2', " R'", " E'", ' D2', ' B2', ' L2'))
            self._add_myperm2('ME4[FL>FR;UL>RU]', (' E ', ' R2', " E'", ' R2', ' E2', ' R ', ' E2', ' R2', ' E2', ' R ', ' E2', ' D ', ' S ', " D'", ' R2', ' D ', " S'", " D'", ' R2'))
            self._add_myperm2('ME4[FL>RF;UL>RU]', (' L2', ' B2', ' D2', ' S ', ' R ', ' D2', " R'", ' S ', ' R ', ' S2', ' D2', ' S2', ' D2', " R'", " S'", ' R ', ' D2', " R'", " S'", ' D2', ' B2', ' L2'))

            self._add_myperm2('ME4s[DB<>UF;LB<>RF]', (' D2', ' L2', ' E2', ' F ', ' E2', ' F2', ' E2', ' F ', ' E2', ' L2', ' D2'))
            self._add_myperm2('ME4[DB>FU;LB>FR;RF>LB;UF>DB]', (' D2', ' L2', ' M2', " F'", " M'", ' F2', ' D2', ' M ', " F'", " M'", ' F ', ' D2', " F'", " M'", ' L2', ' D2', ' M ', ' U ', ' M ', " U'", ' B2', ' U ', " M'", ' U ', " M'", ' U2', ' B2'))
            self._add_myperm2('ME4[DB>UF;LB>FR;RF>LB;UF>BD]', (' D2', ' L2', ' M2', " F'", " M'", ' F2', ' D2', ' M ', " F'", " M'", ' F ', ' D2', " F'", " M'", ' L2', ' D2'))
            self._add_myperm2('ME4[DB>UF;LB>FR]', (' D2', ' L2', ' E2', " F'", ' E ', ' F2', " E'", " F'", ' E ', ' F ', ' L2', " F'", " E'", ' F ', ' L2', " F'", ' E2', ' L2', ' D2'))
            self._add_myperm2('ME4[DB>FU;LB>FR]', (' D2', ' L2', " E'", ' F ', ' L2', " F'", " E'", ' F ', ' E2', ' L2', ' E2', ' L2', " F'", ' E ', ' F ', ' L2', " F'", ' E ', ' L2', ' D2'))

            

            self._add_myperm2('ME2[FL>LF;RF>FR]~v01', (' F2', " E'", ' F ', " E'", ' F ', ' R2', " F'", ' E ', ' F ', ' E ', ' R2'))
            self._add_myperm2('ME2[RF>FR;UF>FU]~v01', (' E ', ' F ', " E'", ' F2', ' L2', ' E ', ' F ', ' E ', " F'", ' L2', ' F ', ' E2'))
            self._add_myperm2('ME2[FL>LF;UB>BU]~v01', (' E ', " B'", ' E ', ' B2', ' L2', " E'", " B'", " E'", ' B ', ' L2', " B'"))
            self._add_myperm2('ME2[BR>RB;FL>LF]~v01', (' E ', ' R ', ' E ', " R'", ' F2', ' R ', " E'", ' R ', " E'", ' R2', ' F2'))
            self._add_myperm2('ME4[DF>FD;FL>LF;RF>FR;UF>FU]~v01', (' E ', " F'", " E'", ' F2', ' E ', " F'", " E'", ' M2', " F'", " M'", ' F2', ' M ', ' U2', " F'", ' M ', ' F ', ' U2', " F'", ' M '))
            self._add_myperm2('ME4[BR>RB;FL>LF;LB>BL;RF>FR]~v01', (' R2', " E'", " R'", " E'", " R'", ' B2', ' R ', ' E ', " R'", ' E2', " L'", ' E ', ' L ', ' B2', " L'", " E'", " L'", " E'", ' L2'))
            self._add_myperm2('ME4[DB>BD;FL>LF;RF>FR;UB>BU]~v01', (" E'", ' B ', " E'", ' B2', ' E ', ' B ', ' E2', " B'", ' R2', ' B ', " E'", " B'", ' R2', ' E ', ' F2', " E'", ' F2', ' B '))


            self._add_myperm2('ME2[FL>LF;RF>FR]~v02', (' R2', " E'", " F'", " E'", ' F ', ' R2', " F'", ' E ', " F'", ' E ', ' F2'))
            self._add_myperm2('ME2[RF>FR;UF>FU]~v02', (' E2', " F'", ' L2', ' F ', " E'", " F'", " E'", ' L2', ' F2', ' E ', " F'", " E'"))
            self._add_myperm2('ME2[FL>LF;UB>BU]~v02', (' B ', ' L2', " B'", ' E ', ' B ', ' E ', ' L2', ' B2', " E'", ' B ', " E'"))
            self._add_myperm2('ME2[BR>RB;FL>LF]~v02', self.invert_moves(self.myperms2['ME2[BR>RB;FL>LF]~v01']))
            self._add_myperm2('ME4[DF>FD;FL>LF;RF>FR;UF>FU]~v02', (" M'", ' F ', ' U2', " F'", " M'", ' F ', ' U2', " M'", ' F2', ' M ', ' F ', ' M2', ' E ', ' F ', " E'", ' F2', ' E ', ' F ', " E'"))
            self._add_myperm2('ME4[BR>RB;FL>LF;LB>BL;RF>FR]~v02', (' L2', ' E ', ' L ', ' E ', ' L ', ' B2', " L'", " E'", ' L ', ' E2', ' R ', " E'", " R'", ' B2', ' R ', ' E ', ' R ', ' E ', ' R2'))
            self._add_myperm2('ME4[DB>BD;FL>LF;RF>FR;UB>BU]~v02', (" B'", ' F2', ' E ', ' F2', " E'", ' R2', ' B ', ' E ', " B'", ' R2', ' B ', ' E2', " B'", " E'", ' B2', ' E ', " B'", ' E '))
            








            

    def _register_myperms2_edge_general(self):
        """4x4以上で使うEdge系・派生手順を登録する。"""
        # 命名メモ:
        # - Wing3Cycle-* は wing 3個の cycle。
        # - Parallel3 / MidEdge3 / Parallel2Plus1 / SameEdgePairPlus1 は
        #   3つの wing の位置関係 family。
        # - WingSwapParallel / WingSwapSkew / WingSwapSkewViaEdge は
        #   wing 2点交換 family。
        # - CornerEdgeBlockSwap-* は corner 2つ + edge block 2つの同時 swap。
        if self.size >= 4:
            






            
        

            self._add_myperm2('W2-3[FL@U>LB@U>RF@U]~v01', (' F2', ' L2', ' F ', "2L'", " F'", ' L2', ' F ', '2L ', ' F '))
            self._add_myperm2('W2-3[FL@U>RF@U>LB@U]~v01', (" F'", "2L'", " F'", ' L2', ' F ', '2L ', " F'", ' L2', ' F2'))
            self._add_myperm2('W2-3[FL@U>LB@U>RF@U]~v02', (' F2', ' L2', " F'", "2R'", ' F ', ' L2', " F'", '2R ', " F'"))
            self._add_myperm2('W2-3[FL@U>RF@U>LB@U]~v02', (' F ', "2R'", ' F ', ' L2', " F'", '2R ', ' F ', ' L2', ' F2'))

            

            

            self._add_myperm2('W2-3[BR@D>RF@U>FL@U]~v01', (' R2', ' F ', "2R'", " F'", ' R2', ' F ', '2R ', " F'"))
            self._add_myperm2('W2-3[BR@D>FL@U>RF@U]~v01', (' F ', "2R'", " F'", ' R2', ' F ', '2R ', " F'", ' R2'))
            self._add_myperm2('W2-3[BR@D>RF@U>FL@U]~v02', (' R2', " F'", "2L'", ' F ', ' R2', " F'", '2L ', ' F '))
            self._add_myperm2('W2-3[BR@D>FL@U>RF@U]~v02', (" F'", "2L'", ' F ', ' R2', " F'", '2L ', ' F ', ' R2'))
            self._add_myperm2('W2-3[BR@D>RF@U>FL@U]~v03', (" R'", '2B ', " L'", "2B'", ' R2', '2B ', ' L ', "2B'", " R'"))
            self._add_myperm2('W2-3[BR@D>FL@U>RF@U]~v03', (' R ', '2B ', " L'", "2B'", ' R2', '2B ', ' L ', "2B'", ' R '))
            self._add_myperm2('W2-3[BR@D>RF@U>FL@U]~v04', (' R ', '2F ', ' L ', "2F'", ' R2', '2F ', " L'", "2F'", ' R '))
            self._add_myperm2('W2-3[BR@D>FL@U>RF@U]~v04', (" R'", '2F ', ' L ', "2F'", ' R2', '2F ', " L'", "2F'", " R'"))
 
            


            self._add_myperm2('W2-3[FL@D>LB@U>RF@U]~v01', ('2U2', ' B2', "2U'", ' B2', '2D ', ' F2', '2U2', ' F2', ' L2', "2D'", ' L2', "2U'"))
            self._add_myperm2('W2-3[FL@D>LB@U>RF@U]~v02', ("2D'", ' L2', '2D ', ' F2', '2D ', ' F2', "2D'", ' F2', "2U'", ' F2', '2U ', ' L2'))


            self._add_myperm2('W2-3[FL@D>UR@F>RF@U]~v01', (' F2', " L'", "2F'", ' L ', ' F2', " L'", '2F ', ' L '))
            self._add_myperm2('W2-3[FL@D>RF@U>UR@F]~v01', (" L'", "2F'", ' L ', ' F2', " L'", '2F ', ' L ', ' F2'))
            self._add_myperm2('W2-3[FL@D>RF@U>UL@B]~v01', (' F2', ' R ', "2B'", " R'", ' F2', ' R ', '2B ', " R'"))
            self._add_myperm2('W2-3[FL@D>UL@B>RF@U]~v01', (' R ', "2B'", " R'", ' F2', ' R ', '2B ', " R'", ' F2'))
            self._add_myperm2('W2-3[FL@D>UR@B>RF@U]~v01', (' F2', ' L ', '2B2', " L'", ' F2', ' L ', '2B2', " L'"))
            self._add_myperm2('W2-3[FL@D>RF@U>UR@B]~v01', (' L ', '2B2', " L'", ' F2', ' L ', '2B2', " L'", ' F2'))
            self._add_myperm2('W2-3[FL@D>RF@U>UL@F]~v01', (' F2', " R'", '2F2', ' R ', ' F2', " R'", '2F2', ' R '))
            self._add_myperm2('W2-3[FL@D>UL@F>RF@U]~v01', (" R'", '2F2', ' R ', ' F2', " R'", '2F2', ' R ', ' F2'))

            self._add_myperm2('W2-3[FL@D>UR@F>RF@U]~v02', (' F ', " D'", '2F2', ' D ', ' F2', " D'", '2F2', ' D ', ' F '))
            self._add_myperm2('W2-3[FL@D>RF@U>UR@F]~v02', (" F'", " D'", '2F2', ' D ', ' F2', " D'", '2F2', ' D ', " F'"))
            self._add_myperm2('W2-3[FL@D>RF@U>UL@B]~v02', (" F'", ' D ', '2B2', " D'", ' F2', ' D ', '2B2', " D'", " F'"))
            self._add_myperm2('W2-3[FL@D>UL@B>RF@U]~v02', (' F ', ' D ', '2B2', " D'", ' F2', ' D ', '2B2', " D'", ' F '))
            self._add_myperm2('W2-3[FL@D>UR@B>RF@U]~v02', (' F ', ' D ', "2B'", " D'", ' F2', ' D ', '2B ', " D'", ' F '))
            self._add_myperm2('W2-3[FL@D>RF@U>UR@B]~v02', (" F'", ' D ', "2B'", " D'", ' F2', ' D ', '2B ', " D'", " F'"))
            self._add_myperm2('W2-3[FL@D>RF@U>UL@F]~v02', (" F'", " D'", "2F'", ' D ', ' F2', " D'", '2F ', ' D ', " F'"))
            self._add_myperm2('W2-3[FL@D>UL@F>RF@U]~v02', (' F ', " D'", "2F'", ' D ', ' F2', " D'", '2F ', ' D ', ' F '))



            self._add_myperm2('W2-3[DF@L>RF@U>FL@U]', (" R'", ' D ', ' R ', "2U'", " R'", " D'", ' R ', '2U '))
            self._add_myperm2('W2-3[DF@L>FL@U>RF@U]', ("2U'", " R'", ' D ', ' R ', '2U ', " R'", " D'", ' R '))
            self._add_myperm2('W2-3[FL@U>UF@R>RF@U]', (' R ', " U'", " R'", "2U'", ' R ', ' U ', " R'", '2U '))
            self._add_myperm2('W2-3[FL@U>RF@U>UF@R]', ("2U'", ' R ', " U'", " R'", '2U ', ' R ', ' U ', " R'"))


            self._add_myperm2('W2-3[DB@L>FL@U>RF@U]', (' L ', ' D ', " L'", '2U ', ' L ', " D'", " L'", "2U'"))
            self._add_myperm2('W2-3[DB@L>RF@U>FL@U]', ('2U ', ' L ', ' D ', " L'", "2U'", ' L ', " D'", " L'"))
            self._add_myperm2('W2-3[FL@U>RF@U>UB@R]', (" L'", " U'", ' L ', '2U ', " L'", ' U ', ' L ', "2U'"))
            self._add_myperm2('W2-3[FL@U>UB@R>RF@U]', ('2U ', " L'", " U'", ' L ', "2U'", " L'", ' U ', ' L '))
            
            self._add_myperm2('W2-3[FL@D>UF@L>RF@U]~v01', (" U'", ' F2', " L'", "2F'", ' L ', ' F2', " L'", '2F ', ' L ', ' U '))
            self._add_myperm2('W2-3[FL@D>RF@U>UF@L]~v01', (' U ', ' F2', ' R ', "2B'", " R'", ' F2', ' R ', '2B ', " R'", " U'"))
            self._add_myperm2('W2-3[FL@D>RF@U>UF@R]~v01', (' U ', ' F2', " R'", '2F2', ' R ', ' F2', " R'", '2F2', ' R ', " U'"))
            self._add_myperm2('W2-3[FL@D>UF@R>RF@U]~v01', (" U'", ' F2', ' L ', '2B2', " L'", ' F2', ' L ', '2B2', " L'", ' U '))
            
            self._add_myperm2('W2-3[FL@D>UF@L>RF@U]~v02', (' U ', ' R ', "2B'", " R'", ' F2', ' R ', '2B ', " R'", ' F2', " U'"))
            self._add_myperm2('W2-3[FL@D>RF@U>UF@L]~v02', (" U'", " L'", "2F'", ' L ', ' F2', " L'", '2F ', ' L ', ' F2', ' U '))
            self._add_myperm2('W2-3[FL@D>RF@U>UF@R]~v02', (" U'", ' L ', '2B2', " L'", ' F2', ' L ', '2B2', " L'", ' F2', ' U '))
            self._add_myperm2('W2-3[FL@D>UF@R>RF@U]~v02', (' U ', " R'", '2F2', ' R ', ' F2', " R'", '2F2', ' R ', ' F2', " U'"))

            self._add_myperm2('W2-3[FL@D>UB@R>RF@U]~v01', (' U ', ' F2', " L'", "2F'", ' L ', ' F2', " L'", '2F ', ' L ', " U'"))
            self._add_myperm2('W2-3[FL@D>RF@U>UB@R]~v01', (" U'", ' F2', ' R ', "2B'", " R'", ' F2', ' R ', '2B ', " R'", ' U '))
            self._add_myperm2('W2-3[FL@D>RF@U>UB@L]~v01', (" U'", ' F2', " R'", '2F2', ' R ', ' F2', " R'", '2F2', ' R ', ' U '))
            self._add_myperm2('W2-3[FL@D>UB@L>RF@U]~v01', (' U ', ' F2', ' L ', '2B2', " L'", ' F2', ' L ', '2B2', " L'", " U'"))

            self._add_myperm2('W2-3[FL@D>UB@R>RF@U]~v02', (" U'", ' R ', "2B'", " R'", ' F2', ' R ', '2B ', " R'", ' F2', ' U '))
            self._add_myperm2('W2-3[FL@D>RF@U>UB@R]~v02', (' U ', " L'", "2F'", ' L ', ' F2', " L'", '2F ', ' L ', ' F2', " U'"))
            self._add_myperm2('W2-3[FL@D>RF@U>UB@L]~v02', (' U ', ' L ', '2B2', " L'", ' F2', ' L ', '2B2', " L'", ' F2', " U'"))
            self._add_myperm2('W2-3[FL@D>UB@L>RF@U]~v02', (" U'", " R'", '2F2', ' R ', ' F2', " R'", '2F2', ' R ', ' F2', ' U '))


            self._add_myperm2('W2-3[BR@U>FL@U>UL@B]~v01', (" R'", ' U2', ' R ', '2U2', " R'", ' U2', ' R ', '2U2'))
            self._add_myperm2('W2-3[BR@U>UL@B>FL@U]~v01', ('2U2', " R'", ' U2', ' R ', '2U2', " R'", ' U2', ' R '))
            self._add_myperm2('W2-3[BR@U>FL@U>DL@F]~v01', (' R ', ' D2', " R'", '2U2', ' R ', ' D2', " R'", '2U2'))
            self._add_myperm2('W2-3[BR@U>DL@F>FL@U]~v01', ('2U2', ' R ', ' D2', " R'", '2U2', ' R ', ' D2', " R'"))
            self._add_myperm2('W2-3[BR@U>FL@U>UL@F]~v01', (' B ', ' U ', " B'", '2U2', ' B ', " U'", " B'", '2U2'))
            self._add_myperm2('W2-3[BR@U>UL@F>FL@U]~v01', ('2U2', ' B ', ' U ', " B'", '2U2', ' B ', " U'", " B'"))
            self._add_myperm2('W2-3[BR@U>FL@U>DL@B]~v01', (" B'", " D'", ' B ', '2U2', " B'", ' D ', ' B ', '2U2'))
            self._add_myperm2('W2-3[BR@U>DL@B>FL@U]~v01', ('2U2', " B'", " D'", ' B ', '2U2', " B'", ' D ', ' B '))

            self._add_myperm2('W2-3[BR@U>FL@U>UL@F]~v02', ('2U2', ' F ', " U'", " F'", '2U2', ' F ', ' U ', " F'"))
            self._add_myperm2('W2-3[BR@U>UL@F>FL@U]~v02', (' F ', " U'", " F'", '2U2', ' F ', ' U ', " F'", '2U2'))
            self._add_myperm2('W2-3[BR@U>FL@U>DL@B]~v02', ('2U2', " F'", ' D ', ' F ', '2U2', " F'", " D'", ' F '))
            self._add_myperm2('W2-3[BR@U>DL@B>FL@U]~v02', (" F'", ' D ', ' F ', '2U2', " F'", " D'", ' F ', '2U2'))

            self._add_myperm2('W2-3[BR@U>FL@U>UL@B]~v02', (' R ', '2B2', ' R ', ' F2', " R'", '2B2', ' R ', ' F2', ' R2'))
            self._add_myperm2('W2-3[BR@U>UL@B>FL@U]~v02', (' R2', ' F2', " R'", '2B2', ' R ', ' F2', " R'", '2B2', " R'"))
            self._add_myperm2('W2-3[BR@U>FL@U>DL@F]~v02', (" R'", '2F2', " R'", ' F2', ' R ', '2F2', " R'", ' F2', ' R2'))
            self._add_myperm2('W2-3[BR@U>DL@F>FL@U]~v02', (' R2', ' F2', ' R ', '2F2', " R'", ' F2', ' R ', '2F2', ' R '))
            self._add_myperm2('W2-3[BR@U>FL@U>UL@F]~v03', (" R'", '2F ', " R'", ' F2', ' R ', "2F'", " R'", ' F2', ' R2'))
            self._add_myperm2('W2-3[BR@U>UL@F>FL@U]~v03', (' R2', ' F2', ' R ', '2F ', " R'", ' F2', ' R ', "2F'", ' R '))
            self._add_myperm2('W2-3[BR@U>FL@U>DL@B]~v03', (' R ', '2B ', ' R ', ' F2', " R'", "2B'", ' R ', ' F2', ' R2'))
            self._add_myperm2('W2-3[BR@U>DL@B>FL@U]~v03', (' R2', ' F2', " R'", '2B ', ' R ', ' F2', " R'", "2B'", " R'"))



            self._add_myperm2('W2-3[DR@B>UB@L>FL@U]', (" F'", '2L2', " F'", ' R ', ' F ', '2L2', " F'", " R'", ' F2'))
            self._add_myperm2('W2-3[DB@L>RF@U>UL@F]', (" F'", '2L2', " F'", ' L ', ' F ', '2L2', " F'", " L'", ' F2'))
            self._add_myperm2('W2-3[DB@R>UL@B>RF@U]', (' U ', "2R'", " B'", ' R2', ' B ', '2R ', " B'", ' R2', ' B ', " U'"))
            self._add_myperm2('W2-3[DR@F>FL@U>UB@R]', (' D ', '2R ', " B'", ' L2', ' B ', "2R'", " B'", ' L2', ' B ', " D'"))

            self._add_myperm2('W2-3[RF@U>UR@B>UF@L]~v01', (" R'", " D'", '2L ', ' D ', " R'", " D'", "2L'", ' D ', ' R2'))
            self._add_myperm2('W2-3[RF@U>UR@B>UF@R]~v01', (" U'", " B'", "2U'", ' B ', " U'", " B'", '2U ', ' B ', ' U2'))
            self._add_myperm2('W2-3[RF@U>UR@B>UF@L]~v02', (" F'", ' L ', '2B2', " L'", " F'", ' L ', '2B2', " L'", ' F2'))
            self._add_myperm2('W2-3[RF@U>UR@B>UF@R]~v02', (" R'", ' D ', '2R2', " D'", " R'", ' D ', '2R2', " D'", ' R2'))
            self._add_myperm2('W2-3[RF@U>UR@B>UF@L]~v03', (' R2', " B'", '2L2', ' B ', " R'", " B'", '2L2', ' B ', " R'"))
            self._add_myperm2('W2-3[RF@U>UR@B>UF@R]~v03', (' U2', " L'", '2U2', ' L ', " U'", " L'", '2U2', ' L ', " U'"))
            self._add_myperm2('W2-3[RF@U>UF@R>UR@F]', (' R ', ' U ', " F'", '2U2', ' F ', " U'", " F'", '2U2', ' F ', " R'"))
            self._add_myperm2('W2-3[RF@D>UF@L>UR@B]', (' R ', ' U ', " F'", '2D2', ' F ', " U'", " F'", '2D2', ' F ', " R'"))


            self._add_myperm2('W2-3[RF@U>UF@L>UL@B]~v01', (' U ', ' L ', '2U ', " L'", " U'", ' L ', "2U'", " L'"))
            self._add_myperm2('W2-3[RF@U>UL@B>UF@L]~v01', (" L ", "2U ", " L'", " U ", " L ", "2U'", " L'", " U'"))
            self._add_myperm2('W2-3[RF@U>UF@L>UL@B]~v02', (' R ', "2B'", " R'", ' F ', ' R ', '2B ', " R'", " F'"))
            self._add_myperm2('W2-3[RF@U>UL@B>UF@L]~v02', (' F ', ' R ', "2B'", " R'", " F'", ' R ', '2B ', " R'"))
            self._add_myperm2('W2-3[RF@D>UL@B>UF@L]', (" L'", '2D2', ' L ', ' U ', " L'", '2D2', ' L ', " U'"))
            self._add_myperm2('W2-3[RF@D>UF@L>UL@B]', (' U ', " L'", '2D2', ' L ', " U'", " L'", '2D2', ' L '))
            self._add_myperm2('W2-3[RF@U>UL@F>UF@R]', (" L'", '2U2', ' L ', ' U ', " L'", '2U2', ' L ', " U'"))
            self._add_myperm2('W2-3[RF@U>UF@R>UL@F]', (' U ', " L'", '2U2', ' L ', " U'", " L'", '2U2', ' L '))
            self._add_myperm2('W2-3[RF@U>UF@R>UL@B]~v01', (" L'", " B'", ' U2', " B'", "2U'", ' B ', ' U2', " B'", '2U ', ' B2', ' L '))
            self._add_myperm2('W2-3[RF@U>UL@B>UF@R]~v01', (" L'", ' B2', "2U'", ' B ', ' U2', " B'", '2U ', ' B ', ' U2', ' B ', ' L '))
            self._add_myperm2('W2-3[RF@U>UF@R>UL@B]~v02', (" R'", ' D2', '2B ', ' D ', ' F2', " D'", "2B'", ' D ', ' F2', ' D ', ' R '))
            self._add_myperm2('W2-3[RF@U>UL@B>UF@R]~v02', (" R'", " D'", ' F2', " D'", '2B ', ' D ', ' F2', " D'", "2B'", ' D2', ' R '))




            self._add_myperm2('W2-3[FL@U>UB@R>UR@F]~v01', (' B ', '2U ', " B'", " U'", ' B ', "2U'", " B'", ' U '))
            self._add_myperm2('W2-3[FL@U>UR@F>UB@R]~v01', (" U'", ' B ', '2U ', " B'", ' U ', ' B ', "2U'", " B'"))
            self._add_myperm2('W2-3[FL@D>UB@L>UR@B]~v01', (' B ', "2D'", " B'", " U'", ' B ', '2D ', " B'", ' U '))
            self._add_myperm2('W2-3[FL@D>UR@B>UB@L]~v01', (" U'", ' B ', "2D'", " B'", ' U ', ' B ', '2D ', " B'"))
            self._add_myperm2('W2-3[FL@U>UB@R>UR@F]~v02', (' U ', ' R ', '2U2', " R'", " U'", ' R ', '2U2', " R'"))
            self._add_myperm2('W2-3[FL@U>UR@F>UB@R]~v02', (' R ', '2U2', " R'", ' U ', ' R ', '2U2', " R'", " U'"))
            self._add_myperm2('W2-3[FL@D>UB@L>UR@B]~v02', (' U ', ' R ', '2D2', " R'", " U'", ' R ', '2D2', " R'"))
            self._add_myperm2('W2-3[FL@D>UR@B>UB@L]~v02', (' R ', '2D2', " R'", ' U ', ' R ', '2D2', " R'", " U'"))
            self._add_myperm2('W2-3[FL@U>UR@F>UB@L]', (" F'", '2L2', " F'", " R'", ' F ', '2L2', " F'", ' R ', ' F2'))
            self._add_myperm2('W2-3[FL@D>UR@B>UB@R]', (" F'", '2R2', " F'", " R'", ' F ', '2R2', " F'", ' R ', ' F2'))
            self._add_myperm2('W2-3[FL@U>UB@R>UR@B]', (' B ', '2U ', ' L ', ' U2', " L'", "2U'", ' L ', ' U2', " L'", " B'"))
            self._add_myperm2('W2-3[FL@D>UB@L>UR@F]', (' B ', "2D'", ' L ', ' U2', " L'", '2D ', ' L ', ' U2', " L'", " B'"))

            self._add_myperm2('W2-3[DL@F>FL@U>RF@U]', ("2F'", " R'", '2F ', ' L ', "2F'", ' R ', '2F ', " L'"))
            self._add_myperm2('W2-3[FL@U>RF@U>UL@B]', ("2B'", ' R ', '2B ', " L'", "2B'", " R'", '2B ', ' L '))
            self._add_myperm2('W2-3[FL@U>RF@U>UR@F]', ('2F ', " R'", '2F ', ' L ', "2F'", ' R ', '2F ', " L'", '2F2'))
            self._add_myperm2('W2-3[DR@B>FL@U>RF@U]', ('2B ', ' R ', '2B ', " L'", "2B'", " R'", '2B ', ' L ', '2B2'))
            self._add_myperm2('W2-3[DL@F>RF@U>FL@U]', (' L ', "2F'", " R'", '2F ', " L'", "2F'", ' R ', '2F '))
            self._add_myperm2('W2-3[FL@U>UL@B>RF@U]', (" L'", "2B'", ' R ', '2B ', ' L ', "2B'", " R'", '2B '))
            self._add_myperm2('W2-3[FL@U>UR@F>RF@U]', ('2F2', ' L ', "2F'", " R'", '2F ', " L'", "2F'", ' R ', "2F'"))
            self._add_myperm2('W2-3[DR@B>RF@U>FL@U]', ('2B2', " L'", "2B'", ' R ', '2B ', ' L ', "2B'", " R'", "2B'"))
            self._add_myperm2('W2-3[LB@D>RF@U>UL@B]', ("2B'", ' R ', '2B ', ' L ', "2B'", " R'", '2B ', " L'"))
            self._add_myperm2('W2-3[LB@D>UR@B>RF@U]', ('2B ', ' L ', "2B'", ' R ', '2B ', " L'", "2B'", " R'"))
            self._add_myperm2('W2-3[LB@D>UL@F>RF@U]', ("2F'", " L'", "2F'", " R'", '2F ', ' L ', "2F'", ' R ', '2F2'))
            self._add_myperm2('W2-3[LB@D>RF@U>UR@F]', ('2F ', " R'", '2F ', " L'", "2F'", ' R ', '2F ', ' L ', '2F2'))
            self._add_myperm2('W2-3[LB@D>UL@B>RF@U]', (' L ', "2B'", ' R ', '2B ', " L'", "2B'", " R'", '2B '))
            self._add_myperm2('W2-3[LB@D>RF@U>UR@B]', (' R ', '2B ', ' L ', "2B'", " R'", '2B ', " L'", "2B'"))
            self._add_myperm2('W2-3[LB@D>RF@U>UL@F]', ('2F2', " R'", '2F ', " L'", "2F'", ' R ', '2F ', ' L ', '2F '))
            self._add_myperm2('W2-3[LB@D>UR@F>RF@U]', ('2F2', " L'", "2F'", " R'", '2F ', ' L ', "2F'", ' R ', "2F'"))


            self._add_myperm2('W2-3[RF@D>RF@U>UR@F]', ('2U2', "2F'", ' L ', '2F ', ' R ', "2F'", " L'", '2F ', " R'", '2U2'))
            self._add_myperm2('W2-3[FL@D>UL@B>FL@U]', ('2D2', "2B'", " R'", '2B ', " L'", "2B'", ' R ', '2B ', ' L ', '2D2'))
            self._add_myperm2('W2-3[RF@D>UR@F>RF@U]', ('2U2', ' R ', "2F'", ' L ', '2F ', " R'", "2F'", " L'", '2F ', '2U2'))
            self._add_myperm2('W2-3[FL@D>FL@U>UL@B]', ('2D2', " L'", "2B'", " R'", '2B ', ' L ', "2B'", ' R ', '2B ', '2D2'))
            self._add_myperm2('W2-3[RF@D>RF@U>UL@B]', ('2D2', "2B'", ' R ', '2B ', ' L ', "2B'", " R'", '2B ', " L'", '2D2'))
            self._add_myperm2('W2-3[FL@D>UR@F>FL@U]', ('2U2', "2F'", " L'", '2F ', " R'", "2F'", ' L ', '2F ', ' R ', '2U2'))
            self._add_myperm2('W2-3[RF@D>UL@B>RF@U]', ('2D2', ' L ', "2B'", ' R ', '2B ', " L'", "2B'", " R'", '2B ', '2D2'))
            self._add_myperm2('W2-3[FL@D>FL@U>UR@F]', ('2U2', " R'", "2F'", " L'", '2F ', ' R ', "2F'", ' L ', '2F ', '2U2'))




            #self.myperms2['OLLParity'] = ("2R'"," U2","2L "," F2","2L'"," F2","2R2"," U2","2R "," U2","2R'"," U2"," F2","2R2"," F2")

            perm_A = ("2R "," U2","2R "," U2"," F2","2R "," F2","2L'"," U2","2L "," U2","2R2")
            perm_a = self.invert_moves(perm_A)

            #perm_a = ("2L2"," U2","2R "," U2","2R'"," F2","2L "," F2"," U2","2L "," U2","2L ")
            #perm_B = ("2L2"," U2","2R "," U2","2R'"," F2","2L "," F2","2L "," U2","2R "," U2","2R'"," F2","2L "," F2")

            ("2L2"," U2","2R "," U2","2R'"," F2","2L "," F2","2L "," F2","2L'"," U2","2L "," U2","2L "," F2")




            perm_k0 = ("2R2"," B2"," D2","2R "," D2","2R'"," D2","2R2"," B2","2L "," B2","2L'"," D2","2R "," B2")
            perm_k1 = ("2L2"," U2"," B2","2L'"," B2","2L "," B2","2L2"," U2","2R'"," U2","2R "," B2","2L'"," U2")
            perm_k2 = ('2R2', ' D2', '2L ', ' U2', "2R'", ' U2', ' B2', "2R'", ' B2', '2R ', ' B2', "2L'", ' B2', ' D2', '2R2')
            perm_k3 = ('2L2', ' B2', ' U2', "2R'", ' U2', '2L ', ' U2', "2L'", ' U2', ' F2', "2L'", ' F2', '2R ', ' B2', '2L2')

            perm_kB = ('2R2', ' D2', "2L'", ' U2', '2R ', ' U2', ' F2', '2R ', ' F2', "2R'", ' F2', '2L ', ' F2', ' D2', '2R2')
            perm_kC = ('2R2', ' D2', ' B2', '2L ', ' B2', "2R'", ' B2', '2R ', ' B2', ' U2', '2R ', ' U2', "2L'", ' D2', '2R2')
            
            

            perm_j0 = ('2L2', ' B2', ' U2', "2L'", ' U2', '2R ', ' B2', "2R'", ' B2', '2L2', ' U2', '2L ', ' U2', "2L'", ' B2')
            perm_j1 = ('2R2', ' D2', ' B2', '2R ', ' B2', "2L'", ' D2', '2L ', ' D2', '2R2', ' B2', "2R'", ' B2', '2R ', ' D2')
            
            perm_j2 = ('2L2', ' B2', ' U2', "2L'", ' U2', '2L2', "2R'", ' F2', "2R'", ' F2', '2R2', ' U2', '2L ', ' U2', "2L'", ' B2')
            perm_j3 = ('2R2', ' D2', ' B2', '2R ', ' B2', '2R2', '2L ', ' U2', '2L ', ' U2', '2L2', ' B2', "2R'", ' B2', '2R ', ' D2')

            perm_b0 = ("2R2"," F2"," U2","2R "," U2","2R2"," F2","2R "," U2","2R2"," U2"," F2","2R "," F2")


            self._add_myperm2('W2-2s[FL@U<>RF@U]~v01', ('2D2', ' F2', "2U'", ' F2', '2U ', ' L2', "2D'", ' L2', ' F2', "2D'", ' F2', "2D'"))
            self._add_myperm2('W2-2s[FL@U<>RF@U]~v02', ('2D2', ' F2', '2D ', ' R2', "2U'", ' B2', '2D ', ' B2', ' R2', '2U ', ' F2', '2D '))
            self._add_myperm2('W2-2s[FL@U<>RF@U]~v03', ('2U2', ' B2', '2U ', ' B2', "2D'", ' B2', '2D ', ' B2', ' R2', '2D ', ' R2', '2U '))
            self._add_myperm2('W2-2s[FL@U<>RF@U]~v04', ('2U2', ' B2', '2D ', ' L2', "2D'", ' L2', '2U ', ' B2', ' R2', '2D ', ' R2', '2U '))



            
            #SwapD ('2L2', ' B2', ' U2', '2L ', ' U2', '2L2', ' B2', '2L ', ' U2', '2L2', ' B ', "2D'", " B'", ' U2', ' B ', '2D ', ' B ', '2L ', ' B2')
            #SwapE ('2L2', ' B ', '2D2', " B'", ' U2', ' B ', '2D2', ' B ', '2R ', ' B2', '2R2', ' U2', '2L ', ' F2', '2L2', ' F2', ' U2', '2R ', ' U2')

            self._add_myperm2('W2-2s[RF@D<>RF@U]~v01', ('2D2', ' L2', "2U'", ' R2', '2D ', ' R2', ' F2', '2D ', ' F2', "2D'", ' F2', '2U ', ' F2', ' L2', '2D2'))
            self._add_myperm2('W2-2s[RF@D<>RF@U]~v02', ('2D2', ' L2', ' F2', "2U'", ' F2', '2D ', ' F2', "2D'", ' F2', ' R2', "2D'", ' R2', '2U ', ' L2', '2D2'))
            

    

            swapc = ('2D2', ' B ', '2R ', " B'", ' R2', ' B ', "2R'", ' B ', '2D ', ' B2', '2D2', ' R2', '2U ', ' F2', '2D2', ' F2', ' R2', "2U'", ' R2')
            swapd = ('2U2', ' B ', "2L'", " B'", ' R2', ' B ', '2L ', ' B ', "2U'", ' B2', '2U2', ' R2', "2D'", ' F2', '2U2', ' F2', ' R2', '2D ', ' R2')
            swapex = ('2U2', " B'", '2R2', ' B ', ' R2', " B'", '2R2', " B'", "2U'", ' B2', '2U2', ' R2', "2D'", ' F2', '2U2', ' F2', ' R2', '2D ', ' R2')
            swapey = ('2D2', " B'", '2L2', ' B ', ' R2', " B'", '2L2', " B'", "2D ", ' B2', '2D2', ' R2', "2U ", ' F2', '2D2', ' F2', ' R2', "2U'", ' R2')
            swapfx = ('2U2', " F'", "2R'", ' F ', ' R2', " F'", '2R ', " F'", "2U'", ' F2', '2U2', ' R2', '2U ', ' F2', '2U2', ' F2', ' R2', '2U ', ' R2')
            swapfy = ('2D2', " F'", '2L ', ' F ', ' R2', " F'", "2L'", " F'", '2D ', ' F2', '2D2', ' R2', "2D'", ' F2', '2D2', ' F2', ' R2', "2D'", ' R2')
            swapg = ('2U2', ' F ', '2L2', " F'", ' R2', ' F ', '2L2', ' F ', "2U'", ' F2', '2U2', ' R2', '2U ', ' F2', '2U2', ' F2', ' R2', '2U ', ' R2')
            swaph = ('2D2', ' F ', '2R2', " F'", ' R2', ' F ', '2R2', ' F ', '2D ', ' F2', '2D2', ' R2', "2D'", ' F2', '2D2', ' F2', ' R2', "2D'", ' R2')

            self._add_myperm2('W2-2s[BR@U<>FL@U]~v03', ("2U'", ' L2', '2U ', '2D ', ' L2', "2D'", ' L2', '2D2', ' L2', ' B2', '2D ', ' B2', ' L2', '2D2', ' L2'))
            self._add_myperm2('W2-2s[BR@U<>FL@U]~v04', ('2D ', ' L2', "2D'", "2U'", ' L2', '2U ', ' L2', '2D2', ' L2', ' B2', '2D ', ' B2', ' L2', '2D2', ' L2'))
            self._add_myperm2('W2-2s[BR@U<>FL@U]~v05', ("2U'", ' L2', '2U ', '2D ', ' L2', "2D'", ' B2', '2D2', ' B2', ' L2', "2D'", ' L2', ' B2', '2D2', ' B2'))
            self._add_myperm2('W2-2s[BR@U<>FL@U]~v06', ('2D ', ' L2', "2D'", "2U'", ' L2', '2U ', ' B2', '2D2', ' B2', ' L2', "2D'", ' L2', ' B2', '2D2', ' B2'))
            self._add_myperm2('W2-2s[BR@U<>FL@U]~v07', ("2U'", ' R2', '2U ', '2D ', ' R2', "2D'", ' R2', '2U2', ' R2', ' F2', '2U ', ' F2', ' R2', '2U2', ' R2'))
            self._add_myperm2('W2-2s[BR@U<>FL@U]~v08', ('2D ', ' R2', "2D'", "2U'", ' R2', '2U ', ' R2', '2U2', ' R2', ' F2', '2U ', ' F2', ' R2', '2U2', ' R2'))
            self._add_myperm2('W2-2s[BR@U<>FL@U]~v09', ("2U'", ' R2', '2U ', '2D ', ' R2', "2D'", ' F2', '2U2', ' F2', ' R2', "2U'", ' R2', ' F2', '2U2', ' F2'))
            self._add_myperm2('W2-2s[BR@U<>FL@U]~v10', ('2D ', ' R2', "2D'", "2U'", ' R2', '2U ', ' F2', '2U2', ' F2', ' R2', "2U'", ' R2', ' F2', '2U2', ' F2'))

            self._add_myperm2('W2-2s[BR@U<>FL@U]~v11', ('2U ', ' B2', "2U'", "2D'", ' B2', '2D ', ' L2', '2U2', ' R2', ' F2', '2D ', ' F2', ' R2', '2U2', ' L2'))
            self._add_myperm2('W2-2s[BR@U<>FL@U]~v12', ("2D'", ' B2', '2D ', '2U ', ' B2', "2U'", ' L2', '2U2', ' R2', ' F2', '2D ', ' F2', ' R2', '2U2', ' L2'))
            self._add_myperm2('W2-2s[BR@U<>FL@U]~v13', ("2U'", ' L2', '2U ', '2D ', ' L2', "2D'", ' L2', '2U2', ' R2', ' F2', '2D ', ' F2', ' R2', '2U2', ' L2'))
            self._add_myperm2('W2-2s[BR@U<>FL@U]~v14', ('2D ', ' L2', "2D'", "2U'", ' L2', '2U ', ' L2', '2U2', ' R2', ' F2', '2D ', ' F2', ' R2', '2U2', ' L2'))
            self._add_myperm2('W2-2s[BR@U<>FL@U]~v15', ('2U ', ' B2', "2U'", "2D'", ' B2', '2D ', ' L2', '2D2', ' R2', ' F2', '2U ', ' F2', ' R2', '2D2', ' L2'))
            self._add_myperm2('W2-2s[BR@U<>FL@U]~v16', ("2D'", ' B2', '2D ', '2U ', ' B2', "2U'", ' L2', '2D2', ' R2', ' F2', '2U ', ' F2', ' R2', '2D2', ' L2'))
            self._add_myperm2('W2-2s[BR@U<>FL@U]~v17', ("2U'", ' L2', '2U ', '2D ', ' L2', "2D'", ' L2', '2D2', ' R2', ' F2', '2U ', ' F2', ' R2', '2D2', ' L2'))
            self._add_myperm2('W2-2s[BR@U<>FL@U]~v18', ('2D ', ' L2', "2D'", "2U'", ' L2', '2U ', ' L2', '2D2', ' R2', ' F2', '2U ', ' F2', ' R2', '2D2', ' L2'))




            self._add_myperm2('W2-2s[BR@U<>FL@U]~v01', ("2U'", ' R2', '2D2', ' B2', "2D'", ' B2', '2D ', '2U ', ' R2', ' F2', '2D ', ' F2', ' R2', '2D2', ' R2'))
            self._add_myperm2('W2-2s[BR@U<>FL@U]~v02', ("2D'", ' F2', '2U2', ' L2', "2U'", ' L2', "2U'", "2D'", ' F2', ' R2', "2D'", ' R2', ' F2', '2D2', ' F2'))

            

            self._add_myperm2('W2-2s[BR@D<>FL@U]~v01', ('2D2', ' B2', ' L2', "2D'", ' L2', '2U ', ' B2', "2U'", ' B2', '2D2', ' L2', '2D ', ' L2', "2D'", ' B2'))
            self._add_myperm2('W2-2s[BR@D<>FL@U]~v02', ('2U2', ' L2', ' B2', "2U'", ' B2', '2U2', "2D'", ' R2', "2D'", ' R2', '2D2', ' B2', '2U ', ' B2', "2U'", ' L2'))


      
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v01', ('2U2', ' R2', ' F2', '2U ', ' F2', '2U2', ' R2', '2U ', ' F2', '2U2', ' F2', ' R2', '2U ', ' R2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v02', ('2D2', ' R2', ' F2', '2D ', ' F2', '2D2', ' R2', "2D'", ' F2', '2D2', ' F2', ' R2', "2D'", ' R2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v03', ('2D2', ' R2', ' B2', '2D ', ' B2', '2D2', ' R2', '2U ', ' F2', '2D2', ' F2', ' R2', "2U'", ' R2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v04', ('2U2', ' R2', ' F2', "2D'", ' F2', '2D2', ' R2', "2D'", ' F2', '2U2', ' F2', ' R2', '2U ', ' R2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v05', ('2U2', ' L2', ' B2', "2D'", ' B2', '2D2', ' L2', "2U'", ' F2', '2U2', ' F2', ' L2', "2D'", ' L2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v06', ('2D2', ' L2', ' B2', "2U'", ' B2', '2U2', ' L2', '2D ', ' F2', '2D2', ' F2', ' L2', '2U ', ' L2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v07', ('2U2', ' R2', ' F2', "2U'", ' F2', '2U2', ' R2', "2D'", '2U2', ' F2', '2D2', ' F2', ' L2', '2U ', ' L2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v08', ('2D2', ' L2', ' B2', '2U2', "2D'", ' B2', '2D2', ' L2', "2U'", ' F2', '2D2', ' F2', ' L2', '2U ', ' L2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v09', ('2D2', ' R2', ' F2', '2D2', '2U ', ' F2', '2U2', ' R2', '2U ', ' F2', '2D2', ' F2', ' R2', "2D'", ' R2'))


            self._add_myperm2('W2-2s[FL@D<>RF@U]~v10', ('2U2', ' R2', ' F2', '2U ', ' F2', ' R2', '2U2', ' R2', "2U'", ' R2', '2U ', '2D ', ' R2', "2D'", ' R2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v11', ('2U2', ' R2', ' F2', '2U ', ' F2', ' R2', '2U2', ' R2', '2D ', ' R2', "2D'", "2U'", ' R2', '2U ', ' R2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v12', ('2U2', ' L2', ' F2', '2U ', ' F2', ' L2', '2U2', ' L2', '2D ', ' F2', "2D'", "2U'", ' F2', '2U ', ' L2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v13', ('2U2', ' L2', ' F2', '2U ', ' F2', ' L2', '2U2', ' L2', "2U'", ' F2', '2U ', '2D ', ' F2', "2D'", ' L2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v14', ('2D2', ' R2', ' F2', '2D ', ' F2', ' R2', '2D2', ' R2', "2U'", ' R2', '2U ', '2D ', ' R2', "2D'", ' R2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v15', ('2D2', ' R2', ' F2', '2D ', ' F2', ' R2', '2D2', ' R2', '2D ', ' R2', "2D'", "2U'", ' R2', '2U ', ' R2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v16', ('2D2', ' L2', ' F2', '2D ', ' F2', ' L2', '2D2', ' L2', '2D ', ' F2', "2D'", "2U'", ' F2', '2U ', ' L2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v17', ('2D2', ' L2', ' F2', '2D ', ' F2', ' L2', '2D2', ' L2', "2U'", ' F2', '2U ', '2D ', ' F2', "2D'", ' L2'))

            self._add_myperm2('W2-2s[FL@D<>RF@U]~v18', ('2D2', ' L2', ' B2', '2U ', ' B2', ' L2', '2D2', ' R2', '2U ', ' F2', "2U'", "2D'", ' F2', '2D ', ' R2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v19', ('2D2', ' L2', ' B2', '2U ', ' B2', ' L2', '2D2', ' R2', "2D'", ' F2', '2D ', '2U ', ' F2', "2U'", ' R2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v20', ('2D2', ' L2', ' B2', '2U ', ' B2', ' L2', '2D2', ' R2', "2U'", ' R2', '2U ', '2D ', ' R2', "2D'", ' R2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v21', ('2D2', ' L2', ' B2', '2U ', ' B2', ' L2', '2D2', ' R2', '2D ', ' R2', "2D'", "2U'", ' R2', '2U ', ' R2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v22', ('2U2', ' L2', ' B2', '2D ', ' B2', ' L2', '2U2', ' R2', '2U ', ' F2', "2U'", "2D'", ' F2', '2D ', ' R2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v23', ('2U2', ' L2', ' B2', '2D ', ' B2', ' L2', '2U2', ' R2', "2D'", ' F2', '2D ', '2U ', ' F2', "2U'", ' R2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v24', ('2U2', ' L2', ' B2', '2D ', ' B2', ' L2', '2U2', ' R2', "2U'", ' R2', '2U ', '2D ', ' R2', "2D'", ' R2'))
            self._add_myperm2('W2-2s[FL@D<>RF@U]~v25', ('2U2', ' L2', ' B2', '2D ', ' B2', ' L2', '2U2', ' R2', '2D ', ' R2', "2D'", "2U'", ' R2', '2U ', ' R2'))

            #('2L2', ' F2', ' D2', "2R'", ' D2', ' B2', '2L2', ' B2', '2R ', ' D2', "2L'", "2R'", ' D2', '2R ', ' F2')

            self._add_myperm2('W2-2s[RF@U<>UF@R]', swapc)
            self._add_myperm2('W2-2s[RF@D<>UF@L]', swapd)
            self._add_myperm2('W2-2s[RF@U<>UR@B]', ('2F2', " L'", '2U2', ' L ', ' U2', " L'", '2U2', " L'", "2F'", ' L2', '2F2', ' U2', "2B'", ' R2', '2F2', ' R2', ' U2', '2B ', ' U2'))
            self._add_myperm2('W2-2s[RF@U<>UF@L]', swapey)
            self._add_myperm2('W2-2s[FL@U<>UB@R]', swapfx)
            self._add_myperm2('W2-2s[RF@U<>UL@B]', ('2B2', " L'", '2U ', ' L ', ' D2', " L'", "2U'", " L'", '2B ', ' L2', '2B2', ' D2', "2B'", ' L2', '2B2', ' L2', ' D2', "2B'", ' D2'))
            self._add_myperm2('W2-2s[FL@U<>UB@L]', swapg)
            self._add_myperm2('W2-2s[FL@D<>UB@R]', swaph)
            
            self._add_myperm2('W2-4[FL@D>RF@D>FL@U>RF@U]~v01', ('2U2', ' R2', '2U ', ' F2', '2U ', ' F2', ' R2', '2U ', ' R2', "2U'", ' R2', '2U ', ' R2', '2U2', ' R2'))
            self._add_myperm2('W2-4[FL@D>RF@D>FL@U>RF@U]~v02', ('2D2', ' B2', "2U'", ' F2', "2D'", ' L2', '2U ', ' L2', ' F2', '2D ', ' R2', "2U'", ' R2', ' B2', '2D2'))
            self._add_myperm2('W2-4[FL@D>RF@U>FL@U>RF@D]', ('2D2', ' B2', ' R2', '2U ', ' R2', "2D'", ' F2', ' L2', "2U'", ' L2', '2D ', ' F2', '2U ', ' B2', '2D2'))
            self._add_myperm2('W2-4[FL@D>RF@D>RF@U>FL@U]', ('2D2', ' F2', '2D ', ' F2', ' B2', "2U'", ' B2', ' F2', '2D ', ' F2', '2D2'))
            self._add_myperm2('W2-4[FL@D>FL@U>RF@D>RF@U]', ('2U ', ' F2', '2U ', ' F2', ' R2', '2U ', ' R2', '2U ', ' L2', '2D2', ' L2', "2U'", ' F2', '2U2', ' F2'))

            self._add_myperm2('W2-4[RF@D>UF@R>RF@U>UF@L]', ('2R ', " B'", '2D ', ' B ', ' U2', " B'", "2D'", ' B ', ' U2', '2R ', ' B ', '2U2', " B'", ' U2', ' B ', '2U2', ' B ', '2R ', ' B2', '2R2', ' U2', '2L ', ' F2', '2R2', ' F2', ' U2', "2L'", ' U2'))
            self._add_myperm2('W2-4[RF@D>RF@U>UF@R>UF@L]', (' U ', ' L ', "2U'", ' F2', "2U'", ' F2', ' L2', "2U'", ' L2', "2U'", ' R2', '2D2', ' R2', '2U ', ' F2', '2U2', ' F2', " L'", " U'"))
            self._add_myperm2('W2-4[RF@D>RF@U>UF@L>UF@R]', (' U ', ' L ', '2U2', ' F2', '2U ', ' F2', ' B2', "2D'", ' F2', ' B2', '2U ', ' F2', '2U2', " L'", " U'"))
            self._add_myperm2('W2-4[RF@D>UF@R>UF@L>RF@U]', (' U ', ' L ', '2U2', ' F2', "2U'", ' B2', ' F2', '2D ', ' B2', ' F2', "2U'", ' F2', '2U2', " L'", " U'"))
            self._add_myperm2('W2-4[RF@D>UF@L>UF@R>RF@U]', (' U ', ' L ', ' F2', '2U2', ' F2', "2U'", ' R2', '2D2', ' R2', '2U ', ' L2', '2U ', ' L2', ' F2', '2U ', ' F2', '2U ', " L'", " U'"))


            self._add_myperm2('W2-4[FL@D>UB@L>FL@U>UB@R]~v01', ("2R'", ' B ', "2D'", " B'", ' U2', ' B ', '2D ', " B'", ' U2', "2R'", " B'", '2U ', ' B ', ' D2', " B'", "2U'", " B'", '2R ', ' B2', '2R2', ' D2', "2R'", ' B2', '2R2', ' B2', ' D2', "2R'", ' D2'))
            self._add_myperm2('W2-4[FL@D>FL@U>UB@L>UB@R]', (' B ', "2U'", ' L2', "2U'", ' L2', ' B2', "2U'", ' B2', "2U'", ' F2', '2D2', ' F2', '2U ', ' L2', '2U2', ' L2', " B'"))
            self._add_myperm2('W2-4[FL@D>FL@U>UB@R>UB@L]', (' B ', '2U2', ' L2', '2U ', ' L2', ' R2', "2D'", ' L2', ' R2', '2U ', ' L2', '2U2', " B'"))
            self._add_myperm2('W2-4[FL@D>UB@L>UB@R>FL@U]', (' B ', '2U2', ' L2', "2U'", ' R2', ' L2', '2D ', ' R2', ' L2', "2U'", ' L2', '2U2', " B'"))
            self._add_myperm2('W2-4[FL@D>UB@R>UB@L>FL@U]', (' B ', ' L2', '2U2', ' L2', "2U'", ' F2', '2D2', ' F2', '2U ', ' B2', '2U ', ' B2', ' L2', '2U ', ' L2', '2U ', " B'"))


            self._add_myperm2('W2-4[LB@D>RF@U>LB@U>RF@D]', ('2U2', ' L2', '2U ', ' L2', "2U'", ' L2', '2U ', ' L2', ' F2', '2U ', ' F2', '2U ', ' L2', '2U2', ' L2'))
            self._add_myperm2('W2-4[LB@D>LB@U>RF@U>RF@D]', ('2D2', ' R2', "2D'", ' R2', '2D ', ' R2', "2D'", ' R2', "2D'", ' R2', '2D ', ' B2', "2D'", ' B2', '2D ', ' R2'))
            self._add_myperm2('W2-4[LB@D>RF@U>RF@D>LB@U]', ('2D2', ' L2', '2D ', ' F2', "2D'", ' F2', "2U'", ' R2', "2D'", ' R2', '2U ', ' L2', '2D2'))

            
            self._add_myperm2('W2-4s[FL@D<>RF@U;FL@U<>RF@D]~v01', ('2D2', ' L2', ' F2', '2D2', ' F2', ' L2', '2D2'))
            self._add_myperm2('W2-4s[FL@D<>RF@U;FL@U<>RF@D]~v02', ('2D2', ' R2', ' B2', '2U2', ' B2', ' R2', '2D2'))
            self._add_myperm2('W2-4s[FL@D<>FL@U;RF@D<>RF@U]~v01', ("2D'", ' L2', '2U2', ' B2', "2U'", ' B2', "2U'", ' L2', '2D2', ' F2', "2D'", ' F2'))
            self._add_myperm2('W2-4s[FL@D<>FL@U;RF@D<>RF@U]~v02', ("2D'", ' L2', '2U2', ' B2', "2U'", ' B2', '2D ', ' L2', '2U2', ' F2', '2U ', ' F2'))
            self._add_myperm2('W2-4s[FL@D<>RF@D;FL@U<>RF@U]', (' R2', "2U'", ' R2', "2D'", ' R2', ' L2', "2U'", '2D ', ' L2', ' R2', "2D'", ' R2', "2D'", ' R2'))

            self._add_myperm2('W2-4s[LB@D<>RF@D;LB@U<>RF@U]', (' B2', '2D2', ' B2', ' R2', '2D2', ' R2', ' B2', '2D2', ' B2'))
            self._add_myperm2('W2-4s[LB@D<>LB@U;RF@D<>RF@U]~v01', (' B2', '2D ', ' B2', '2U2', ' L2', '2U ', ' L2', '2U ', ' B2', '2D2', ' R2', '2D ', ' R2', ' B2'))
            self._add_myperm2('W2-4s[LB@D<>LB@U;RF@D<>RF@U]~v02', (' R2', '2U ', ' R2', '2D2', ' F2', '2D ', ' F2', "2U'", ' R2', '2D2', ' B2', "2D'", ' B2', ' R2'))
            self._add_myperm2('W2-4s[LB@D<>RF@U;LB@U<>RF@D]', ('2U ', ' R2', '2D ', ' R2', ' L2', '2U ', "2D'", ' L2', ' R2', '2D ', ' R2', '2D '))


            self._add_myperm2('W2-4s[RF@D<>UF@L;RF@U<>UF@R]', ("2U'", ' R2', " B'", '2R2', ' B ', ' R2', " B'", '2R2', ' B ', ' R2', ' B ', "2L'", " B'", ' R2', ' B ', '2L ', " B'", '2U '))
            self._add_myperm2('W2-4s[RF@D<>UF@R;RF@U<>UF@L]', ("2U'", ' R2', ' B ', "2L'", " B'", ' R2', ' B ', '2L ', " B'", ' R2', " B'", '2R2', ' B ', ' R2', " B'", '2R2', ' B ', '2U '))
            self._add_myperm2('W2-4s[RF@D<>RF@U;UF@L<>UF@R]', (' U ', ' L ', '2D ', ' R2', '2U2', ' B2', '2U ', ' B2', '2U ', ' R2', '2D2', ' F2', '2D ', ' F2', " L'", " U'"))

            self._add_myperm2('W2-4s[FL@D<>UB@R;FL@U<>UB@L]~v01', ("2D'", ' L2', " F'", '2L2', ' F ', ' L2', " F'", '2L2', ' F ', ' L2', ' F ', "2R'", " F'", ' L2', ' F ', '2R ', " F'", '2D '))
            self._add_myperm2('W2-4s[FL@D<>UB@L;FL@U<>UB@R]~v01', ("2D'", ' L2', ' F ', "2R'", " F'", ' L2', ' F ', '2R ', " F'", ' L2', " F'", '2L2', ' F ', ' L2', " F'", '2L2', ' F ', '2D '))
            self._add_myperm2('W2-4s[FL@D<>FL@U;UB@L<>UB@R]', (' B ', '2D ', ' F2', '2U2', ' R2', '2U ', ' R2', '2U ', ' F2', '2D2', ' L2', '2D ', ' L2', " B'"))

            self._add_myperm2('W2-6p[3x2][BR@D>FL@D>RF@U;BR@U>FL@U>RF@D]', ("2D'", '2U ', ' F2', '2D ', "2U'", ' F2'))
            self._add_myperm2('W2-6p[3x2][BR@D>FL@U>RF@U;BR@U>FL@D>RF@D]', (' B ', '2D ', "2U'", " B'", ' R2', ' B ', "2D'", '2U ', " B'", ' R2'))
            self._add_myperm2('W2-6p[3x2][BR@D>FL@D>RF@D;BR@U>FL@U>RF@U]', (' R ', "2B'", '2F ', ' R ', ' F2', " R'", '2B ', "2F'", ' R ', ' F2', ' R2'))
            self._add_myperm2('W2-6p[3x2][BR@D>FL@U>RF@D;BR@U>FL@D>RF@U]', (' F2', ' L ', "2D'", '2U ', " L'", ' F2', ' L ', '2D ', "2U'", " L'"))
            self._add_myperm2('W2-6[BR@D>FL@U>RF@D>BR@U>FL@D>RF@U]', ('2D2', ' L2', '2U ', ' R2', '2D ', ' B2', "2U'", ' B2', ' R2', "2D'", ' F2', '2U ', '2D2', ' F2', ' L2', '2D2'))
            self._add_myperm2('W2-6[BR@D>FL@D>RF@D>BR@U>FL@U>RF@U]', ('2D2', ' R2', "2D'", ' F2', "2D'", ' F2', ' R2', "2D'", ' R2', '2D ', ' R2', "2D'", ' F2', '2D2', ' F2', ' R2', '2D2', ' R2'))
            self._add_myperm2('W2-6[BR@D>FL@U>RF@U>BR@U>FL@D>RF@D]', ("2U'", ' R2', '2U ', "2D'", ' R2', ' B2', ' L2', "2U'", ' L2', ' B2', '2U2', ' F2', "2U'", ' R2', '2D ', ' B2', "2U'", ' B2', ' R2', "2D'", ' F2', "2U'"))
            self._add_myperm2('W2-6[BR@D>FL@D>RF@U>BR@U>FL@U>RF@D]', ("2D'", '2U ', ' F2', "2U'", '2D ', ' F2', '2D2', ' L2', '2U ', ' R2', "2D'", ' R2', ' B2', "2D'", ' B2', '2D ', ' B2', "2U'", ' B2', ' L2', '2D2'))
            
            self._add_myperm2('W2-4s[FL@D<>UF@R;FL@U<>UF@L]', (" F'", '2U ', "2D'", ' F ', ' L2', " F'", "2U'", '2D ', ' F ', ' L2', '2D2', ' F2', ' L2', '2D2', ' L2', ' F2', '2D2'))
            self._add_myperm2('W2-4s[FL@D<>UF@L;FL@U<>UF@R]', (' F ', '2D2', '2U2', " F'", ' L2', ' F ', '2D2', '2U2', " F'", ' L2', '2U2', ' F2', ' L2', '2U2', ' L2', ' F2', '2U2'))
            self._add_myperm2('W2-4s[FL@D<>UB@L;FL@U<>UB@R]~v02', (' U2', ' F ', '2R ', "2L'", " F'", ' U2', ' F ', "2R'", '2L ', " F'", '2L2', ' B2', ' U2', '2L2', ' U2', ' B2', '2L2'))
            self._add_myperm2('W2-4s[FL@D<>UB@R;FL@U<>UB@L]~v02', (' U2', " F'", '2L2', '2R2', ' F ', ' U2', " F'", '2L2', '2R2', ' F ', '2R2', ' B2', ' U2', '2R2', ' U2', ' B2', '2R2'))
        
            self._add_myperm2('W2-4[FL@D>UF@R>FL@U>UF@L]', (" F'", '2U ', "2D'", ' F ', ' L2', " F'", "2U'", '2D ', ' F ', ' L2', '2D2', ' F2', "2D'", ' L2', "2D'", ' L2', ' F2', "2D'", ' F2', '2D ', ' F2', "2D'", ' F2', '2D2', ' F2'))
            self._add_myperm2('W2-4[FL@D>UF@L>FL@U>UF@R]', (' F ', '2D2', '2U2', " F'", ' L2', ' F ', '2D2', '2U2', " F'", ' L2', '2U2', ' F2', '2U ', ' L2', '2U ', ' L2', ' F2', '2U ', ' F2', "2U'", ' F2', '2U ', ' F2', '2U2', ' F2'))
            self._add_myperm2('W2-4[FL@D>UB@R>FL@U>UB@L]', (' U2', ' F ', '2R ', "2L'", " F'", ' U2', ' F ', "2R'", '2L ', " F'", '2L2', ' B2', '2L ', ' U2', '2L ', ' U2', ' B2', '2L ', ' B2', "2L'", ' B2', '2L ', ' B2', '2L2', ' B2'))
            self._add_myperm2('W2-4[FL@D>UB@L>FL@U>UB@R]~v02', (' U2', " F'", '2L2', '2R2', ' F ', ' U2', " F'", '2L2', '2R2', ' F ', '2R2', ' B2', "2R'", ' U2', "2R'", ' U2', ' B2', "2R'", ' B2', '2R ', ' B2', "2R'", ' B2', '2R2', ' B2'))


            self._add_myperm2('W2-3[FL@D>RF@U>FL@U]~v01', ("2U'", ' R ', '2B ', " R'", ' F2', ' R ', "2B'", " R'", ' F2', '2U '))
            self._add_myperm2('W2-3[FL@D>FL@U>RF@U]~v01', ("2U'", ' F2', ' R ', '2B ', " R'", ' F2', ' R ', "2B'", " R'", '2U '))
            self._add_myperm2('W2-3[FL@D>RF@U>FL@U]~v02', ("2U'", " R'", '2F ', ' R ', ' F2', " R'", "2F'", ' R ', ' F2', '2U '))
            self._add_myperm2('W2-3[FL@D>FL@U>RF@U]~v02', ("2U'", ' F2', " R'", '2F ', ' R ', ' F2', " R'", "2F'", ' R ', '2U '))
            self._add_myperm2('W2-3[FL@D>RF@U>FL@U]~v03', ('2U2', ' L2', " B'", "2R'", ' B ', ' L2', " B'", '2R ', ' B ', '2U2'))
            self._add_myperm2('W2-3[FL@D>FL@U>RF@U]~v03', ('2U2', " B'", "2R'", ' B ', ' L2', " B'", '2R ', ' B ', ' L2', '2U2'))
            self._add_myperm2('W2-3[FL@D>RF@U>FL@U]~v04', ('2U2', ' L2', ' B ', "2L'", " B'", ' L2', ' B ', '2L ', " B'", '2U2'))
            self._add_myperm2('W2-3[FL@D>FL@U>RF@U]~v04', ('2U2', ' B ', "2L'", " B'", ' L2', ' B ', '2L ', " B'", ' L2', '2U2'))
            self._add_myperm2('W2-3[FL@U>RF@U>RF@D]', ('2U2', ' R2', ' B ', '2L ', " B'", ' R2', ' B ', "2L'", " B'", '2U2'))
            self._add_myperm2('W2-3[FL@U>RF@D>RF@U]', ('2U2', ' B ', '2L ', " B'", ' R2', ' B ', "2L'", " B'", ' R2', '2U2'))

            self._add_myperm2('W2-3[BR@U>FL@U>FL@D]~v01', ('2U2', " L'", '2F ', ' U2', "2F'", ' L ', '2U2', " L'", '2F ', ' U2', "2F'", ' L '))
            self._add_myperm2('W2-3[BR@U>FL@D>FL@U]~v01', (" L'", '2F ', ' U2', "2F'", ' L ', '2U2', " L'", '2F ', ' U2', "2F'", ' L ', '2U2'))
            self._add_myperm2('W2-3[BR@U>FL@U>FL@D]~v02', ('2U2', ' L2', '2D ', ' L2', "2D'", ' L2', '2U2', ' L2', '2D ', ' L2', "2D'", ' L2'))
            self._add_myperm2('W2-3[BR@U>FL@D>FL@U]~v02', (' L2', '2D ', ' L2', "2D'", ' L2', '2U2', ' L2', '2D ', ' L2', "2D'", ' L2', '2U2'))



            self._add_myperm2('W2-8p[4x2]~v01', ('2U ', ' F2', ' B2', "2U'", '2D ', ' B2', ' F2', '2U '))
            self._add_myperm2('W2-8p[4x2]~v02', ('2D ', ' F2', ' B2', "2D'", ' F2', ' B2', '2D2', ' L2', ' R2', '2U ', ' L2', ' R2', '2D '))
            self._add_myperm2('W2-8p[2x2+4]', ('2U ', "2D'", ' F2', ' B2', "2U'", '2D ', ' B2', '2U ', ' F2', '2U ', ' F2', '2U ', ' F2', '2U ', ' F2', '2U ', ' F2'))
            self._add_myperm2('W2-8s', ('2U ', "2D'", ' F2', ' B2', "2U'", '2D ', ' B2', ' F2'))

            self._add_myperm2('W2-4[BR@U>RF@U>FL@D>FL@U]', ('2D ', ' F2', "2U'", ' R2', '2U ', ' R2', ' F2', ' L2', "2U'", ' B2', '2U2', ' B2', "2U'", ' L2'))
            self._add_myperm2('W2-4[BR@U>FL@U>FL@D>RF@U]', (' L2', '2U ', ' B2', '2U2', ' B2', '2U ', ' L2', ' F2', ' R2', "2U'", ' R2', '2U ', ' F2', "2D'"))
            self._add_myperm2('W2-4[BR@U>FL@D>FL@U>RF@U]', ('2D ', ' R2', '2D ', ' R2', '2D ', ' R2', '2D2', ' L2', "2D'", ' R2', '2D ', ' L2'))
            self._add_myperm2('W2-4[BR@U>RF@U>FL@U>FL@D]', (' L2', "2D'", ' R2', '2D ', ' L2', '2D2', ' R2', "2D'", ' R2', "2D'", ' R2', "2D'"))
            self._add_myperm2('W2-4[BR@U>FL@U>RF@U>FL@D]', ("2U'", ' L2', ' F2', '2U ', ' F2', "2U'", ' F2', '2U2', '2D ', ' F2', "2D'", ' L2', '2U2'))
            self._add_myperm2('W2-4[BR@U>FL@D>RF@U>FL@U]', ('2U2', ' L2', '2D ', ' F2', "2D'", '2U2', ' F2', '2U ', ' F2', "2U'", ' F2', ' L2', '2U '))
            self._add_myperm2('W2-4s[BR@U<>FL@D;FL@U<>RF@U]', (' R2', ' F2', ' R ', "2F'", " R'", ' F2', ' R ', '2F2', ' R ', ' F2', " R'", "2F'", ' R ', ' F2'))
            self._add_myperm2('W2-4s[BR@U<>FL@U;FL@D<>RF@U]', ('2U2', ' L2', "2U'", ' L2', "2U'", ' F2', '2D2', ' R2', "2D'", ' R2', "2D'", ' F2'))
            self._add_myperm2('W2-4s[BR@U<>RF@U;FL@D<>FL@U]', ('2U ', ' L2', "2D'", ' L2', '2D2', ' R2', "2D'", ' F2', '2D ', ' F2', '2D ', ' R2', '2D2', ' F2', "2U'", ' F2'))
            self._add_myperm2('W2-4[BR@U>FL@U>RF@D>FL@D]', ('2U2', ' L2', "2U'", ' L2', '2D ', ' L2', "2D'", ' L2', ' F2', "2U'", ' F2', ' L2', ' B2', '2U ', ' B2', ' L2'))
            self._add_myperm2('W2-4[BR@U>FL@D>RF@D>FL@U]', (' B2', '2U2', ' F2', '2D ', ' F2', "2D'", ' L2', '2U ', ' L2', ' B2', ' F2', '2U ', ' F2', "2U'", ' B2', '2U2', ' B2'))
            self._add_myperm2('W2-4[BR@U>RF@D>FL@D>FL@U]', ('2U ', ' F2', '2U ', ' F2', ' R2', '2U ', '2D ', ' R2', "2D'", ' R2', "2U'", ' R2', "2U'"))
            self._add_myperm2('W2-4[BR@U>FL@U>FL@D>RF@D]', ('2U ', ' R2', '2U ', ' R2', '2D ', ' R2', "2D'", "2U'", ' R2', ' F2', "2U'", ' F2', "2U'"))
            self._add_myperm2('W2-4[BR@D>FL@U>FL@D>RF@U]', ('2D2', ' F2', "2D'", ' F2', "2D'", "2U'", ' B2', ' L2', '2U ', ' L2', ' B2', '2U '))
            self._add_myperm2('W2-4[BR@D>RF@U>FL@D>FL@U]', ("2U'", ' B2', ' L2', "2U'", ' L2', ' B2', '2U ', '2D ', ' F2', '2D ', ' F2', '2D2'))
            self._add_myperm2('W2-4s[BR@U<>RF@D;FL@D<>FL@U]', ('2U2', ' R2', '2U ', ' F2', "2U'", ' F2', "2U'", ' R2', '2U2', ' F2', '2D ', ' F2', "2D'", ' L2', '2U ', ' L2'))
            self._add_myperm2('W2-4s[BR@U<>FL@D;FL@U<>RF@D]', ("2D'", ' F2', '2D2', ' B2', "2D'", ' R2', '2D ', ' R2', '2D ', ' B2', '2D2', ' R2', "2U'", ' R2', '2U ', ' F2'))
            self._add_myperm2('W2-4s[BR@D<>FL@D;FL@U<>RF@U]', ("2D'", ' F2', '2D ', ' F2', '2D ', ' R2', '2D2', ' F2', "2U'", ' F2', '2U ', ' L2', "2D'", ' L2', '2D2', ' R2'))
            self._add_myperm2('W2-4[BR@U>FL@D>RF@U>RF@D]', (' F2', '2D ', ' F2', "2U'", ' R2', '2U ', ' R2', ' F2', ' L2', "2U'", ' B2', '2U2', ' B2', "2U'", ' L2', ' F2'))
            self._add_myperm2('W2-4[BR@U>RF@U>RF@D>FL@D]', (' F2', '2D ', ' R2', '2D ', ' R2', '2D ', ' R2', '2D2', ' L2', "2D'", ' R2', '2D ', ' L2', ' F2'))
            self._add_myperm2('W2-4[BR@U>RF@D>FL@D>RF@U]', ("2U'", ' F2', "2U'", ' F2', "2U'", ' R2', ' B2', ' L2', '2D ', ' L2', ' B2', ' R2', "2U'"))
            self._add_myperm2('W2-4[BR@U>RF@U>FL@D>RF@D]', ('2U ', ' R2', ' B2', ' L2', "2D'", ' L2', ' B2', ' R2', '2U ', ' F2', '2U ', ' F2', '2U '))
            self._add_myperm2('W2-4s[BR@U<>RF@U;FL@D<>RF@D]', (" R'", '2F ', ' R ', ' F2', " R'", "2F'", ' R ', ' F2', ' R2', " F'", '2R ', ' F ', ' R2', " F'", "2R'", ' F '))
            self._add_myperm2('W2-4s[BR@U<>RF@D;FL@D<>RF@U]', ('2D ', ' R2', '2D ', ' R2', '2D2', ' F2', '2U ', ' L2', '2U ', ' L2', '2U2', ' F2'))
            self._add_myperm2('W2-4s[BR@U<>FL@D;RF@D<>RF@U]', ('2D ', ' R2', '2U2', ' F2', "2U'", ' R2', "2U'", ' R2', '2U ', ' F2', '2U2', ' B2', '2U ', ' B2', "2D'", ' R2'))
            self._add_myperm2('W2-4[FL@D>LB@U>FL@U>RF@U]', ('2D2', ' R2', '2D ', ' L2', "2D'", ' L2', ' R2', ' F2', "2D'", ' F2', '2U ', ' L2', "2U'", ' L2', '2D2'))
            self._add_myperm2('W2-4[FL@D>RF@U>LB@U>FL@U]', ('2U2', ' B2', "2U'", ' L2', '2U2', ' L2', ' B2', ' L2', '2U ', ' L2', ' B2', '2U ', ' B2'))
            self._add_myperm2('W2-4[FL@D>FL@U>LB@U>RF@U]', (' B2', "2U'", ' B2', ' L2', "2U'", ' L2', ' B2', ' L2', '2U2', ' L2', '2U ', ' B2', '2U2'))
            self._add_myperm2('W2-4s[FL@D<>FL@U;LB@U<>RF@U]', ('2U ', ' L2', "2U'", ' L2', "2U'", ' F2', '2D2', ' R2', "2D'", ' R2', "2D'", ' F2', '2U '))
            self._add_myperm2('W2-4s[FL@D<>LB@U;FL@U<>RF@U]', ('2U2', ' B2', '2D ', ' B2', "2D'", ' R2', '2D2', ' R2', '2D ', ' B2', "2D'", ' B2', '2U2'))
            self._add_myperm2('W2-4[BR@U>LB@U>RF@D>FL@D]', ('2U ', ' F2', '2U ', ' F2', '2U ', ' F2', '2U ', ' F2', '2U '))
            self._add_myperm2('W2-4[BR@U>RF@D>FL@D>LB@U]', ("2D'", ' B2', "2D'", ' B2', '2D ', ' L2', '2D2', ' B2', "2D'", ' B2', "2D'", ' L2'))
            self._add_myperm2('W2-4[BR@U>RF@D>LB@U>FL@D]', ('2U ', ' F2', '2U ', ' F2', '2U2', ' L2', "2U'", ' F2', '2U ', ' F2', '2U ', ' L2'))
            self._add_myperm2('W2-4s[BR@U<>RF@D;FL@D<>LB@U]', (' F2', '2U ', ' F2', ' R2', ' L2', ' B2', '2D ', ' B2', ' L2', ' R2'))
            self._add_myperm2('W2-4s[BR@U<>FL@D;LB@U<>RF@D]', (" B'", "2R'", ' B ', ' L2', " B'", '2R ', ' B ', " F'", "2R'", ' F ', ' L2', " F'", '2R ', ' F '))
            self._add_myperm2('W2-4s[BR@U<>LB@U;FL@D<>RF@D]', (' R2', "2D'", ' L2', '2D ', ' R2', "2D'", ' L2', '2D ', '2U ', ' R2', "2U'", ' L2', '2U ', ' R2', "2U'", ' L2'))
            self._add_myperm2('W2-4[BR@U>RF@D>FL@U>LB@U]', ('2D2', ' B2', "2U'", ' B2', '2U ', ' R2', "2D'", ' R2', ' B2', '2D2', ' F2', '2D ', ' B2', "2D'", ' F2'))
            self._add_myperm2('W2-4[BR@U>LB@U>RF@D>FL@U]', ('2D ', ' B2', ' F2', '2D ', ' B2', "2D'", ' F2', '2D2', ' B2', '2D ', ' B2', '2D '))
            self._add_myperm2('W2-4[BR@U>FL@U>RF@D>LB@U]', ("2D'", ' B2', "2D'", ' B2', '2D2', ' F2', '2D ', ' B2', "2D'", ' F2', ' B2', "2D'"))
            self._add_myperm2('W2-4s[BR@U<>FL@U;LB@U<>RF@D]', ("2U'", ' F2', "2U'", ' F2', "2U'", ' R2', '2D2', ' B2', "2D'", ' B2', "2D'", ' R2', "2U'"))
            self._add_myperm2('W2-4s[BR@U<>RF@D;FL@U<>LB@U]', ('2U2', ' R2', '2D ', ' R2', "2U'", '2D2', ' R2', '2U ', ' R2', "2U'", ' R2', '2D2', ' R2', ' B2', "2U'", ' B2', "2D'"))
            self._add_myperm2('W2-4[BR@U>LB@U>FL@U>RF@U]', (' F2', '2U ', ' F2', '2U ', ' F2', '2U ', ' F2', '2U ', ' F2', '2U ', ' F2'))
            self._add_myperm2('W2-4[BR@U>LB@U>RF@U>FL@U]', (' L2', '2D ', ' L2', "2D'", ' B2', ' L2', "2U'", ' L2', ' B2', "2U'", ' B2', "2U'", ' B2', "2U'", ' B2', "2U'", ' B2'))
            self._add_myperm2('W2-4s[BR@U<>FL@U;LB@U<>RF@U]', ("2U'", ' L2', ' F2', ' B2', ' R2', "2D'", ' R2', ' B2', ' F2', ' L2'))
            self._add_myperm2('W2-4s[BR@U<>RF@U;FL@U<>LB@U]', (" L'", "2B'", " L'", ' B2', ' L ', '2B ', " L'", ' B2', ' L2', ' B ', "2L'", ' B ', ' R2', " B'", '2L ', ' B ', ' R2', ' B2'))
            self._add_myperm2('W2-4[BR@U>LB@D>FL@U>RF@D]', ('2U ', ' F2', '2U2', ' L2', "2U'", ' L2', '2U2', ' R2', ' B2', '2D ', ' B2', ' R2', ' F2'))
            self._add_myperm2('W2-4[BR@U>FL@U>RF@D>LB@D]', (' R2', '2U ', ' R2', '2D ', ' B2', "2D'", ' B2', "2U'", ' R2', '2D ', ' R2', "2D'", ' F2', '2U ', ' F2'))
            self._add_myperm2('W2-4s[BR@U<>FL@U;LB@D<>RF@D]', (' R2', '2D2', ' R2', ' F2', ' R2', '2D ', "2U'", ' R2', '2U ', '2D ', ' F2'))
            self._add_myperm2('W2-4s[BR@U<>RF@D;FL@U<>LB@D]', ('2U2', ' F2', ' L2', '2U ', "2D'", ' L2', "2U'", '2D ', ' L2', '2U2', ' L2', ' F2'))
            

        self._add_myperm2('C4s[UBR<>URF;UFL<>ULB]', (" R "," B'"," R'"," F "," R "," B "," R'"," F'"," R "," B "," R'"," F "," R "," B'"," R'"," F'"))
        self._add_myperm2('C4s[UBR<>UFL;ULB<>URF]', (" U'", ' L2', ' F2', ' B2', ' R2', " D'", ' R2', ' B2', ' F2', ' L2'))
        self._add_myperm2('C4[UBR>LBU;UFL>RFU]', (" L "," F2"," R2"," D2"," R "," D2"," R "," F2"," L2"," U2"," L "," U2"))
        self._add_myperm2('C4[UBR>RFU;UFL>LBU]', (' R ', ' U2', ' R ', ' F2', " R'", ' U2', ' B2', ' L ', ' D2', " L'", ' B2', " R'"))
        self._add_myperm2('C4s[DBL<>UFL;DRB<>URF]', (' L2', ' B2', " U'", ' B2', ' L2', ' R2', ' F2', " D'", ' F2', ' R2'))
        self._add_myperm2('C4[DBL>RBD;UFL>RFU]', (" F'", ' L ', ' D ', " L'", " D'", ' L ', ' D ', " L'", " D'", ' L ', ' D ', " L'", " D'", ' F '))
        self._add_myperm2('C4s[DBL<>DRB;UFL<>URF]', (" D'", ' F2', ' D ', ' L2', " D'", ' F2', ' R2', ' U ', ' B2', " U'", ' R2', ' D '))
        self._add_myperm2('C4s[DLF<>ULB;DRB<>URF]', (' R2', ' D ', ' U ', ' R2', ' F2', ' L2', ' B2', ' D ', ' U ', ' B2', ' L2', ' F2'))

        

        self._add_myperm2('C2[UFL>FLU;URF>FUR]', (' R ', ' U2', ' R ', " F'", ' D2', ' F ', " R'", ' U2', ' R ', " F'", ' D2', ' F ', ' R2'))
        self._add_myperm2('C2[ULB>BUL;URF>RFU]', (" U2"," R'"," B "," D2"," B'"," R "," U2"," R'"," B "," D2"," B'"," R "))
        self._add_myperm2('C2[DRB>BDR;UFL>FLU]', (' L ', ' D2', " L'", ' B ', ' U2', " B'", ' L ', ' D2', " L'", ' B ', ' U2', " B'"))
        self._add_myperm2('C3[UBR>BRU;UFL>FLU;URF>RFU]', (' B ', " L'", " B'", ' R ', ' B ', ' L ', " B'", ' U2', ' R ', ' D ', " R'", ' U2', ' R ', " D'", ' R2'))
        self._add_myperm2('C3[DBL>BLD;UFL>FLU;URF>RFU]', (" F'", ' D ', ' F ', ' U ', " F'", " D'", ' F ', ' U2', " R'", ' U ', ' L2', " U'", ' R ', ' U ', ' L2'))
        self._add_myperm2('C3[DFR>FRD;UBR>BRU;UFL>FLU]', (' R2', ' B ', " L'", ' B2', ' U ', " F'", " U'", ' B ', ' U ', ' F ', " U'", ' R2', ' B ', ' L ', " B'"))
        self._add_myperm2('C3[DFR>RDF;UBR>RUB;UFL>LUF]', self.invert_moves(self.myperms2['C3[DFR>FRD;UBR>BRU;UFL>FLU]']))
        


        self._add_myperm2('C3[UBR>UFL>URF]', (" R "," B'"," R "," F2"," R'"," B "," R "," F2"," R2"))
        self._add_myperm2('C3[UBR>LUF>URF]', (" R ",' U2', ' R ', ' D ', " R'", ' U2', ' R ', " D'", " R2"))
        self._add_myperm2('C3[UBR>FLU>URF]~v01', (' F ', ' R ', ' B ', " R'", " F'", ' R ', " B'", " R'"))
        self._add_myperm2('C3[UBR>FLU>FUR]', (" L'", ' B ', ' U2', " B'", ' L ', ' B ', " L'", ' U2', ' L ', " B'"))
        self._add_myperm2('C3[UBR>UFL>FUR]', (" B'", ' R2', " B'", ' L2', ' B ', ' R2', " B'", ' L2', ' B2'))
        self._add_myperm2('C3[UBR>LUF>FUR]', (' F2', " D'", ' F ', ' U2', " F'", ' D ', ' F ', ' U2', ' F '))
        self._add_myperm2('C3[UBR>LUF>RFU]~v01', (' F ', " U'", " B'", ' U ', " F'", " U'", ' B ', ' U '))
        self._add_myperm2('C3[UBR>FLU>RFU]~v01', (' R ', ' B ', " L'", " B'", " R'", ' B ', ' L ', " B'"))
        self._add_myperm2('C3[UBR>UFL>RFU]', (' L2', ' B2', " L'", ' F2', ' L ', ' B2', " L'", ' F2', " L'"))

        self._add_myperm2('C3[UBR>FLU>URF]~v02', (" L'", ' B ', ' L ', " F'", " L'", " B'", ' L ', ' F '))
        self._add_myperm2('C3[UBR>LUF>RFU]~v02', (' U ', ' L ', " U'", " R'", ' U ', " L'", " U'", ' R '))
        self._add_myperm2('C3[UBR>FLU>RFU]~v02', (" F'", " L'", ' F ', " R'", " F'", ' L ', ' F ', ' R '))

        self._add_myperm2('C3[DBL>LUF>FUR]', (" U'", ' F ', ' D2', " F'", ' U ', ' F ', ' D2', " F'"))
        self._add_myperm2('C3[DBL>UFL>URF]~v01', (' L2', ' F ', ' R ', " F'", ' L2', ' F ', " R'", " F'"))
        self._add_myperm2('C3[DBL>FLU>RFU]~v01', (" F'", ' D ', ' F ', ' U ', " F'", " D'", ' F ', " U'"))
        self._add_myperm2('C3[DBL>UFL>RFU]', (' L ', ' F2', " L'", ' U2', ' L ', ' U2', ' L ', ' F2', " L'", ' U2', " L'", ' U2'))
        self._add_myperm2('C3[DBL>FLU>FUR]', (" R'", ' F2', " R'", ' B ', ' R ', ' F2', " R'", " B'", ' R2'))

        self._add_myperm2('C3[DBL>FUR>LUF]', (' F ', ' D2', " F'", " U'", ' F ', ' D2', " F'", ' U '))
        self._add_myperm2('C3[DBL>URF>UFL]~v01', (' F ', ' R ', " F'", ' L2', ' F ', " R'", " F'", ' L2'))
        self._add_myperm2('C3[DBL>RFU>FLU]~v01', (' U ', " F'", ' D ', ' F ', " U'", " F'", " D'", ' F '))
        self._add_myperm2('C3[DBL>RFU>UFL]', (' U2', ' L ', ' U2', ' L ', ' F2', " L'", ' U2', " L'", ' U2', ' L ', ' F2', " L'"))
        self._add_myperm2('C3[DBL>FUR>FLU]', (' R2', ' B ', ' R ', ' F2', " R'", " B'", ' R ', ' F2', ' R '))

        self._add_myperm2('C3[DBL>UFL>URF]~v02', (" D'", ' R2', ' D ', ' L2', " D'", ' R2', ' D ', ' L2'))
        self._add_myperm2('C3[DBL>FLU>RFU]~v02', (" U'", " R'", ' D2', ' R ', ' U ', " R'", ' D2', ' R '))
        self._add_myperm2('C3[DBL>URF>UFL]~v02', (' L2', " D'", ' R2', ' D ', ' L2', " D'", ' R2', ' D '))
        self._add_myperm2('C3[DBL>RFU>FLU]~v02', (" R'", ' D2', ' R ', " U'", " R'", ' D2', ' R ', ' U '))

        

        self._add_myperm2('C3[DFR>UBR>UFL]~v01', (" R "," U2"," R'"," U2"," R'"," F2"," R "," U2"," R "," U2"," R'"," F2"))
        self._add_myperm2('C3[DFR>RUB>FLU]~v01', (" D'", ' F2', ' D ', " B'", " D'", ' F ', ' D ', ' B ', " D'", ' F ', ' D '))
        self._add_myperm2('C3[DFR>BRU>LUF]', (" U'", " F'", ' U ', ' B ', " U'", " F'", ' U ', " B'", " U'", ' F2', ' U '))
        self._add_myperm2('C3[DFR>BRU>FLU]', (" B'", ' D ', ' B ', ' U2', " B'", " D'", ' B ', ' U2'))
        self._add_myperm2('C3[DFR>FLU>BRU]', (' U2', " B'", ' D ', ' B ', ' U2', " B'", " D'", ' B '))
        self._add_myperm2('C3[DFR>UFL>UBR]~v01', self.invert_moves(self.myperms2['C3[DFR>UBR>UFL]~v01']))
        self._add_myperm2('C3[DFR>FLU>RUB]~v01', self.invert_moves(self.myperms2['C3[DFR>RUB>FLU]~v01']))
        self._add_myperm2('C3[DFR>LUF>BRU]', self.invert_moves(self.myperms2['C3[DFR>BRU>LUF]']))
        self._add_myperm2('C3[DFR>RUB>FLU]~v02', (' U ', ' F2', ' U ', ' B ', " U'", ' F ', ' U ', " B'", " U'", ' F ', " U'"))
        self._add_myperm2('C3[DFR>FLU>RUB]~v02', self.invert_moves(self.myperms2['C3[DFR>RUB>FLU]~v02']))
        self._add_myperm2('C3[DFR>UBR>UFL]~v02', (' F2', " U'", ' R2', ' U ', ' R2', ' D ', ' R2', " D'", ' R2', " D'", ' F2', ' D '))
        self._add_myperm2('C3[DFR>UFL>UBR]~v02', self.invert_moves(self.myperms2['C3[DFR>UBR>UFL]~v02']))

        self._add_myperm2('EAll2[FL>LF;RF>FR]', (' R ', ' F ', ' U ', " F'", ' U2', ' F2', ' U ', ' D ', ' R ', " U'", " R'", " D'", ' F2', ' U ', " R'")) 
        self._add_myperm2('EAll2[RF>FR;UF>FU]', (' D ', ' R ', " U'", " R'", " D'", ' F2', ' U ', ' F ', ' U ', " F'", ' U2', ' F2', ' U '))
        self._add_myperm2('EAll2[RF>FR;UB>BU]', (' U ', ' R ', " U'", ' R2', ' U2', ' R ', ' L ', ' F ', " R'", " F'", " L'", ' U2', ' R '))
        self._add_myperm2('EAll2[LB>BL;RF>FR]', (' L ', " F'", " U'", ' F ', ' U2', ' F2', " U'", " D'", " L'", ' U ', ' L ', ' D ', ' F2', " U'", " L'"))
        self._add_myperm2('EAll4[DF>FD;FL>LF;RF>FR;UF>FU]', (" R'", ' F2', ' R ', " F'", ' U ', ' F ', ' L ', ' D ', ' F2', " D'", ' F ', " L'", " F'", " U'"))
        self._add_myperm2('EAll4[BR>RB;FL>LF;LB>BL;RF>FR]', (' F ', ' L2', ' R ', ' F2', ' R ', " F'", ' U ', ' F ', ' L ', ' D ', ' F2', " D'", ' F ', " L'", " F'", " U'", ' L2', ' R2', " F'"))
        self._add_myperm2('EAll4[DB>BD;FL>LF;RF>FR;UB>BU]', (' R2', " L'", ' B2', " L'", ' B ', " D'", " B'", " R'", " U'", ' B2', ' U ', " B'", ' R ', ' B ', ' D ', ' R2', ' L2'))
        
        #(" F'", ' U2', " F'", ' R ', ' D2', " R'", ' F ', ' U2', " F'", ' R ', ' D2', " R'", ' F2')
        #(' F2', ' R ', ' D2', " R'", ' F ', ' U2', " F'", ' R ', ' D2', " R'", ' F ', ' U2', ' F ')

        self._add_myperm2('EAll3[BR>LF>FR]', (' B ', " D'", ' U ', " R'", ' F2', ' R ', " U'", ' D ', " B'", ' R2'))
        self._add_myperm2('EAll3[BR>FL>FR]', (' R2', ' U ', " D'", ' F2', ' D ', " U'"))
        self._add_myperm2('EAll3[BR>LF>RF]', (' F2', " L'", ' D ', " U'", ' F ', ' R2', " F'", ' U ', " D'", ' L '))
        self._add_myperm2('EAll3[BR>FL>RF]', (" L'", " D'", " F'", " U'", ' F ', " U'", ' D ', " R'", ' U ', ' R ', ' U ', ' L '))
        self._add_myperm2('EAll3[BR>FR>LF]', (' R2', ' B ', " D'", ' U ', " R'", ' F2', ' R ', " U'", ' D ', " B'"))
        self._add_myperm2('EAll3[BR>FR>FL]', (' U ', " D'", ' F2', ' D ', " U'", ' R2'))
        self._add_myperm2('EAll3[BR>RF>LF]', (" L'", ' D ', " U'", ' F ', ' R2', " F'", ' U ', " D'", ' L ', ' F2'))
        self._add_myperm2('EAll3[BR>RF>FL]', (" L'", " U'", " R'", " U'", ' R ', " D'", ' U ', " F'", ' U ', ' F ', ' D ', ' L '))
        self._add_myperm2('EAll3[RF>UB>UR]', (' B ', ' D2', ' U2', " F'", " U'", ' F ', ' D2', ' U2', " B'", ' U '))
        self._add_myperm2('EAll3[RF>BU>UR]', (' F ', ' U ', ' R ', " F'", " U'", ' D ', ' R2', " D'", ' U ', " F'", " R'", " U'", " F'"))
        self._add_myperm2('EAll3[RF>UB>RU]', (" R'", ' F ', ' L2', ' R2', " B'", ' R ', ' B ', ' L2', ' R2', " F'"))
        self._add_myperm2('EAll3[RF>BU>RU]', (" R'", " U'", " R'", " U'", " R'", ' U ', ' R ', ' U ', ' R ', ' U '))
        self._add_myperm2('EAll3[RF>UR>UB]', (" U'", ' B ', ' U2', ' D2', " F'", ' U ', ' F ', ' U2', ' D2', " B'"))
        self._add_myperm2('EAll3[RF>UR>BU]', (' F ', ' U ', ' R ', ' F ', " U'", ' D ', ' R2', " D'", ' U ', ' F ', " R'", " U'", " F'"))
        self._add_myperm2('EAll3[RF>RU>UB]', (' F ', ' R2', ' L2', " B'", " R'", ' B ', ' R2', ' L2', " F'", ' R '))
        self._add_myperm2('EAll3[RF>RU>BU]', (" U'", " R'", " U'", " R'", " U'", ' R ', ' U ', ' R ', ' U ', ' R '))
        self._add_myperm2('EAll3[DR>BU>FL]', (' R ', " F'", ' R ', " L'", ' U2', " R'", ' L ', " F'", " R'"))
        self._add_myperm2('EAll3[DR>BU>LF]', (' L ', ' B ', " D'", ' B ', ' D ', " F'", ' B ', " R'", " B'", ' R ', ' F ', ' B2', " L'"))
        self._add_myperm2('EAll3[DR>FL>BU]', (' R ', ' F ', " L'", ' R ', ' U2', ' L ', " R'", ' F ', " R'"))
        self._add_myperm2('EAll3[DR>LF>BU]', (' L ', ' B2', " F'", " R'", ' B ', ' R ', " B'", ' F ', " D'", " B'", ' D ', " B'", " L'"))
        self._add_myperm2('EAll3[FL>UR>FR]', (' B2', ' D2', " R'", ' D2', ' B2', ' U2', " L'", ' U2'))
        self._add_myperm2('EAll3[FL>RU>FR]', (" U'", ' F ', ' R ', " L'", ' U2', ' L ', " R'", ' F ', ' U '))
        self._add_myperm2('EAll3[FL>UR>RF]', (" F'", ' U ', ' F ', " D'", ' U ', " L'", " U'", ' L ', ' D ', " U'"))
        self._add_myperm2('EAll3[FL>RU>RF]', (" U'", ' D ', ' R ', ' U ', " R'", ' U ', " D'", ' F ', " U'", " F'"))
        self._add_myperm2('EAll3[FL>FR>UR]', (' U2', ' L ', ' U2', ' B2', ' D2', ' R ', ' D2', ' B2'))
        self._add_myperm2('EAll3[FL>FR>RU]', (" U'", " F'", ' R ', " L'", ' U2', ' L ', " R'", " F'", ' U '))
        self._add_myperm2('EAll3[FL>RF>UR]', (' U ', " D'", " L'", ' U ', ' L ', " U'", ' D ', " F'", " U'", ' F '))
        self._add_myperm2('EAll3[FL>RF>RU]', (' F ', ' U ', " F'", ' D ', " U'", ' R ', " U'", " R'", " D'", ' U '))
        self._add_myperm2('EAll3[FL>BU>RU]', (' U ', " F'", " U'", " F'", " U'", " F'", ' U ', ' F ', ' U ', ' F '))
        self._add_myperm2('EAll3[FL>UB>RU]', (" F'", ' U ', ' R ', " F'", " U'", ' D ', ' R2', " D'", ' U ', " F'", " R'", " U'", ' F '))
        self._add_myperm2('EAll3[FL>BU>UR]', (' B ', ' L ', " B'", ' F ', ' U2', " F'", ' B ', ' L ', " B'"))
        self._add_myperm2('EAll3[FL>UB>UR]', (' L ', ' U ', ' L ', ' U ', " L'", " U'", " L'", " U'", " L'", ' U '))
        self._add_myperm2('EAll3[FL>RU>BU]', (" F'", " U'", " F'", " U'", ' F ', ' U ', ' F ', ' U ', ' F ', " U'"))
        self._add_myperm2('EAll3[FL>RU>UB]', (" F'", ' U ', ' R ', ' F ', " U'", ' D ', ' R2', " D'", ' U ', ' F ', " R'", " U'", ' F '))
        self._add_myperm2('EAll3[FL>UR>BU]', (' B ', " L'", " B'", ' F ', ' U2', " F'", ' B ', " L'", " B'"))
        self._add_myperm2('EAll3[FL>UR>UB]', (" U'", ' L ', ' U ', ' L ', ' U ', ' L ', " U'", " L'", " U'", " L'"))
        self._add_myperm2('EAll3[LB>UF>RF]', (" B'", ' L2', ' D2', ' R2', ' F ', ' D2', ' L2', ' U2', ' B2'))
        self._add_myperm2('EAll3[LB>FU>RF]', (' B2', ' U ', " R'", " F'", ' B ', ' U2', " B'", ' F ', " R'", " U'", ' B2'))
        self._add_myperm2('EAll3[LB>UF>FR]', (' B2', ' R ', " U'", " R'", ' D ', " U'", ' B ', ' U ', " B'", " D'", ' U ', ' B2'))
        self._add_myperm2('EAll3[LB>FU>FR]', (' L2', " U'", ' D ', " F'", ' U ', ' F ', " D'", ' U ', " L'", " U'", " L'"))
        self._add_myperm2('EAll3[LB>RF>UF]', (' B2', ' U2', ' L2', ' D2', " F'", ' R2', ' D2', ' L2', ' B '))
        self._add_myperm2('EAll3[LB>RF>FU]', (' B2', ' U ', ' R ', " F'", ' B ', ' U2', " B'", ' F ', ' R ', " U'", ' B2'))
        self._add_myperm2('EAll3[LB>FR>UF]', (' B2', " U'", ' D ', ' B ', " U'", " B'", ' U ', " D'", ' R ', ' U ', " R'", ' B2'))
        self._add_myperm2('EAll3[LB>FR>FU]', (' L ', ' U ', ' L ', " U'", ' D ', " F'", " U'", ' F ', " D'", ' U ', ' L2'))
        self._add_myperm2('EAll3[FL>UF>FR]', (" U'", ' B2', ' D2', " R'", ' D2', ' B2', ' U2', " L'", " U'"))
        self._add_myperm2('EAll3[FL>FU>FR]', (' U2', ' F ', " L'", ' R ', ' U2', ' L ', " R'", ' F ', ' U2'))
        self._add_myperm2('EAll3[FL>UF>RF]', (" U'", " F'", ' U ', ' F ', " D'", ' U ', " L'", " U'", ' L ', ' D '))
        self._add_myperm2('EAll3[FL>FU>RF]', (' D ', ' R ', " U'", " R'", " D'", ' U ', ' F ', ' U ', " F'", " U'"))
        self._add_myperm2('EAll3[FL>FR>UF]', (' U ', ' L ', ' U2', ' B2', ' D2', ' R ', ' D2', ' B2', ' U '))
        self._add_myperm2('EAll3[FL>FR>FU]', (' U2', " F'", ' R ', " L'", ' U2', " R'", ' L ', " F'", ' U2'))
        self._add_myperm2('EAll3[FL>RF>UF]', (" D'", " L'", ' U ', ' L ', " U'", ' D ', " F'", " U'", ' F ', ' U '))
        self._add_myperm2('EAll3[FL>RF>FU]', (' U ', ' F ', " U'", " F'", " U'", ' D ', ' R ', ' U ', " R'", " D'"))
        self._add_myperm2('EAll3[FL>UB>FR]', (' U ', ' B2', ' D2', " R'", ' D2', ' B2', ' U2', " L'", ' U '))
        self._add_myperm2('EAll3[FL>BU>FR]', (' F ', ' R ', " L'", ' U2', ' L ', " R'", ' F '))
        self._add_myperm2('EAll3[FL>UB>RF]', (" U'", " F'", " U'", ' F ', ' U ', " D'", " L'", ' U ', ' L ', ' D '))
        self._add_myperm2('EAll3[FL>BU>RF]', (' D ', ' R ', ' U ', " R'", ' U ', " D'", ' F ', " U'", " F'", " U'"))
        self._add_myperm2('EAll3[FL>FR>UB]', (" U'", ' L ', ' U2', ' B2', ' D2', ' R ', ' D2', ' B2', " U'"))
        self._add_myperm2('EAll3[FL>FR>BU]', (" F'", ' R ', " L'", ' U2', ' L ', " R'", " F'"))
        self._add_myperm2('EAll3[FL>RF>UB]', (" D'", " L'", " U'", ' L ', ' D ', " U'", " F'", ' U ', ' F ', ' U '))
        self._add_myperm2('EAll3[FL>RF>BU]', (' U ', ' F ', ' U ', " F'", ' D ', " U'", ' R ', " U'", " R'", " D'"))
        self._add_myperm2('EAll3[RF>RU>FU]', (" U'", " R'", " U'", " R'", " U'", " R'", ' U ', ' R ', ' U ', ' R ', ' U2'))
        self._add_myperm2('EAll3[RF>UR>FU]', (" R'", ' U ', ' R ', ' U ', " F'", " U'", ' B ', " F'", ' L ', ' F ', " L'", " B'", ' F ', " R'", " U'", ' R '))
        self._add_myperm2('EAll3[RF>FU>RU]', (' U2', " R'", " U'", " R'", " U'", ' R ', ' U ', ' R ', ' U ', ' R ', ' U '))
        self._add_myperm2('EAll3[RF>FU>UR]', (" R'", ' U ', ' R ', " F'", ' B ', ' L ', " F'", " L'", ' F ', " B'", ' U ', ' F ', " U'", " R'", " U'", ' R '))

        self._add_myperm2('C2[DFR>RFU]+EAll2[FL>FR]', (" U'", ' R ', " U'", ' B2', ' D ', " L'", " D'", ' B2', ' U2', " R'"))
        self._add_myperm2('C2[DLF>LUF]+EAll2s[FL<>RF]', (' D ', " U'", ' R ', " U'", " R'", ' U2', " F'", " U'", " R'", " F'", ' R ', ' U ', ' F ', " D'"))
        self._add_myperm2('C2[UFL>FUR]+EAll2s[RF<>UF]', (' U2', " B'", " U'", ' B ', " U'", ' L2', ' D ', " F'", " D'", ' L2'))
        self._add_myperm2('C2[UFL>FUR]+EAll2s[FL<>UF]', (" U'", " R'", " U'", ' R ', " U'", ' B2', ' D ', " L'", " D'", ' B2', " U'"))
        self._add_myperm2('C2[DFR>RFU]+EAll2[RF>LU]', (' R ', ' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F2'))
        self._add_myperm2('C2[DLF>FLU]+EAll2s[FL<>UB]', (' L ', " D'", " L'", ' D ', " L'", ' F2', ' R ', " U'", " R'", ' F2', ' L '))
        self._add_myperm2('C2[DLF>FLU]+EAll2s[BR<>FL]', (' F2', ' U ', " F'", ' U ', ' L2', " D'", ' B ', ' D ', ' L2', ' U2', " F'"))
        self._add_myperm2('C2[DLF>FLU]+EAll2[BR>LF]~v01', (' L2', ' B ', ' L ', " B'", " L'", " D'", ' B ', ' D ', ' L ', " U'", ' L ', ' U '))
        self._add_myperm2('C2[DFR>FUR]+EAll2[FL>FR]', (" R'", " F'", ' R ', ' F ', ' R ', " U'", ' R2', ' F ', ' R ', ' F ', " R'", " F'", ' R ', ' U '))
        self._add_myperm2('C2[DFR>FUR]+EAll2s[FL<>RF]', (' R ', " U'", " R'", ' U2', " F'", " U'", " R'", " F'", ' R ', ' U ', ' F ', " U'"))
        self._add_myperm2('C2s[UFL<>URF]+EAll2s[RF<>UF]', (' U2', ' B2', " R'", ' B ', ' R ', ' B ', " U'", ' B2', ' R ', ' B ', ' R ', " B'", " R'", ' B ', ' U ', ' B ', ' U2'))
        self._add_myperm2('C2s[UFL<>URF]+EAll2s[FL<>UF]', (" U'", ' R2', " F'", ' R ', ' F ', ' R ', " U'", ' R2', ' F ', ' R ', ' F ', " R'", " F'", ' R ', ' U ', ' R ', ' U '))
        self._add_myperm2('C2[DFR>FUR]+EAll2[RF>LU]', (' R2', ' U ', " R'", " U'", " R'", ' F ', ' R2', " U'", " R'", " U'", ' R ', ' U ', " R'", " F'", " R'"))
        self._add_myperm2('C2[DLF>LUF]+EAll2s[FL<>UB]', (' L ', ' D2', " B'", ' D ', ' B ', ' D ', " L'", ' D2', ' B ', ' D ', ' B ', " D'", " B'", ' D ', ' L ', ' D ', " L'"))
        self._add_myperm2('C2[DLF>LUF]+EAll2s[BR<>FL]', (" F'", ' R ', " F'", " R'", " F'", ' U ', ' F2', " R'", " F'", " R'", ' F ', ' R ', " F'", " U'", ' F2'))
        self._add_myperm2('C2[DLF>FLU]+EAll2[BR>LF]~v02', (' L ', ' D ', " B'", " D'", " L'", " D'", ' L ', ' D ', ' L ', " B'", ' L2', ' D ', ' L ', ' D ', " L'", " D'", ' L ', ' B ', ' D ', ' B ', " D'", " L'"))
        self._add_myperm2('C2s[DFR<>URF]+EAll2[FL>FR]', (' B2', ' R ', " D'", " R'", ' B2', ' F2', ' D ', ' F2', " D'", ' F2', ' L2', " U'", " L'", ' U ', " L'"))
        self._add_myperm2('C2s[DLF<>UFL]+EAll2s[FL<>RF]', (" R'", ' F ', " D'", ' B2', " U'", ' L ', ' U ', ' B2', ' F2', " L'", ' F2', ' L ', ' F2', ' D2', ' R ', ' D ', " R'", ' D2', " F'", ' R '))
        self._add_myperm2('C2[UFL>RFU]+EAll2[FL>FU]', (' U2', ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' U2', " L'", ' U2', ' L ', ' U2', ' F2', ' R ', ' F ', " R'", ' U2'))
        self._add_myperm2('C2[UFL>RFU]+EAll2s[FL<>UF]', (" U'", " L'", ' D2', ' R ', " F'", " R'", ' D2', ' U2', ' F ', ' U2', " F'", ' U2', ' L2', " B'", " L'", ' B ', ' U '))
        self._add_myperm2('C2[UBR>FUR]+EAll2s[FL<>UR]', (" L'", ' D2', ' R ', " F'", " R'", ' D2', ' U2', ' F ', ' U2', " F'", ' U2', ' L2', " B'", " L'", ' B '))
        self._add_myperm2('C2s[DLF<>UFL]+EAll2[FL>RU]', (" F'", ' U ', ' B2', " D'", ' R ', ' D ', ' B2', ' F2', " R'", ' F2', ' R ', ' F2', ' U2', ' L ', ' U ', " L'", ' F '))
        self._add_myperm2('C2s[DFR<>URF]+EAll2s[LB<>RF]', (' B2', ' L2', " F'", ' D ', ' F ', ' L2', ' R2', " D'", ' R2', ' D ', ' R2', ' B2', ' U ', ' B ', " U'", " B'"))
        self._add_myperm2('C2s[DLF<>UFL]+EAll2[BR>LF]', (' R ', ' F ', " D'", ' B2', " U'", ' L ', ' U ', ' B2', ' F2', " L'", ' F2', ' L ', ' F2', ' D2', ' R ', ' D ', " R'", ' D2', " F'", " R'"))   
        

        self._add_myperm2('C2s[DFR<>UFL]+EAll2[FL>FR]', (' L2', ' U2', ' F2', ' L ', ' F2', ' L2', ' U2', ' L ', ' F2', ' L2', ' F2', ' U2', ' L ', ' U2'))
        self._add_myperm2('C2s[ULB<>URF]+EAll2s[BR<>FL]', (' F2', ' D2', ' B2', ' L2', ' U2', ' B2', ' D2', ' F2', ' U2', ' R ', ' U2', ' R2', ' F2', ' R ', ' U2', ' R2', ' U2', ' F2', ' R ', ' F2'))
        self._add_myperm2('C2[DRB>RFU]+EAll2[FL>FR]', (" R'", ' D ', ' B2', " U'", ' L ', ' U ', ' B2', ' D2', ' R ', ' D '))
        self._add_myperm2('C2[ULB>FUR]+EAll2s[BR<>FL]', (' U ', ' F2', ' U ', ' F ', ' R ', " F'", " R'", " F'", ' U ', ' F2', " R'", " F'", " R'", ' F ', ' R ', " F'", ' U2', ' F2', " U'"))
        self._add_myperm2('C2[DBL>FUR]+EAll2s[RF<>UF]', (' L2', ' U2', " B'", " U'", ' B ', " U'", ' L2', ' D ', " F'", " D'"))
        self._add_myperm2('C2[DBL>FUR]+EAll2[RF>FU]', (' D2', ' R ', ' U ', " R'", ' D2', ' R ', " U'", ' B2', ' L ', ' U ', " L'", ' B2', ' R ', " D'", ' R ', ' D ', ' R '))
        self._add_myperm2('C2[DLF>BRU]+EAll2[FL>BU]', (' U ', " B'", ' U ', ' R2', " D'", ' F ', ' D ', ' R2', ' F ', ' U2', ' B ', ' U2', " F'", ' U2'))
        self._add_myperm2('C2[DLF>BRU]+EAll2s[FL<>UB]', (' U2', ' L ', ' D ', " L'", ' U2', ' L ', " D'", ' L2', ' F2', ' R ', ' U ', " R'", ' F2', ' L ', " D'", ' L ', ' D ', " L'"))
        self._add_myperm2('C2[DRB>LUF]+EAll2[RF>RU]', (' D2', ' F2', " R'", ' D2', ' B2', ' U2', ' L2', ' B2', ' D2', " F'", " R'", ' F ', ' D2', " B'", ' L ', ' B '))
        self._add_myperm2('C2[DRB>LUF]+EAll2s[RF<>UR]', (' U2', " B'", " R'", ' B ', ' R ', ' B ', " U'", ' B2', ' R ', ' B ', ' R ', " B'", " R'", ' B ', ' L2', ' D2', ' F ', ' D2', ' L2', ' U2', ' B ', ' U '))
        self._add_myperm2('C2[DBL>FUR]+EAll2[FL>BU]', (' D ', ' F ', " D'", ' L2', ' U ', " B'", " U'", ' L2', ' D2', " F'", ' D2', ' U2', ' L2', ' D2', " B'", ' R2', ' D2', ' L2', " F'"))
        self._add_myperm2('C2[DBL>FUR]+EAll2s[FL<>UB]', (' L ', ' U2', ' L2', ' R2', ' D ', ' L2', " D'", ' L2', ' R2', ' U ', ' L2', ' U ', " L'", " R'", ' D ', " B'", " D'", ' B ', ' D ', ' B ', ' D2', " R'", ' D ', ' B ', ' D ', " B'", " D'", ' R2'))
        self._add_myperm2('C2s[DFR<>URF]+EAll2[FL>FR]', (' R ', " L'", ' D2', " R'", ' B ', ' R ', ' D2', ' U2', " B'", ' U2', ' B ', ' U2', ' L2', ' F ', ' L ', " F'", ' L2', " R'"))
        self._add_myperm2('C2s[DLF<>UFL]+EAll2s[FL<>RF]', (" L'", ' R2', " U'", ' B ', ' D2', ' F ', " L'", " F'", ' D2', ' U2', ' L ', ' U2', " L'", ' U2', ' B2', " R'", " B'", ' R ', ' B2', ' U ', ' R2', ' L '))
        self._add_myperm2('C2s[DLF<>UFL]+EAll2[FL>FR]', (' L2', ' R2', ' F2', " L'", ' U ', ' L ', ' F2', ' B2', " U'", ' B2', ' U ', ' B2', ' R2', ' D ', ' R ', " D'", " R'", ' L2'))
        self._add_myperm2('C2s[DFR<>URF]+EAll2s[FL<>RF]', (' R2', " L'", " B'", ' U ', ' F2', ' D ', " R'", " D'", ' F2', ' B2', ' R ', ' B2', " R'", ' B2', ' U2', " L'", " U'", ' L ', ' U2', ' B ', ' L ', ' R2'))
        self._add_myperm2('C2[DRB>LUF]+EAll2[FL>FR]', (' D2', ' L ', ' D2', ' B2', ' U ', ' R ', " U'", ' B2', ' D ', " L'", " D'"))
        self._add_myperm2('C2[DRB>FLU]+EAll2[FL>FR]', (' D2', ' L ', " D'", ' B2', ' U ', " R'", " U'", ' B2', " D'", ' R ', " D'", " L'", ' D ', " R'"))
        self._add_myperm2('C2s[DRB<>UFL]+EAll2[FL>FR]', (' D2', ' F2', " D'", ' F2', ' D ', ' F2', ' D2', ' L2', ' U ', ' L ', " U'", ' L ', ' F2', " R'", ' D ', ' R ', ' F2'))
        self._add_myperm2('C2[UFL>RFU]+EAll2s[FL<>UL]', (' B2', ' D ', ' L ', " D'", ' B2', ' U ', " R'", ' U ', ' R ', ' U2', ' R2', ' B2', ' R ', ' F2', " R'", ' B2', ' R ', ' F2', ' R '))
        self._add_myperm2('C2[UFL>RFU]+EAll2[FL>LU]', (' U2', " B'", ' U ', ' F2', " U'", ' B ', ' U ', ' F2', ' U ', " F'", " D'", ' F ', ' U2', " F'", ' D ', ' F ', ' U2', ' F ', ' U ', " F'", ' U ', ' F ', ' U ', " F'", ' U2'))
        self._add_myperm2('C2[UFL>RFU]+EAll2s[FL<>UR]', (" L'", ' U2', " L'", ' B2', ' D2', ' R2', ' D2', ' B2', ' L ', ' U2', ' B2', " D'", " R'", ' D ', ' B2', " U'", ' L ', ' U '))
        self._add_myperm2('C2[UFL>RFU]+EAll2[FL>RU]', (' L2', ' B ', " L'", ' F2', ' L ', " B'", " L'", ' F2', ' D ', " L'", " F'", ' D ', ' R ', ' F ', ' D2', " F'", ' D ', ' R ', ' D ', " R'", ' D2', ' F '))
        self._add_myperm2('C2[DBL>FDL]+EAll2s[RF<>UF]', (' D2', ' F ', ' R2', " B'", ' D2', ' B ', ' D2', ' F2', ' R2', " F'", ' R2', ' F ', ' R2', ' D2', ' F2', ' D2', ' B ', ' R2', ' D2', ' L2', ' F ', ' U2', ' L2'))
        self._add_myperm2('C2[DBL>FDL]+EAll2[RF>FU]', (' U ', ' L ', " U'", ' D ', ' F2', " D'", ' U ', " L'", ' U2', ' B2', ' U2', " B'", ' U2', ' B ', ' U2', " B'", ' U2', ' L2', " B'", ' L2', " B'", ' U2', ' B2', " U'"))
        self._add_myperm2('C2[DBL>FDL]+EAll2s[RF<>UB]', (' D2', ' F ', ' R2', " B'", ' D2', ' B ', ' D2', ' F2', ' R2', " F'", ' R2', ' F ', ' R2', ' D2', ' F2', ' U2', ' F ', ' U2', ' L2', ' D2', ' B ', ' D2', ' L2'))
        self._add_myperm2('C2[DBL>FDL]+EAll2[RF>BU]', (" U'", ' L ', " U'", ' D ', ' F2', ' U ', " D'", " L'", ' U2', ' F2', ' U2', ' F ', ' U2', " F'", ' U2', ' F ', ' U2', ' L2', ' F ', ' L2', ' F ', ' U2', ' F2', ' U '))
        self._add_myperm2('C2s[DLF<>UFL]+EAll2s[LB<>RF]', (' D2', ' R2', " U'", ' L2', ' U ', " D'", ' B2', ' F2', " D'", ' F2', ' R2', " D'", ' R2', ' D ', ' R2', " D'", ' R2', ' D2', ' R2', ' F2'))
        self._add_myperm2('C2s[DLF<>UFL]+EAll2[BR>LF]', (" F'", " D'", ' R ', ' U2', ' L ', " F'", " L'", ' U2', ' D2', ' F ', ' D2', " F'", ' D2', ' R2', " B'", " R'", ' B ', ' R2', ' D ', ' F '))
        self._add_myperm2('C2[DRB>FLU]+EAll2s[FL<>RF]', (' F ', ' D ', " F'", " U'", ' F ', " D'", " F'", ' U ', " R'", " B'", ' R ', " F'", " R'", ' B ', ' R2', " F'", " R'", ' F ', " D'", " L'", " F'", ' L ', ' D ', ' R ', ' F ', " R'"))
        self._add_myperm2('C2s[DBL<>UBR]+EAll2[FL>FR]', (' B2', ' D2', ' B2', ' U2', ' F2', ' U2', ' D2', ' R ', ' D2', ' B2', ' R2', ' B2', ' R ', ' D2', ' R2', ' B2', " R'", ' B2', ' D2', ' R2'))
        self._add_myperm2('C2[DBL>RUB]+EAll2[FL>FR]', (' B2', ' U2', ' D2', ' F2', ' U2', ' R ', " D'", ' F2', ' U ', " L'", " U'", ' F2', ' D2', " R'", ' D '))
        
        

    def _register_myperms2_center_general(self):
        """4x4以上で使うCenter系・Bar系の手順を登録する。"""
        # 命名メモ:
        # - X-Center / Plus-Center / Oblique-Center は center の配置 family。
        # - Adjacent3Center / Line3Center は 3面の center 配置 family。
        # - OuterCenterBar / MidCenterBar は center の bar を動かす family。
        if self.size >= 4:
            self._add_myperm2('CtrX6p[3x2][L@2B.2D>L@2F.2U>R@2B.2U;L@2B.2U>R@2B.2D>R@2F.2U]', ('2B2', '2D2', '2B2', '2D2'))
            self._add_myperm2('CtrX6p[3x2][L@2B.2U>L@2F.2D>R@2B.2U;L@2F.2U>R@2B.2D>R@2F.2U]', (" L'", '2B2', '2D2', '2B2', '2D2', ' L '))
            self._add_myperm2('CtrX6p[3x2][L@2B.2D>R@2B.2U>L@2F.2U;L@2F.2D>R@2B.2D>R@2F.2U]', (' R2', '2F2', '2U2', '2F2', '2U2', ' R2'))
            self._add_myperm2('CtrX6p[3x2][L@2B.2D>R@2B.2D>R@2B.2U;L@2F.2U>R@2F.2D>R@2F.2U]', ('2B2', " R'", '2D2', '2B2', '2D2', '2B2', ' R ', '2B2'))
            self._add_myperm2('CtrX6p[3x2][L@2B.2D>L@2B.2U>R@2B.2D;L@2F.2D>L@2F.2U>R@2F.2U]', ('2B2', ' L ', '2D2', '2B2', '2D2', '2B2', " L'", '2B2'))
            self._add_myperm2('CtrX6p[3x2][L@2B.2D>R@2B.2U>L@2B.2U;L@2F.2U>R@2F.2D>R@2F.2U]', ('2B2', ' L ', '2U2', '2B2', '2U2', '2B2', " L'", '2B2'))
            self._add_myperm2('CtrX6p[3x2][L@2B.2U>R@2B.2U>R@2B.2D;L@2F.2D>L@2F.2U>R@2F.2U]', ('2B2', " R'", '2U2', '2B2', '2U2', '2B2', ' R ', '2B2'))
            self._add_myperm2('CtrX4s[B@2R.2U<>F@2R.2U;L@2F.2U<>R@2F.2U]~v01', ('2B ', "2U'", "2B'", '2U ', "2B'", '2R ', '2B ', "2R'"))
            self._add_myperm2('CtrX4s[B@2R.2U<>F@2R.2U;L@2B.2U<>R@2B.2U]', ('2B ', "2U'", '2B ', '2U ', "2B'", '2R ', "2B'", "2R'"))
            self._add_myperm2('CtrX4s[B@2R.2U<>F@2R.2U;L@2F.2U<>R@2F.2U]~v02', ('2R ', "2B'", "2R'", '2B ', "2U'", '2B ', '2U ', "2B'"))
            self._add_myperm2('CtrX6p[3x2][L@2B.2D>R@2F.2U>L@2F.2D;L@2F.2U>R@2F.2D>R@2B.2D]', ('2U2', ' R ', '2F2', " R'", '2F2', '2U2', '2F2', ' R ', '2F2', " R'"))
            self._add_myperm2('CtrX4s[L@2B.2D<>R@2F.2D;L@2F.2U<>R@2B.2U]', ('2F2', '2U2', '2F2', '2U2', ' R2', '2F2', '2U2', '2F2', '2U2', ' R2'))


            

            self._add_myperm2('CtrX10p[5x2]~v02', ("2U2","2R2","2U'","2R2","2U'","2R2","2U'","2R2","2U "))
            
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>L@2F.2U>U@2R.2F;D@2R.2F>F@2R.2U>R@2F.2U]', ("2R ","2U ","2R'","2U'"))
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>L@2F.2U>D@2R.2B;F@2R.2U>R@2F.2U>U@2R.2B]', ("2R'", '2U ', '2R ', "2U'"))
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>L@2F.2U>R@2B.2U;F@2R.2U>R@2F.2U>L@2B.2U]', ("2U'", '2B2', '2U ', '2B2'))
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>R@2B.2U>L@2F.2U;F@2R.2U>L@2B.2U>R@2F.2U]', ('2B2', "2U'", '2B2', '2U '))
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>R@2F.2U>U@2R.2B;D@2R.2B>F@2R.2U>L@2F.2U]', ('2U ', "2R'", "2U'", '2R2', "2B'", "2R'", '2B '))
            self._add_myperm2('CtrX10p[5x2]~v01', ("2F'", '2D2', "2F'", '2D2', '2L ', '2F ', "2L'", "2U'", '2F ', '2U '))
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>L@2B.2U>U@2L.2F;D@2L.2F>F@2L.2U>R@2B.2U]', ('2L ', '2D ', '2L ', "2D'", '2L2'))
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>L@2B.2U>D@2L.2B;F@2L.2U>R@2B.2U>U@2L.2B]', ("2L'", '2D ', "2L'", "2D'", '2L2'))
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>U@2L.2F>L@2B.2U;D@2L.2F>R@2B.2U>F@2L.2U]', ('2L2', '2D ', "2L'", "2D'", "2L'"))
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>D@2L.2B>L@2B.2U;F@2L.2U>U@2L.2B>R@2B.2U]', ('2L2', '2D ', '2L ', "2D'", '2L '))
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>R@2F.2U>L@2F.2D;F@2L.2U>L@2F.2U>R@2F.2D]', ('2F2', "2U'", '2F2', "2U'", '2F2', '2U2', '2F2'))
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>R@2F.2D>L@2F.2U;F@2R.2U>L@2F.2D>R@2F.2U]', ('2F2', '2U2', '2F2', "2U'", '2F2', "2U'", '2F2'))
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>R@2B.2D>L@2B.2U;F@2L.2U>L@2B.2D>R@2B.2U]', ("2D'", '2L2', '2D2', '2L2', "2D'"))
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>R@2B.2U>L@2F.2U;F@2L.2U>L@2B.2U>R@2F.2U]', ("2B'", "2L'", '2B2', '2L ', "2B'"))
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>L@2F.2U>R@2B.2U;F@2L.2U>R@2F.2U>L@2B.2U]', ('2B ', "2L'", '2B2', '2L ', '2B '))

            

            self._add_myperm2('CtrX6p[3x2][B@2R.2U>L@2B.2D>L@2F.2U;F@2R.2U>R@2B.2D>R@2F.2U]', ('2U2', '2F2', "2U'", '2F2', "2U'"))
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>R@2F.2D>R@2B.2U;F@2R.2U>L@2F.2D>L@2B.2U]', ("2U'", '2B2', "2U'", '2B2', '2U2'))
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>R@2F.2D>R@2F.2U;F@2R.2U>L@2F.2D>L@2F.2U]', ("2B'", "2R'", "2U'", '2R ', '2U ', '2B '))
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>R@2B.2U>B@2R.2D;F@2L.2U>L@2B.2U>F@2R.2D]~v01', ('2L ', "2B'", '2L2', '2B ', "2L'", '2D2', '2L2', '2D2'))
            self._add_myperm2('CtrX8s~v01', ("2B'", "2L'", '2B2', '2L2', '2U ', "2L'", "2U'", "2B'"))
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>R@2B.2U>B@2R.2D;F@2L.2U>L@2B.2U>F@2R.2D]~v02', ("2L'", '2B2', "2R'", '2B2', '2L ', '2B ', '2R ', "2B'"))
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>R@2B.2U>B@2R.2D;F@2L.2U>L@2B.2U>F@2R.2D]~v03', ("2B'", "2L'", '2B ', "2L'", '2D2', '2L ', '2D2', '2L '))
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>R@2F.2U>R@2B.2U;F@2L.2U>L@2F.2U>L@2B.2U]', ('2L ', "2U'", "2L'", '2U ', "2L'", "2B'", '2L ', '2B '))
            self._add_myperm2('CtrX8s~v02', ('2L ', "2B'", '2L2', '2B ', "2L'", '2U2', '2L2', '2U2'))
            self._add_myperm2('CtrX6p[3x2][B@2L.2D>B@2R.2U>L@2F.2U;F@2L.2D>F@2L.2U>R@2F.2U]', ("2D'", ' F ', "2U'", '2B2', '2U ', " F'", '2B2', '2D '))
            self._add_myperm2('CtrX6p[3x2][B@2L.2D>B@2L.2U>L@2F.2U;F@2L.2D>F@2R.2U>R@2F.2U]', ("2D'", " B'", "2U'", '2B2', '2U ', ' B ', '2B2', '2D '))
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>L@2F.2D>L@2F.2U;F@2R.2U>R@2F.2D>R@2B.2U]', ('2F ', " R'", '2U ', '2R ', "2U'", ' R ', "2R'", "2F'"))
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>L@2F.2D>L@2B.2U;F@2R.2U>R@2F.2D>R@2F.2U]', ('2F ', ' L ', '2U ', '2R ', "2U'", " L'", "2R'", "2F'"))
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>R@2F.2D>R@2B.2U;F@2L.2D>L@2F.2D>L@2B.2U]', (' F2', "2U'", '2B2', "2U'", '2B2', '2U2', ' F2'))
            self._add_myperm2('CtrX6p[3x2][B@2L.2D>R@2F.2D>R@2B.2U;F@2R.2U>L@2F.2D>L@2B.2U]', (' B2', "2U'", '2B2', "2U'", '2B2', '2U2', ' B2'))
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>L@2B.2D>L@2F.2U;F@2L.2D>R@2B.2D>R@2F.2U]', (' B2', "2D'", '2B2', '2D ', ' B2', '2D2', '2B2', '2D2'))
            self._add_myperm2('CtrX6p[3x2][B@2L.2D>L@2B.2D>L@2F.2U;F@2R.2U>R@2B.2D>R@2F.2U]', (' F2', "2D'", '2B2', '2D ', ' F2', '2D2', '2B2', '2D2'))
            self._add_myperm2('CtrX8s~v03', (' B2', "2D'", '2B2', '2D ', ' B2', '2U2', '2B2', '2U2'))
            self._add_myperm2('CtrX8s~v04', (' F2', "2D'", '2B2', '2D ', ' F2', '2U2', '2B2', '2U2'))
            self._add_myperm2('CtrX6p[3x2][B@2L.2D>B@2R.2U>R@2B.2U;F@2R.2U>L@2B.2U>L@2F.2U]', ("2U'", '2B2', " F'", "2D'", '2B2', '2D ', ' F ', '2U '))
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>R@2B.2U>R@2F.2U;F@2L.2D>F@2R.2U>L@2B.2U]', ("2U'", '2B2', ' B ', "2D'", '2B2', '2D ', " B'", '2U '))



            self._add_myperm2('CtrX3[L@2F.2D>R@2F.2U>R@2B.2U]', (' R ', '2F2', " R'", '2U2', ' R ', '2F2', " R'", '2U2'))
            self._add_myperm2('CtrX3[L@2F.2D>R@2B.2D>R@2B.2U]~v01', (" L'", '2B2', ' L ', '2F2', " L'", '2B2', ' L ', '2F2'))
            self._add_myperm2('CtrX3[L@2F.2D>L@2F.2U>R@2B.2D]', ('2F2', '2U ', ' B ', "2U'", '2F2', '2U ', " B'", "2U'"))
            self._add_myperm2('CtrX3[L@2F.2D>R@2B.2D>R@2B.2U]~v02', ('2F2', "2D'", ' F ', '2D ', '2F2', "2D'", " F'", '2D '))
            self._add_myperm2('CtrX3[L@2F.2D>L@2F.2U>R@2B.2U]~v01', ('2B2', ' L ', '2D2', " L'", '2B2', ' L ', '2D2', " L'"))
            self._add_myperm2('CtrX3[L@2F.2D>L@2F.2U>R@2B.2U]~v02', ('2B2', ' R ', '2F2', " R'", '2B2', ' R ', '2F2', " R'"))
            self._add_myperm2('CtrX3[L@2F.2U>R@2B.2D>R@2B.2U]', ("2D'", " F'", '2D ', '2B2', "2D'", ' F ', '2D ', '2B2'))
            self._add_myperm2('CtrX3[L@2F.2D>L@2F.2U>R@2B.2U]~v03', ('2U ', " B'", "2U'", '2B2', '2U ', ' B ', "2U'", '2B2'))
            self._add_myperm2('CtrX3[L@2B.2U>R@2B.2U>L@2F.2U]', (' L ', '2B2', ' L ', '2D2', " L'", '2B2', ' L ', '2D2', ' L2'))
            self._add_myperm2('CtrX3[L@2B.2U>R@2B.2D>R@2B.2U]', (' L ', '2B2', ' L ', '2F2', " L'", '2B2', ' L ', '2F2', ' L2'))
            self._add_myperm2('CtrX8s~v05', ('2B2', ' R ', "2D'", "2U'", '2B2', '2D ', '2U ', '2B2', " R'", '2B2'))
            self._add_myperm2('CtrX8s~v06', ('2B2', ' R ', '2B2', "2U'", "2D'", '2B2', '2U ', '2D ', " R'", '2B2'))
            self._add_myperm2('CtrX8s~v07', ('2D2', " R'", "2B'", "2F'", '2D2', '2B ', '2F ', '2D2', ' R ', '2D2'))
            self._add_myperm2('CtrX8s~v08', ('2D2', " R'", '2D2', "2F'", "2B'", '2D2', '2F ', '2B ', ' R ', '2D2'))
            self._add_myperm2('CtrX6p[2+4][L@2B.2D>L@2B.2U>R@2B.2U>L@2F.2D;R@2B.2D<>R@2F.2U]', ('2U2', ' R ', '2D2', " R'", '2U2', ' R ', '2F2', '2D2', '2F2', " R'"))
            


            self._add_myperm2('CtrX3[F@2R.2U>R@2F.2D>R@2B.2U]~v01', ('2F ', "2R'", "2F'", ' R2', '2F ', '2R ', "2F'", ' R2'))
            self._add_myperm2('CtrX3[F@2R.2U>R@2B.2D>R@2F.2U]', ('2D ', '2R2', "2D'", ' R2', '2D ', '2R2', "2D'", ' R2'))
            self._add_myperm2('CtrX3[F@2L.2D>F@2R.2U>R@2B.2U]~v01', ('2L ', "2B'", "2L'", ' F2', '2L ', '2B ', "2L'", ' F2'))
            self._add_myperm2('CtrX3[F@2L.2D>F@2R.2U>R@2F.2U]', ("2D'", '2B2', '2D ', ' F2', "2D'", '2B2', '2D ', ' F2'))
            self._add_myperm2('CtrX3[F@2R.2U>R@2B.2U>R@2F.2U]', ('2U ', ' F ', "2D'", " F'", "2U'", ' F ', '2D ', " F'"))
            self._add_myperm2('CtrX3[F@2R.2D>F@2R.2U>R@2F.2U]', ("2U'", ' R ', '2D ', " R'", '2U ', ' R ', "2D'", " R'"))
            self._add_myperm2('CtrX3[F@2R.2U>R@2F.2U>R@2B.2U]', (' F ', "2D'", " F'", '2U ', ' F ', '2D ', " F'", "2U'"))
            self._add_myperm2('CtrX3[F@2R.2D>F@2R.2U>R@2B.2U]', (" R'", "2U'", ' R ', '2D ', " R'", '2U ', ' R ', "2D'"))


            
            self._add_myperm2('CtrX3[F@2L.2D>F@2R.2U>R@2B.2D]', (' F2', '2U ', '2F2', "2U'", ' F2', '2U ', '2F2', "2U'"))
            self._add_myperm2('CtrX3[F@2L.2U>R@2B.2U>F@2R.2D]', (' F2', '2U ', '2B2', "2U'", ' F2', '2U ', '2B2', "2U'"))
            self._add_myperm2('CtrX3[F@2L.2D>F@2R.2U>R@2F.2D]', (' F2', "2R'", '2F ', '2R ', ' F2', "2R'", "2F'", '2R '))
            self._add_myperm2('CtrX3[F@2L.2U>R@2F.2U>F@2R.2D]', (' F2', "2L'", '2B ', '2L ', ' F2', "2L'", "2B'", '2L '))

            self._add_myperm2('CtrX3[F@2R.2U>R@2F.2D>R@2B.2U]~v02', (' R2', '2B ', '2R ', "2B'", ' R2', '2B ', "2R'", "2B'"))
            self._add_myperm2('CtrX3[F@2L.2D>F@2R.2U>R@2B.2U]~v02', (' F2', '2R ', '2B ', "2R'", ' F2', '2R ', "2B'", "2R'"))

            self._add_myperm2('CtrX3[F@2R.2U>U@2R.2F>R@2F.2D]', (" F ","2D "," R2","2D'","2R'","2D "," R2","2D'","2R "," F'"))
            self._add_myperm2('CtrX3[F@2L.2D>U@2L.2B>R@2B.2U]', (" F ","2U'"," R2","2U ","2L ","2U'"," R2","2U ","2L'"," F'"))
            self._add_myperm2('CtrX3[F@2L.2U>U@2R.2B>R@2B.2D]', (" F'","2D'","2L'"," U2","2L ","2D ","2L'"," U2","2L "," F "))
            self._add_myperm2('CtrX3[F@2R.2D>U@2L.2F>R@2F.2U]', (" F'","2U ","2R "," U2","2R'","2U'","2R "," U2","2R'"," F "))

            self._add_myperm2('CtrX3[F@2R.2U>U@2R.2F>R@2B.2U]', (" R'", '2B ', ' U2', "2B'", "2U'", '2B ', ' U2', "2B'", '2U ', ' R '))
            self._add_myperm2('CtrX3[F@2R.2U>U@2L.2B>R@2B.2U]', (" F'","2U'"," R2","2U ","2L ","2U'"," R2","2U ","2L'"," F "))
            self._add_myperm2('CtrX3[F@2L.2U>U@2R.2B>R@2F.2U]', (' R ', "2B'", "2D'", ' F2', '2D ', '2B ', "2D'", ' F2', '2D ', " R'"))
            self._add_myperm2('CtrX3[F@2L.2U>U@2L.2F>R@2F.2U]', (" F ","2U ","2R "," U2","2R'","2U'","2R "," U2","2R'"," F'"))
            
            self._add_myperm2('CtrX3[F@2R.2U>U@2R.2F>R@2F.2U]', ('2B ', ' U2', "2B'", "2U'", '2B ', ' U2', "2B'", '2U '))
            self._add_myperm2('CtrX3[F@2L.2U>U@2L.2B>R@2B.2U]', ("2U'"," R2","2U ","2L ","2U'"," R2","2U ","2L'"))
            self._add_myperm2('CtrX3[F@2L.2U>U@2R.2B>R@2B.2U]', ("2B'", "2D'", ' F2', '2D ', '2B ', "2D'", ' F2', '2D '))
            self._add_myperm2('CtrX3[F@2R.2U>U@2L.2F>R@2F.2U]', ("2U ","2R "," U2","2R'","2U'","2R "," U2","2R'"))

            self._add_myperm2('CtrX3[F@2R.2U>R@2F.2U>U@2R.2F]', ("2U'", '2B ', ' U2', "2B'", '2U ', '2B ', ' U2', "2B'"))
            self._add_myperm2('CtrX3[F@2L.2U>R@2B.2U>U@2L.2B]', ('2L ', "2U'", ' R2', '2U ', "2L'", "2U'", ' R2', '2U '))
            self._add_myperm2('CtrX3[F@2L.2U>R@2B.2U>U@2R.2B]', ("2D'", ' F2', '2D ', "2B'", "2D'", ' F2', '2D ', '2B '))
            self._add_myperm2('CtrX3[F@2R.2U>R@2F.2U>U@2L.2F]', ('2R ', ' U2', "2R'", '2U ', '2R ', ' U2', "2R'", "2U'"))

            self._add_myperm2('CtrX3[F@2R.2U>U@2L.2B>R@2F.2U]', ("2B'", " L'", "2U'", ' L ', '2B ', " L'", '2U ', ' L '))
            self._add_myperm2('CtrX3[F@2R.2U>U@2R.2B>R@2B.2U]', ("2R "," D'","2B ",' D ',"2R'"," D'","2B'",' D '))
            self._add_myperm2('CtrX3[F@2R.2U>U@2R.2B>R@2F.2U]', ('2U ', ' B ', "2R'", " B'", "2U'", ' B ', '2R ', " B'"))
            self._add_myperm2('CtrX3[F@2L.2U>U@2L.2B>R@2F.2U]', ("2L'"," D ","2B "," D'","2L "," D ","2B'"," D'"))

            self._add_myperm2('CtrX3[F@2L.2U>U@2R.2F>R@2B.2U]', (' F2', "2D'", " B'", "2R'", ' B ', '2D ', " B'", '2R ', ' B ', ' F2'))
            self._add_myperm2('CtrX3[F@2R.2U>U@2L.2F>R@2B.2U]', (" U2","2R "," D'","2B ",' D ',"2R'"," D'","2B'",' D '," U2"))
            self._add_myperm2('CtrX3[F@2L.2U>U@2L.2F>R@2B.2U]', (' R2', '2F ', ' L ', "2U'", " L'", "2F'", ' L ', '2U ', " L'", ' R2'))
            self._add_myperm2('CtrX3[F@2L.2U>U@2R.2F>R@2F.2U]', (" U2","2L'"," D ","2B "," D'","2L "," D ","2B'"," D'"," U2"))

            self._add_myperm2('CtrX3[F@2R.2U>L@2B.2D>R@2F.2U]', ('2U ', ' F ', '2D ', " F'", "2U'", ' F ', "2D'", " F'"))
            self._add_myperm2('CtrX3[F@2R.2U>R@2B.2U>L@2F.2D]', ("2U'", ' F ', "2D'", " F'", '2U ', ' F ', '2D ', " F'"))
            self._add_myperm2('CtrX3[F@2R.2U>L@2F.2D>R@2B.2U]', ('2U ', ' R ', '2F2', " R'", "2U'", ' R ', '2F2', " R'"))
            self._add_myperm2('CtrX3[F@2R.2U>R@2F.2U>L@2B.2D]', ("2U'", ' L ', '2B2', " L'", '2U ', ' L ', '2B2', " L'"))

            self._add_myperm2('CtrX3[F@2R.2U>L@2B.2D>R@2B.2U]', ('2D2', " R'", "2U'", ' R ', '2D2', " R'", '2U ', ' R '))
            self._add_myperm2('CtrX3[F@2R.2U>R@2B.2U>L@2B.2D]', ('2D2', " L'", '2U ', ' L ', '2D2', " L'", "2U'", ' L '))
            self._add_myperm2('CtrX3[F@2R.2U>L@2F.2D>R@2F.2U]', ('2F2', '2D ', '2R2', "2D'", ' R2', '2D ', '2R2', "2D'", ' R2', '2F2'))
            self._add_myperm2('CtrX3[F@2R.2U>R@2F.2U>L@2F.2D]', ('2B2', "2D'", '2R2', '2D ', ' L2', "2D'", '2R2', '2D ', ' L2', '2B2'))
            self._add_myperm2('CtrX3[F@2R.2U>R@2B.2D>L@2B.2U]', ('2B2', '2D ', '2R2', "2D'", ' R2', '2D ', '2R2', "2D'", ' R2', '2B2'))
            self._add_myperm2('CtrX3[F@2R.2U>L@2B.2U>R@2B.2D]', ('2F2', "2D'", '2R2', '2D ', ' L2', "2D'", '2R2', '2D ', ' L2', '2F2'))

            self._add_myperm2('CtrX3[F@2R.2U>L@2F.2U>R@2B.2U]', ('2B2', "2D'", ' F ', '2D ', '2B2', "2D'", " F'", '2D '))
            self._add_myperm2('CtrX3[B@2R.2U>R@2F.2U>L@2B.2U]', ('2B2', "2D'", " B'", '2D ', '2B2', "2D'", ' B ', '2D '))
            self._add_myperm2('CtrX3[F@2R.2U>L@2B.2D>R@2F.2D]', ('2D ', ' F ', "2D'", '2F2', '2D ', " F'", "2D'", '2F2'))
            self._add_myperm2('CtrX3[F@2R.2U>R@2B.2U>L@2F.2U]', ("2D'", ' F ', '2D ', '2B2', "2D'", " F'", '2D ', '2B2'))
            
            self._add_myperm2('CtrX3[F@2R.2U>L@2B.2U>R@2F.2U]', ('2D ', ' F2', "2D'", '2B2', '2D ', ' F2', "2D'", '2B2'))
            self._add_myperm2('CtrX3[B@2R.2U>R@2B.2U>L@2F.2U]', ('2D ', ' B2', "2D'", '2B2', '2D ', ' B2', "2D'", '2B2'))
            self._add_myperm2('CtrX3[F@2L.2U>R@2F.2D>L@2B.2D]', ('2F2', '2D ', ' F2', "2D'", '2F2', '2D ', ' F2', "2D'"))
            self._add_myperm2('CtrX3[F@2L.2U>L@2F.2U>R@2B.2U]', ('2B2', "2D'", ' F2', '2D ', '2B2', "2D'", ' F2', '2D '))

            self._add_myperm2('CtrX3[F@2R.2U>L@2F.2U>R@2B.2D]', ('2R2', '2D ', ' B ', '2U ', " B'", "2D'", ' B ', "2U'", " B'", '2R2'))
            self._add_myperm2('CtrX3[F@2R.2U>R@2F.2D>L@2B.2U]', ('2R2', "2D'", ' B ', "2U'", " B'", '2D ', ' B ', '2U ', " B'", '2R2'))
            self._add_myperm2('CtrX3[F@2R.2U>L@2B.2U>R@2F.2D]', ('2R2', '2D ', ' R ', '2B2', " R'", "2D'", ' R ', '2B2', '2R2', " R'"))
            self._add_myperm2('CtrX3[F@2R.2U>R@2B.2D>L@2F.2U]', ('2R2', "2D'", ' L ', '2F2', " L'", '2D ', ' L ', '2F2', '2R2', " L'"))
            self._add_myperm2('CtrX3[F@2R.2U>L@2F.2U>R@2F.2D]', ('2R2', '2U2', " R'", "2D'", ' R ', '2U2', " R'", '2D ', '2R2', ' R '))
            self._add_myperm2('CtrX3[F@2R.2U>R@2F.2D>L@2F.2U]', ('2R2', '2U2', " L'", '2D ', ' L ', '2U2', " L'", "2D'", '2R2', ' L '))
            self._add_myperm2('CtrX3[F@2R.2U>L@2F.2U>R@2F.2U]', ('2R2', " R'", '2U2', " R'", "2D'", ' R ', '2U2', " R'", '2D ', '2R2', ' R2'))
            self._add_myperm2('CtrX3[B@2R.2U>R@2F.2U>L@2F.2U]', ('2R2', ' L ', '2U2', ' L ', "2D'", " L'", '2U2', ' L ', '2D ', '2R2', ' L2'))
            self._add_myperm2('CtrX3[F@2R.2U>L@2F.2D>R@2F.2D]', (' L2', '2R2', '2D ', ' L ', '2U2', " L'", "2D'", ' L ', '2U2', ' L ', '2R2'))
            self._add_myperm2('CtrX3[F@2R.2U>R@2F.2U>L@2F.2U]', (' R2', '2R2', "2D'", ' R ', '2U2', " R'", '2D ', ' R ', '2U2', ' R ', '2R2'))



            if self.size % 2 == 1:
                self._add_myperm2('CtrPlus4s[L@2B.E<>R@2B.E;L@2F.E<>R@2F.E]', (' S2', '2U2', ' S2', '2U2'))
                self._add_myperm2('CtrPlus4s[L@S.2D<>R@2B.E;L@S.2U<>R@2F.E]', (' R ', ' E2', '2F2', ' E2', '2F2', " R'"))
                self._add_myperm2('CtrPlus4s[B@2R.E<>F@2R.E;L@S.2D<>R@S.2D]', (" E ","2R2"," E'","2R2"))
                self._add_myperm2('CtrPlus4s[B@2L.E<>F@2L.E;L@2B.E<>R@2B.E]', ('2L ', ' S ', '2L2', " S'", '2L '))
                self._add_myperm2('CtrPlus8s~v02', ('2D2', ' M2', '2D ', " R'", '2U ', ' M2', "2U'", ' R ', '2D '))
                self._add_myperm2('CtrPlus8s~v03', ('2U2', ' M2', "2U'", " R'", "2D'", ' M2', '2D ', ' R ', "2U'"))
                self._add_myperm2('CtrPlus4s[B@M.2D<>F@M.2D;L@2F.E<>R@2F.E]', ('2U ', ' M2', '2U2', ' M2', '2U '))
                self._add_myperm2('CtrPlus4s[B@M.2D<>F@M.2D;L@2B.E<>R@2B.E]', (" M'", "2D'", ' M ', '2D2', " M'", "2D'", ' M '))
                self._add_myperm2('CtrPlus4s[L@2B.E<>R@2B.E;L@S.2U<>R@S.2U]', (" S'", '2R2', ' S ', '2R ', " E'", '2R2', ' E ', "2R'"))
                self._add_myperm2('CtrPlus4s[L@2B.E<>R@2F.E;L@S.2U<>R@S.2U]', ('2B2', " R'", ' E2', ' S2', ' R ', '2B2', " R'", ' S2', ' E2', ' R '))
                self._add_myperm2('CtrPlus4s[L@2B.E<>R@S.2D;L@S.2U<>R@2F.E]', (' E2', ' S2', ' L ', '2U2', " L'", ' E2', ' S2', ' L ', '2U2', " L'"))

                
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>U@2R.S>L@2F.E;D@2R.S>R@2F.E>F@2R.E]', ("2R'", ' S ', '2R ', " S'"))
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>L@2F.E>U@2R.S;D@2R.S>F@2R.E>R@2F.E]', (' S ', "2R'", " S'", '2R '))
                self._add_myperm2('CtrPlus6p[3x2][B@M.2U>R@2F.E>L@2B.E;F@M.2U>L@2F.E>R@2B.E]', ('2U ', ' S2', "2U'", ' S2'))
                self._add_myperm2('CtrPlus6p[3x2][B@M.2U>R@2B.E>L@2F.E;F@M.2U>L@2B.E>R@2F.E]', (' S2', "2U'", ' S2', '2U '))
                self._add_myperm2('CtrPlus10p[5x2]~v01', ('2U ', " M'", '2B ', ' M ', "2B'", "2U'", " E'", "2F'", " E'", '2F ', ' E2'))
                self._add_myperm2('CtrPlus10p[5x2]~v02', ("2U'", ' M ', "2F'", " M'", ' E ', "2R'", " E'", '2R ', '2F ', '2U '))
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>D@2R.S>L@2F.E;F@2R.E>U@2R.S>R@2F.E]', ("2R'", ' S ', "2R'", " S'", '2R2'))
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>R@2B.E>D@2R.S;F@2R.E>L@2B.E>U@2R.S]', ('2R2', " S'", '2R ', ' S ', '2R '))
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>L@2B.E>U@2L.S;D@2L.S>F@2R.E>R@2B.E]', (" S'", '2R ', " S'", "2R'", ' S2'))
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>D@2L.S>R@2B.E;F@2R.E>U@2L.S>L@2B.E]', (' S2', "2R'", ' S ', '2R ', ' S '))



                self._add_myperm2('CtrPlus6p[3x2][B@M.2U>R@2B.E>R@2F.E;F@M.2U>L@2B.E>L@2F.E]', ('2U2', ' S2', '2U ', ' S2', '2U '))
                self._add_myperm2('CtrPlus6p[3x2][B@M.2U>L@2F.E>L@2B.E;F@M.2U>R@2F.E>R@2B.E]', ('2U ', ' S2', '2U ', ' S2', '2U2'))
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>L@2F.E>B@M.2U;F@2R.E>R@2F.E>F@M.2U]', ("2R'", " S'", "2U'", ' S ', '2U ', '2R '))
                self._add_myperm2('CtrPlus8s~v01', (" E'", "2L'", ' E2', "2F'", " E'", '2F ', '2L '))
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>L@S.2U>L@S.2D;F@2R.E>R@S.2U>R@S.2D]', ('2F ', " E'", '2F ', " E'", "2F'", ' E2', "2F'"))
                self._add_myperm2('CtrPlus6p[3x2][B@M.2D>B@M.2U>L@2F.E;F@2L.E>R@2F.E>F@M.2D]', ("2D'", ' F ', "2U'", ' S2', '2U ', " F'", ' S2', '2D '))
                self._add_myperm2('CtrPlus6p[3x2][B@2L.E>L@2F.E>B@M.2D;F@M.2D>F@M.2U>R@2F.E]', ("2D'", " B'", "2U'", ' S2', '2U ', ' B ', ' S2', '2D '))
                self._add_myperm2('CtrPlus6p[3x2][B@2L.E>B@M.2U>R@2F.E;F@2L.E>F@2R.E>L@2F.E]', ('2L ', " F'", '2U ', ' S ', "2U'", ' F ', " S'", "2L'"))
                self._add_myperm2('CtrPlus6p[3x2][B@2L.E>B@2R.E>R@2F.E;F@2L.E>F@M.2U>L@2F.E]', ('2L ', ' B ', '2U ', ' S ', "2U'", " B'", " S'", "2L'"))
                self._add_myperm2('CtrPlus6p[3x2][B@M.2D>B@M.2U>R@2F.E;F@2R.E>L@2F.E>F@M.2D]', ('2D ', " F'", '2U ', ' S2', "2U'", ' F ', ' S2', "2D'"))
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>R@2F.E>B@M.2D;F@M.2D>F@M.2U>L@2F.E]', ('2D ', ' B ', '2U ', ' S2', "2U'", " B'", ' S2', "2D'"))
                self._add_myperm2('CtrPlus6p[3x2][B@M.2U>L@2F.E>L@2B.E;F@M.2D>R@2F.E>R@2B.E]~v01', (' F2', '2U ', ' S2', '2U ', ' S2', '2U2', ' F2'))
                self._add_myperm2('CtrPlus6p[3x2][B@M.2D>L@2F.E>L@2B.E;F@M.2U>R@2F.E>R@2B.E]~v01', (' B2', '2U ', ' S2', '2U ', ' S2', '2U2', ' B2'))
                self._add_myperm2('CtrPlus6p[3x2][B@M.2U>L@2F.E>L@2B.E;F@M.2D>R@2F.E>R@2B.E]~v02', (' F2', '2U ', ' S2', "2U'", ' F2', '2D2', ' S2', '2D2'))
                self._add_myperm2('CtrPlus6p[3x2][B@M.2D>L@2F.E>L@2B.E;F@M.2U>R@2F.E>R@2B.E]~v02', (' B2', '2U ', ' S2', "2U'", ' B2', '2D2', ' S2', '2D2'))
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>B@M.2D>R@2F.E;F@M.2D>L@2F.E>F@M.2U]', ('2D ', ' S2', ' B ', '2U ', ' S2', "2U'", " B'", "2D'"))
                self._add_myperm2('CtrPlus6p[3x2][B@M.2D>R@2F.E>B@M.2U;F@2R.E>F@M.2D>L@2F.E]', ('2D ', ' S2', " F'", '2U ', ' S2', "2U'", ' F ', "2D'"))


                self._add_myperm2('MidCenterBar(VV)', (" E'", ' F ', ' D ', " U'", ' E ', " R'", ' E ', ' R ', " D'", ' U ', " E'", " F'"))
                self._add_myperm2('MidCenterBar(HV)', (' F ', ' E ', " U'", ' D ', " R'", " E'", ' R ', " E'", ' U ', " D'", " F'", ' E '))
                self._add_myperm2('MidCenterBar(HH)', (' E ', " U'", ' D ', " R'", " E'", ' R ', " E'", ' U ', " D'", " F'", ' E ', ' F '))

                self._add_myperm2('MidCenterBar-Opp(VV)', (' S2', ' R ', ' B2', ' F2', ' S2', " L'", ' S2', ' L ', ' B2', ' F2', ' S2', " R'"))
                self._add_myperm2('MidCenterBar-Opp(HV)', (' R ', ' S2', ' F2', ' B2', " L'", ' S2', ' L ', ' S2', ' F2', ' B2', " R'", ' S2'))

                self._add_myperm2('MidCenterBar-Adjacent3Center-A', (' R ', ' F ', ' D ', " U'", ' E ', " R'", ' E ', ' R ', " D'", ' U ', " E'", " F'", " E'", " R'"))
                self._add_myperm2('MidCenterBar-Adjacent3Center-B', (' F ', ' D ', " U'", ' E ', " R'", ' E ', ' R ', " D'", ' U ', " E'", " F'", " E'"))
                self._add_myperm2('MidCenterBar-Adjacent3Center-C', (' R ', ' D ', " U'", ' E ', " R'", ' E ', ' R ', " D'", ' U ', " E'", " F'", " E'", ' F ', " R'"))
                self._add_myperm2('MidCenterBar-Adjacent3Center-D', (' D ', " U'", ' E ', " R'", ' E ', ' R ', " D'", ' U ', " E'", " F'", " E'", ' F '))
                self._add_myperm2('MidCenterBar-Adjacent3Center-E', (" L'", ' R ', ' F ', ' D ', " U'", ' E ', " R'", ' E ', ' R ', " D'", ' U ', " E'", " F'", " E'", " R'", ' L '))
                self._add_myperm2('MidCenterBar-Adjacent3Center-F', (" L'", ' F ', ' D ', " U'", ' E ', " R'", ' E ', ' R ', " D'", ' U ', " E'", " F'", " E'", ' L '))
                self._add_myperm2('MidCenterBar-Adjacent3Center-G', (' L ', " R'", " U'", ' D ', ' E ', ' R ', ' E ', " R'", ' U ', " D'", " E'", ' F ', " E'", " F'", ' R ', " L'"))
                self._add_myperm2('MidCenterBar-Adjacent3Center-H', (" L'", ' D ', " U'", ' E ', " R'", ' E ', ' R ', " D'", ' U ', " E'", " F'", " E'", ' F ', ' L '))

                self._add_myperm2('MidCenterBar-Adjacent3Center-OA', (' E ', ' U ', " S'", " U'", " B'", ' F ', ' S ', ' R ', ' S ', " R'", ' B ', " F'", " S'", " E'"))
                self._add_myperm2('MidCenterBar-Adjacent3Center-OB', (' E ', ' S ', ' F ', " B'", ' R ', " S'", " R'", " S'", " F'", ' B ', ' U ', ' S ', " U'", " E'"))
                self._add_myperm2('MidCenterBar-Adjacent3Center-OC', (' E ', " S'", " U'", " B'", ' F ', ' S ', ' R ', ' S ', " R'", ' B ', " F'", " S'", ' U ', " E'"))
                self._add_myperm2('MidCenterBar-Adjacent3Center-OD', (' E ', " U'", ' S ', ' F ', " B'", ' R ', " S'", " R'", " S'", " F'", ' B ', ' U ', ' S ', " E'"))



                self._add_myperm2('CtrPlus3[L@S.2D>R@S.2U>R@2B.E]', (' R ', '2F2', " R'", ' E2', ' R ', '2F2', " R'", ' E2'))
                self._add_myperm2('CtrPlus3[L@2F.E>L@S.2D>R@2B.E]', (' R ', '2F2', " R'", ' S2', ' R ', '2F2', " R'", ' S2'))
                self._add_myperm2('CtrPlus4s[L@2F.E<>L@S.2U;L@S.2D<>R@2B.E]', ('2U2', ' L ', ' S2', '2U2', ' S2', '2U2', " L'", '2U2'))
                self._add_myperm2('CtrPlus3[L@2F.E>R@S.2U>R@2F.E]', ('2U2', ' R ', ' S2', " R'", '2U2', ' R ', ' S2', " R'"))
                self._add_myperm2('CtrPlus3[L@2F.E>R@2F.E>L@S.2D]', ('2U2', ' R ', ' E2', " R'", '2U2', ' R ', ' E2', " R'"))
                self._add_myperm2('CtrPlus3[L@2F.E>R@S.2D>R@2B.E]', (' S2', ' R ', '2U2', " R'", ' S2', ' R ', '2U2', " R'"))
                self._add_myperm2('CtrPlus3[L@2F.E>R@2B.E>L@S.2D]', (' S2', ' R ', '2F2', " R'", ' S2', ' R ', '2F2', " R'"))
                

                self._add_myperm2('CtrPlus3[F@2R.E>R@2F.E>R@2B.E]', (' S ', "2R'", " S'", ' R2', ' S ', '2R ', " S'", ' R2'))
                self._add_myperm2('CtrPlus3[F@2L.E>R@2B.E>R@2F.E]', (' S ', '2L ', " S'", ' R2', ' S ', "2L'", " S'", ' R2'))
                self._add_myperm2('CtrPlus3[F@M.2D>R@2B.E>F@M.2U]~v01', ('2U ', ' S2', "2U'", ' F2', '2U ', ' S2', "2U'", ' F2'))
                self._add_myperm2('CtrPlus3[F@M.2D>F@M.2U>R@2F.E]', ("2D'", ' S2', '2D ', ' F2', "2D'", ' S2', '2D ', ' F2'))
                self._add_myperm2('CtrPlus3[F@M.2U>R@S.2U>R@2F.E]', ('2U ', ' F ', " E'", " F'", "2U'", ' F ', ' E ', " F'"))
                self._add_myperm2('CtrPlus3[F@M.2D>R@S.2D>R@2B.E]', ("2D'", ' F ', " E'", " F'", '2D ', ' F ', ' E ', " F'"))
                self._add_myperm2('CtrPlus3[F@M.2U>R@2B.E>R@2F.E]', (' S ', ' D2', " S'", "2U'", ' S ', ' D2', " S'", '2U '))
                self._add_myperm2('CtrPlus3[F@M.2D>R@2B.E>F@M.2U]~v02', (' M ', ' D2', " M'", "2D'", ' M ', ' D2', " M'", '2D '))

                
                self._add_myperm2('CtrPlus3[F@M.2D>F@M.2U>R@2B.E]~v01', (' F2', '2U ', ' S2', "2U'", ' F2', '2U ', ' S2', "2U'"))
                self._add_myperm2('CtrPlus3[F@M.2D>R@2F.E>F@M.2U]', (' F2', "2D'", ' S2', '2D ', ' F2', "2D'", ' S2', '2D '))
                self._add_myperm2('CtrPlus3[F@M.2U>R@2F.E>R@2B.E]', ("2U'", ' S ', ' D2', " S'", '2U ', ' S ', ' D2', " S'"))
                self._add_myperm2('CtrPlus3[F@M.2D>F@M.2U>R@2B.E]~v02', ("2D'", ' M ', ' D2', " M'", '2D ', ' M ', ' D2', " M'"))

                self._add_myperm2('CtrPlus3[F@2R.E>R@2B.E>R@2F.E]', (' R2', ' S ', "2R'", " S'", ' R2', ' S ', '2R ', " S'"))
                self._add_myperm2('CtrPlus3[F@2L.E>R@2F.E>R@2B.E]', (' R2', ' S ', '2L ', " S'", ' R2', ' S ', "2L'", " S'"))

                self._add_myperm2('CtrPlus3[F@2L.E>F@2R.E>R@2B.E]', (' F2', '2R ', " S'", "2R'", ' F2', '2R ', ' S ', "2R'"))
                self._add_myperm2('CtrPlus3[F@2L.E>R@2F.E>F@2R.E]', (' F2', "2L'", " S'", '2L ', ' F2', "2L'", ' S ', '2L '))

                self._add_myperm2('CtrPlus3[F@2R.E>R@2B.E>F@M.2D]', (" R'", ' E ', ' R ', '2D ', " R'", " E'", ' R ', "2D'"))
                self._add_myperm2('CtrPlus3[F@2L.E>R@2F.E>F@M.2U]', (" R'", ' E ', ' R ', "2U'", " R'", " E'", ' R ', '2U '))

                self._add_myperm2('CtrPlus3[F@2R.E>R@S.2U>R@2B.E]', (' F ', "2D'", " F'", " E'", ' F ', '2D ', " F'", ' E '))
                self._add_myperm2('CtrPlus3[F@2L.E>R@S.2D>R@2F.E]', (' F ', '2U ', " F'", " E'", ' F ', "2U'", " F'", ' E '))




                self._add_myperm2('CtrPlus3[F@2L.E>F@2R.E>R@S.2U]', (' F2', "2B'", ' E ', '2B ', " E'", ' F2', ' E ', "2B'", " E'", '2B '))
                self._add_myperm2('CtrPlus3[F@2L.E>R@S.2D>F@2R.E]', (' F2', '2F ', ' E ', "2F'", " E'", ' F2', ' E ', '2F ', " E'", "2F'"))
                self._add_myperm2('CtrPlus3[F@2L.E>R@2F.E>R@S.2D]', (" E'", ' F ', '2U ', " F'", ' E ', ' F ', "2U'", " F'"))
                self._add_myperm2('CtrPlus3[F@2R.E>R@2B.E>R@S.2U]', (" E'", ' F ', "2D'", " F'", ' E ', ' F ', '2D ', " F'"))
                self._add_myperm2('CtrPlus3[F@2L.E>R@S.2U>F@2R.E]', ("2B'", ' E ', '2B ', " E'", ' F2', ' E ', "2B'", " E'", '2B ', ' F2'))
                self._add_myperm2('CtrPlus3[F@2L.E>F@2R.E>R@S.2D]', ('2F ', ' E ', "2F'", " E'", ' F2', ' E ', '2F ', " E'", "2F'", ' F2'))

                self._add_myperm2('CtrPlus3[F@2R.E>U@M.2F>R@2F.E]', (" F ","2D "," R2","2D'"," M ","2D "," R2","2D'"," M'"," F'"))
                self._add_myperm2('CtrPlus3[F@2L.E>U@M.2B>R@2B.E]', (" F ","2U'"," R2","2U "," M ","2U'"," R2","2U "," M'"," F'"))
                self._add_myperm2('CtrPlus3[F@2R.E>R@2F.E>U@M.2F]', (' R ', ' E ', '2B ', ' U2', "2B'", " E'", '2B ', ' U2', "2B'", " R'"))
                self._add_myperm2('CtrPlus3[F@2L.E>R@2B.E>U@M.2B]', (' R ', ' E ', "2F'", ' U2', '2F ', " E'", "2F'", ' U2', '2F ', " R'"))

                self._add_myperm2('CtrPlus3[F@2L.E>U@M.2F>R@2F.E]', (" F'","2D "," R2","2D'"," M ","2D "," R2","2D'"," M'"," F "))
                self._add_myperm2('CtrPlus3[F@2R.E>U@M.2B>R@2B.E]', (" F'","2U'"," R2","2U "," M ","2U'"," R2","2U "," M'"," F "))
                self._add_myperm2('CtrPlus3[F@2R.E>R@2B.E>U@M.2F]', (" R'", ' E ', '2B ', ' U2', "2B'", " E'", '2B ', ' U2', "2B'", ' R '))
                self._add_myperm2('CtrPlus3[F@2L.E>R@2F.E>U@M.2B]', (" R'", ' E ', "2F'", ' U2', '2F ', " E'", "2F'", ' U2', '2F ', ' R '))
            
                self._add_myperm2('CtrPlus3[F@2R.E>U@2L.S>R@2F.E]', ('2L ', ' F2', "2L'", " S'", '2L ', ' F2', "2L'", ' S '))
                self._add_myperm2('CtrPlus3[F@2L.E>U@2R.S>R@2B.E]', ("2R'", ' F2', '2R ', " S'", "2R'", ' F2', '2R ', ' S '))
                self._add_myperm2('CtrPlus3[F@2R.E>R@2F.E>U@2L.S]', (" S'", '2L ', ' F2', "2L'", ' S ', '2L ', ' F2', "2L'"))
                self._add_myperm2('CtrPlus3[F@2L.E>R@2B.E>U@2R.S]', (" S'", "2R'", ' F2', '2R ', ' S ', "2R'", ' F2', '2R '))

                self._add_myperm2('CtrPlus3[F@2L.E>R@2B.E>U@2L.S]', ('2L2', ' U ', ' S ', " U'", "2L'", ' U ', " S'", " U'", "2L'"))
                self._add_myperm2('CtrPlus3[F@2R.E>R@2F.E>U@2R.S]', ('2R2', ' U ', ' S ', " U'", '2R ', ' U ', " S'", " U'", '2R '))
                self._add_myperm2('CtrPlus3[F@2L.E>U@2L.S>R@2B.E]', ('2L ', ' U ', ' S ', " U'", '2L ', ' U ', " S'", " U'", '2L2'))
                self._add_myperm2('CtrPlus3[F@2R.E>U@2R.S>R@2F.E]', ("2R'", ' U ', ' S ', " U'", "2R'", ' U ', " S'", " U'", '2R2'))
                

                self._add_myperm2('CtrPlus3[F@2L.E>U@2L.S>R@2F.E]', ("2L'"," D'"," S'",' D ','2L '," D'",' S ',' D '))
                self._add_myperm2('CtrPlus3[F@2R.E>U@2R.S>R@2B.E]', ("2R "," D'"," S'",' D ',"2R'"," D'"," S ",' D '))
                self._add_myperm2('CtrPlus3[F@2R.E>R@2B.E>U@2R.S]', (" S'", " D'", '2R ', ' D ', ' S ', " D'", "2R'", ' D '))
                self._add_myperm2('CtrPlus3[F@2L.E>R@2F.E>U@2L.S]', (" S'", " D'", "2L'", ' D ', ' S ', " D'", '2L ', ' D '))

                self._add_myperm2('CtrPlus3[F@2L.E>U@2R.S>R@2F.E]', (" U2","2L'"," D'"," S'",' D ','2L '," D'",' S ',' D '," U2"))
                self._add_myperm2('CtrPlus3[F@2R.E>U@2L.S>R@2B.E]', (" U2","2R "," D'"," S'",' D ',"2R'"," D'"," S ",' D '," U2"))
                self._add_myperm2('CtrPlus3[F@2R.E>R@2B.E>U@2L.S]', (' U2', " S'", '2R ', ' S ', " U'", " S'", ' U ', "2R'", " U'", ' S ', " U'"))
                self._add_myperm2('CtrPlus3[F@2L.E>R@2F.E>U@2R.S]', (' U2', " S'", "2L'", ' S ', " U'", " S'", ' U ', '2L ', " U'", ' S ', " U'"))

                self._add_myperm2('CtrPlus3[F@M.2U>L@S.2D>R@2F.E]', ('2U ', ' F ', ' E ', " F'", "2U'", ' F ', " E'", " F'"))
                self._add_myperm2('CtrPlus3[F@M.2U>L@S.2U>R@2F.E]', ('2U ', " F'", ' E ', ' F ', "2U'", " F'", " E'", ' F '))
                self._add_myperm2('CtrPlus3[F@M.2U>L@S.2U>R@2B.E]', (' F2', "2D'", ' F ', ' E ', " F'", '2D ', ' F ', " E'", ' F '))
                self._add_myperm2('CtrPlus3[F@M.2U>L@S.2D>R@2B.E]', (' F2', "2D'", " F'", ' E ', ' F ', '2D ', " F'", " E'", " F'"))
                self._add_myperm2('CtrPlus3[F@M.2U>L@2F.E>R@S.2U]', ('2U ', ' R ', ' S2', " R'", "2U'", ' R ', ' S2', " R'"))
                self._add_myperm2('CtrPlus3[F@M.2U>L@2F.E>R@S.2D]', ('2U ', " R'", ' S2', ' R ', "2U'", " R'", ' S2', ' R '))

                self._add_myperm2('CtrPlus3[F@2R.E>L@2F.E>R@2F.E]', ('2U2', ' R ', ' E ', " R'", '2U2', ' R ', " E'", " R'"))
                self._add_myperm2('CtrPlus3[F@2L.E>L@2F.E>R@2F.E]', ('2U2', " R'", ' E ', ' R ', '2U2', " R'", " E'", ' R '))
                self._add_myperm2('CtrPlus3[F@2L.E>L@2F.E>R@S.2U]', (" R'", '2U2', " R'", ' E ', ' R ', '2U2', " R'", " E'", ' R2'))
                self._add_myperm2('CtrPlus3[F@2R.E>L@2F.E>R@S.2D]', (' R ', '2U2', ' R ', ' E ', " R'", '2U2', ' R ', " E'", ' R2'))

                self._add_myperm2('CtrPlus3[F@2R.E>L@2F.E>R@S.2U]', (" E'", " F'", "2U'", ' F ', ' E ', " F'", '2U ', ' F '))
                self._add_myperm2('CtrPlus3[F@2L.E>L@2F.E>R@S.2D]', (" E'", ' F ', "2U'", " F'", ' E ', ' F ', '2U ', " F'"))
                self._add_myperm2('CtrPlus3[F@2R.E>L@S.2D>R@2F.E]', (" E'", " R'", '2F2', ' R ', ' E ', " R'", '2F2', ' R '))
                self._add_myperm2('CtrPlus3[F@2L.E>L@S.2U>R@2F.E]', (" E'", ' R ', '2B2', " R'", ' E ', ' R ', '2B2', " R'"))

                self._add_myperm2('CtrPlus3[F@M.2U>L@S.2D>R@S.2U]', (' E2', " R'", "2U'", ' R ', ' E2', " R'", '2U ', ' R '))
                self._add_myperm2('CtrPlus3[F@M.2U>L@S.2U>R@S.2D]', (' E2', ' R ', "2U'", " R'", ' E2', ' R ', '2U ', " R'"))

                self._add_myperm2('CtrPlus3[F@M.2U>L@2B.E>R@2B.E]', ('2D ', " M'", ' U2', ' M ', '2D2', " M'", ' U2', ' M ', '2D '))
                self._add_myperm2('CtrPlus3[F@M.2U>L@2F.E>R@2F.E]', ('2D ', ' S2', "2D'", ' F2', '2D ', ' S2', '2D2', ' S2', '2D ', ' F2', "2D'", ' S2', '2D '))
                self._add_myperm2('CtrPlus3[F@M.2U>L@2F.E>R@2B.E]', (' S2', "2D'", ' F2', '2D ', ' S2', "2D'", ' F2', '2D '))
                self._add_myperm2('CtrPlus3[F@M.2U>L@2B.E>R@2F.E]', ('2D ', ' F2', "2D'", ' S2', '2D ', ' F2', "2D'", ' S2'))
                self._add_myperm2('CtrPlus3[F@M.2U>L@S.2D>R@S.2D]', (" M'", '2F ', ' M ', ' F2', " M'", '2F2', ' M ', ' F2', " M'", '2F ', ' M '))
                self._add_myperm2('CtrPlus3[F@M.2U>L@S.2U>R@S.2U]', (' M ', '2B ', " M'", ' F2', ' M ', '2B2', " M'", ' F2', ' M ', '2B ', " M'"))

                self._add_myperm2('CtrPlus3[F@2L.E>L@S.2D>R@S.2D]', ('2F2', ' L ', ' E ', " L'", '2F2', ' L ', " E'", " L'"))
                self._add_myperm2('CtrPlus3[F@2R.E>L@S.2U>R@S.2U]', ('2B2', ' L ', ' E ', " L'", '2B2', ' L ', " E'", " L'"))
                self._add_myperm2('CtrPlus3[F@2R.E>L@S.2D>R@S.2D]', (' R ', " E'", " R'", '2F2', ' R ', ' E ', " R'", '2F2'))
                self._add_myperm2('CtrPlus3[F@2L.E>L@S.2U>R@S.2U]', (' R ', " E'", " R'", '2B2', ' R ', ' E ', " R'", '2B2'))
                self._add_myperm2('CtrPlus3[F@2L.E>L@S.2U>R@S.2D]', (' L2', '2F2', ' L ', ' E ', " L'", '2F2', ' L ', " E'", ' L '))
                self._add_myperm2('CtrPlus3[F@2R.E>L@S.2D>R@S.2U]', (' L2', '2B2', ' L ', ' E ', " L'", '2B2', ' L ', " E'", ' L '))

                self._add_myperm2('CtrPlus3[F@2R.E>L@2B.E>R@2F.E]', (' S2', '2U ', " F'", "2U'", ' S2', '2U ', ' F ', "2U'"))
                self._add_myperm2('CtrPlus3[F@2L.E>L@2B.E>R@2F.E]', (' S2', '2U ', ' F ', "2U'", ' S2', '2U ', " F'", "2U'"))
                
            if self.size >= 6:
                self._add_myperm2('CtrObl6p[3x2][L@3B.2D>L@3F.2U>R@3B.2U;L@3B.2U>R@3B.2D>R@3F.2U]', ('2B2', '3D2', '2B2', '3D2'))
                self._add_myperm2('CtrObl6p[3x2][L@2B.3D>R@3F.2U>L@2F.3U;L@2B.3U>R@3F.2D>R@3B.2U]', (' L ', '2B2', '3U2', '2B2', '3U2', " L'"))
                self._add_myperm2('CtrObl6p[3x2][L@2B.3D>R@2B.3U>R@2F.3D;L@2B.3U>R@2F.3U>L@2F.3D]', (' L2', '2U2', '3B2', '2U2', '3B2', ' L2'))
                self._add_myperm2('CtrObl8s~v03', ('2D2', '3R2', '2D ', " L'", '2U ', '3R2', "2U'", ' L ', '2D '))
                self._add_myperm2('CtrObl8s~v04', ('2U2', '3R2', "2U'", " R'", "2D'", '3R2', '2D ', ' R ', "2U'"))
                self._add_myperm2('CtrObl8s~v05', ('2B2', '3R2', '2B ', " R'", '2F ', '3R2', "2F'", ' R ', '2B '))
                self._add_myperm2('CtrObl8s~v06', ('2B2', '3R2', "2B'", " L'", "2F'", '3R2', '2F ', ' L ', "2B'"))
                self._add_myperm2('CtrObl6p[3x2][L@3B.2D>R@3B.2U>R@2F.3U;L@3B.2U>R@2B.3U>R@2F.3D]', ('2B2', ' R ', '2B2', '3D2', '2B2', " R'", '2B2', ' R ', '3D2', " R'"))
                self._add_myperm2('CtrObl6p[3x2][L@3F.2D>R@2F.3D>R@3F.2U;L@3F.2U>R@3F.2D>R@3B.2U]', (' R ', '2B2', " R'", '2B2', '3U2', '2B2', ' R ', '2B2', " R'", '3U2'))
                self._add_myperm2('CtrObl8p[3+5]~v02', ('2D2', '3F2', '2D2', '3F2', ' L ', '3D2', '2B2', '3D2', '2B2', " L'"))
                self._add_myperm2('CtrObl6p[3x2][L@2B.3D>L@3B.2U>R@2B.3U;L@2B.3U>R@2B.3D>R@3B.2U]', ('2B ', "2D'", '3R ', '2D2', "3R'", "2D'", "2B'"))
                self._add_myperm2('CtrObl6p[3x2][L@2B.3D>R@2B.3U>R@3F.2U;L@2B.3U>L@3F.2U>R@2B.3D]', ('2B ', "2D'", "3L'", '2D2', '3L ', "2D'", "2B'"))
                self._add_myperm2('CtrObl6p[3x2][L@2B.3D>L@3B.2U>R@3F.2U;L@2B.3U>R@3B.2U>R@2F.3U]', (' R ', '2B ', "2D'", '3R ', '2D2', "3R'", "2D'", "2B'", " R'"))
                self._add_myperm2('CtrObl6p[3x2][L@2B.3D>R@3F.2U>R@2F.3D;L@2B.3U>L@3F.2U>R@3B.2U]', (' R ', '2B ', "2D'", "3L'", '2D2', '3L ', "2D'", "2B'", " R'"))
                self._add_myperm2('CtrObl6p[3x2][L@2B.3D>L@3B.2U>R@2F.3D;L@2B.3U>R@2F.3U>R@3F.2D]', (' R2', '2B ', "2D'", '3R ', '2D2', "3R'", "2D'", "2B'", ' R2'))
                self._add_myperm2('CtrObl6p[3x2][L@2B.3D>R@2F.3D>R@3B.2D;L@2B.3U>L@3F.2U>R@2F.3U]', (' R2', '2B ', "2D'", "3L'", '2D2', '3L ', "2D'", "2B'", ' R2'))
                self._add_myperm2('CtrObl8s~v07', ('3B2', '2D2', '3B2', '2D2', ' R2', '3F2', '2D2', '3F2', '2D2', ' R2'))
                self._add_myperm2('CtrObl8s~v08', (' R ', '3B2', '2D2', '3B2', '2D2', ' R2', '3F2', '2D2', '3F2', '2D2', ' R '))
                self._add_myperm2('CtrObl4s[L@2F.3D<>R@2B.3D;L@3F.2U<>R@3B.2U]', (" R'", '2B2', '2U2', ' R ', '3F2', " R'", '2U2', '2B2', ' R ', '3F2'))
                self._add_myperm2('CtrObl4s[L@2F.3D<>R@3B.2U;L@3F.2U<>R@2F.3U]', ('2B2', '2U2', ' R ', '3F2', " R'", '2U2', '2B2', ' R ', '3F2', " R'"))
                self._add_myperm2('CtrObl4s[L@2B.3U<>R@2F.3U;L@3F.2U<>R@3B.2U]', (' L ', '2U2', '2B2', ' L ', '3D2', " L'", '2B2', '2U2', ' L ', '3D2', ' L2'))
                self._add_myperm2('CtrObl8p[3+5]~v01', (" R'", '2U2', ' R ', '3B2', '2U2', '3B2', '2U2', '3F2', " R'", '2U2', ' R ', '3F2'))


                self._add_myperm2('CtrObl10p[5x2]~v03', ("3U2","2R2","3U'","2R2","3U'","2R2","3U'","2R2","3U ")      )
                
                self._add_myperm2('CtrObl6p[3x2][B@3R.2U>L@2F.3U>D@2L.3B;F@3R.2U>R@2F.3U>U@2L.3B]', ("2U'", "3B'", '2U ', '3B '))
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>D@2L.3B>L@2F.3U;F@2L.3U>U@2L.3B>R@2F.3U]', ("2L'", '3B ', '2L ', "3B'"))
                self._add_myperm2('CtrObl6p[3x2][B@3R.2U>L@2F.3U>R@2B.3U;F@3R.2U>R@2F.3U>L@2B.3U]', ("2U'", '3B2', '2U ', '3B2'))
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>L@3B.2U>R@3F.2U;F@2L.3U>R@3B.2U>L@3F.2U]', ('2B2', '3U ', '2B2', "3U'"))
                self._add_myperm2('CtrObl10p[5x2]~v01', ('2F ', '3R ', '2U ', "3R'", "2U'", "2F'", "3B'", '2R2', "3B'", '2R2', '3B2'))
                self._add_myperm2('CtrObl10p[5x2]~v02', ("3B'", '2R ', "3D'", "2R'", '3D ', "2B'", '3B ', '3L2', "2B'", '3L2', '2B2'))
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>D@3R.2B>L@3B.2U;F@2L.3U>U@3R.2B>R@3B.2U]', ('2B ', '3U ', '2B ', "3U'", '2B2'))
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>L@3B.2U>D@2L.3B;F@2L.3U>R@3B.2U>U@2L.3B]', ("2L'", '3D ', "2L'", "3D'", '2L2'))
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>L@3B.2U>D@3R.2B;F@2L.3U>R@3B.2U>U@3R.2B]', ('2B2', '3U ', "2B'", "3U'", "2B'"))
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>D@2L.3B>L@3B.2U;F@2L.3U>U@2L.3B>R@3B.2U]', ('2L2', '3D ', '2L ', "3D'", '2L '))
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>R@3F.2U>L@3F.2D;F@2L.3U>L@3F.2U>R@3F.2D]~v01', ('2F2', "3U'", '2F2', "3U'", '2F2', '3U2', '2F2'))
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>R@3F.2U>L@3F.2D;F@2L.3U>L@3F.2U>R@3F.2D]~v02', ('2B2', '3U2', '2B2', "3U'", '2B2', "3U'", '2B2'))
                self._add_myperm2('CtrObl6p[3x2][B@3R.2U>R@2B.3U>L@2B.3D;F@3R.2U>L@2B.3U>R@2B.3D]', ("2D'", '3R2', '2D2', '3R2', "2D'"))
                self._add_myperm2('CtrObl6p[3x2][B@3R.2U>L@3B.2U>R@3F.2U;F@3R.2U>R@3B.2U>L@3F.2U]', ('2B ', '3R ', '2B2', "3R'", '2B '))
                self._add_myperm2('CtrObl6p[3x2][B@3R.2U>R@3F.2U>L@3B.2U;F@3R.2U>L@3F.2U>R@3B.2U]', ("2B'", '3R ', '2B2', "3R'", "2B'"))

                self._add_myperm2('CtrObl6p[3x2][B@3R.2U>L@2B.3D>L@2F.3U;F@3R.2U>R@2B.3D>R@2F.3U]', ('2U2', '3F2', "2U'", '3F2', "2U'"))
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>L@3F.2D>L@3B.2U;F@2L.3U>R@3F.2D>R@3B.2U]', ('3U ', '2B2', '3U ', '2B2', '3U2'))
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>L@2F.3U>B@3R.2U;F@2L.3U>R@2F.3U>F@3R.2U]', ("2L'", "3B'", "2U'", '3B ', '2U ', '2L '))
                self._add_myperm2('CtrObl6p[3x2][B@3R.2U>L@2F.3U>L@3B.2U;F@3R.2U>R@2F.3U>R@3B.2U]', ('2B ', "3R'", '2U ', '3R ', "2U'", "2B'"))
                self._add_myperm2('CtrObl6p[3x2][B@3L.2D>B@3R.2U>L@2F.3U;F@2L.3U>R@2F.3U>F@3L.2D]', ("2D'", ' F ', "2U'", '3B2', '2U ', " F'", '3B2', '2D '))
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>L@2F.3U>B@3L.2D;F@3L.2D>F@3R.2U>R@2F.3U]', ("2D'", " B'", "2U'", '3B2', '2U ', ' B ', '3B2', '2D '))
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>B@2R.3D>L@2F.3U;F@2L.3U>F@3R.2U>R@2F.3U]', ("2L'", ' B ', "2U'", "3B'", '2U ', " B'", '3B ', '2L '))
                self._add_myperm2('CtrObl6p[3x2][B@3R.2U>L@3F.2D>L@3B.2U;F@3R.2U>R@3F.2D>R@2F.3U]', ('2F ', ' L ', '2U ', '3R ', "2U'", " L'", "3R'", "2F'"))
                self._add_myperm2('CtrObl6p[3x2][B@2R.3D>L@2F.3U>B@3L.2D;F@3L.2D>F@3R.2U>R@2F.3U]', ("2D'", ' B ', "2U'", '3B2', '2U ', " B'", '3B2', '2D '))
                self._add_myperm2('CtrObl6p[3x2][B@3R.2U>L@2B.3D>L@3B.2U;F@3R.2U>R@2B.3D>R@2F.3U]', ('2D ', ' L ', '2U ', '3R2', "2U'", " L'", '3R2', "2D'"))
                self._add_myperm2('CtrObl6p[3x2][B@3L.2D>B@3R.2U>L@2B.3D;F@3L.2D>F@3R.2U>R@2F.3U]', (' R2', '2D ', '3R2', '2D ', '3R2', '2D2', ' R2'))
                self._add_myperm2('CtrObl6p[3x2][B@3L.2D>L@2B.3D>L@2F.3U;F@3R.2U>R@2B.3D>R@2F.3U]', (' F2', "2D'", '3B2', "2D'", '3B2', '2D2', ' F2'))
                self._add_myperm2('CtrObl8s~v01', (' R2', '2D ', '3R2', "2D'", ' R2', '2U2', '3R2', '2U2'))
                self._add_myperm2('CtrObl8s~v02', (' F2', "2D'", '3B2', '2D ', ' F2', '2U2', '3B2', '2U2'))
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>L@3B.2U>B@2R.3D;F@2L.3U>R@3B.2U>F@3R.2U]', ('3U ', '2B2', ' F ', '3D ', '2B2', "3D'", " F'", "3U'"))
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>L@3B.2U>B@3R.2U;F@2L.3U>R@3B.2U>F@2R.3D]', ('3U ', '2B2', " B'", '3D ', '2B2', "3D'", ' B ', "3U'"))
                

                self._add_myperm2('CtrObl3[L@2F.3D>R@3B.2U>L@3B.2D]', (" L'", '2B2', ' L ', '3D2', " L'", '2B2', ' L ', '3D2'))
                self._add_myperm2('CtrObl6p[3x2][L@2B.3D>L@3F.2D>L@2F.3U;L@2F.3D>R@3B.2U>L@3F.2U]', ('3U2', " L'", '3U2', '2B2', '3U2', '2B2', ' L ', '3U2'))
                self._add_myperm2('CtrObl3[L@2F.3D>R@3B.2U>L@3F.2U]', (' R ', '3F2', " R'", '2B2', ' R ', '3F2', " R'", '2B2'))
                self._add_myperm2('CtrObl3[L@2F.3D>L@3F.2U>R@3B.2U]', ('2B2', ' L ', '3D2', " L'", '2B2', ' L ', '3D2', " L'"))
                self._add_myperm2('CtrObl3[L@2B.3U>L@3F.2U>R@3B.2U]', ('2B2', " L'", '3D2', ' L ', '2B2', " L'", '3D2', ' L '))
                self._add_myperm2('CtrObl6p[3x2][L@3F.2U>R@2B.3D>R@3B.2U;R@2B.3U>R@3B.2D>R@2F.3D]', ('3D2', " R'", '2B2', '3D2', '2B2', '3D2', ' R ', '3D2'))
                self._add_myperm2('CtrObl6p[3x2][L@2B.3D>L@3F.2D>L@2F.3U;L@2B.3U>L@3F.2U>R@3B.2U]', ('3U2', ' L ', '2B2', '3U2', '2B2', '3U2', " L'", '3U2'))
                self._add_myperm2('CtrObl3[L@3F.2U>R@2B.3D>R@3B.2U]', ('2B2', ' L ', '3F2', " L'", '2B2', ' L ', '3F2', " L'"))
                self._add_myperm2('CtrObl3[L@3F.2U>R@2F.3U>R@3B.2U]', ('2B2', " L'", '3B2', ' L ', '2B2', " L'", '3B2', ' L '))
                self._add_myperm2('CtrObl3[L@2B.3U>R@3B.2U>L@3F.2U]', (" L'", '3D2', ' L ', '2B2', " L'", '3D2', ' L ', '2B2'))
                self._add_myperm2('CtrObl6p[3x2][L@2B.3D>L@2F.3U>L@3F.2D;L@2B.3U>R@3B.2U>L@3F.2U]', ('3U2', ' L ', '3U2', '2B2', '3U2', '2B2', " L'", '3U2'))
                self._add_myperm2('CtrObl3[L@2B.3U>R@2B.3D>R@3B.2U]', (" L'", '3D2', ' L ', '2D2', " L'", '3D2', ' L ', '2D2'))


                self._add_myperm2('CtrObl3[F@3R.2U>R@3F.2D>R@3B.2U]~v01', ('2F ', "3R'", "2F'", ' R2', '2F ', '3R ', "2F'", ' R2'))
                self._add_myperm2('CtrObl3[F@3L.2D>F@3R.2U>R@3B.2U]~v01', ('3L ', "2B'", "3L'", ' F2', '3L ', '2B ', "3L'", ' F2'))
                self._add_myperm2('CtrObl3[F@3R.2U>R@2B.3D>R@2F.3U]', ('2D ', '3R2', "2D'", ' R2', '2D ', '3R2', "2D'", ' R2'))
                self._add_myperm2('CtrObl3[F@3L.2D>F@3R.2U>R@2F.3U]', ("2D'", '3B2', '2D ', ' F2', "2D'", '3B2', '2D ', ' F2'))
                self._add_myperm2('CtrObl3[F@3R.2U>R@3B.2U>R@2F.3U]', ('2U ', ' F ', "3D'", " F'", "2U'", ' F ', '3D ', " F'"))
                self._add_myperm2('CtrObl3[F@2R.3D>F@3R.2U>R@2F.3U]', ("2U'", ' R ', '3D ', " R'", '2U ', ' R ', "3D'", " R'"))
                self._add_myperm2('CtrObl3[F@2L.3U>F@3R.2U>R@2F.3U]', ("2U'", " R'", "3U'", ' R ', '2U ', " R'", '3U ', ' R '))
                self._add_myperm2('CtrObl3[F@3R.2U>R@3F.2D>R@2F.3U]', ('2U ', " F'", '3U ', ' F ', "2U'", " F'", "3U'", ' F '))
                self._add_myperm2('CtrObl3[F@3R.2U>R@2F.3U>R@3B.2U]', (' F ', "3D'", " F'", '2U ', ' F ', '3D ', " F'", "2U'"))
                self._add_myperm2('CtrObl3[F@2R.3D>F@3R.2U>R@3B.2U]', (" R'", "2U'", ' R ', '3D ', " R'", '2U ', ' R ', "3D'"))
                
                self._add_myperm2('CtrObl3[F@3L.2D>F@3R.2U>R@2B.3D]', (' F2', '2U ', '3F2', "2U'", ' F2', '2U ', '3F2', "2U'"))
                self._add_myperm2('CtrObl3[F@2L.3U>R@3B.2U>F@2R.3D]', (' F2', '3U ', '2B2', "3U'", ' F2', '3U ', '2B2', "3U'"))
                self._add_myperm2('CtrObl3[F@3L.2D>F@3R.2U>R@3F.2D]', (' F2', "3R'", '2F ', '3R ', ' F2', "3R'", "2F'", '3R '))
                self._add_myperm2('CtrObl3[F@2L.3U>R@2B.3D>R@2F.3U]', (' R2', '3B ', "2L'", "3B'", ' R2', '3B ', '2L ', "3B'"))

                self._add_myperm2('CtrObl3[F@3R.2U>R@3F.2D>R@3B.2U]~v02', (' R2', '2B ', '3R ', "2B'", ' R2', '2B ', "3R'", "2B'"))
                self._add_myperm2('CtrObl3[F@3L.2D>F@3R.2U>R@3B.2U]~v02', (' F2', '3R ', '2B ', "3R'", ' F2', '3R ', "2B'", "3R'"))

                

                self._add_myperm2('CtrObl3[F@3R.2U>R@3F.2D>U@2R.3F]', (" R'", "2D'", ' F2', '2D ', '3F ', "2D'", ' F2', '2D ', "3F'", ' R '))
                self._add_myperm2('CtrObl3[F@3L.2D>R@3B.2U>U@2L.3B]', (" R'", '2U ', ' F2', "2U'", "3B'", '2U ', ' F2', "2U'", '3B ', ' R '))
                self._add_myperm2('CtrObl3[F@2R.3D>R@2F.3U>U@3L.2F]', (' R ', '3D ', '2B ', ' U2', "2B'", "3D'", '2B ', ' U2', "2B'", " R'"))
                self._add_myperm2('CtrObl3[F@2L.3U>R@2B.3D>U@3R.2B]', (' R ', "3U'", "2F'", ' U2', '2F ', '3U ', "2F'", ' U2', '2F ', " R'"))

                self._add_myperm2('CtrObl3[F@3R.2U>R@3B.2U>U@2R.3F]', (' R ', "2D'", ' F2', '2D ', '3F ', "2D'", ' F2', '2D ', "3F'", " R'"))
                self._add_myperm2('CtrObl3[F@3R.2U>R@3B.2U>U@2L.3B]', (' F ', '2R ', ' U2', "2R'", "3D'", '2R ', ' U2', "2R'", '3D ', " F'"))
                self._add_myperm2('CtrObl3[F@2L.3U>R@2F.3U>U@3L.2F]', (" F'", '3L ', '2D ', ' R2', "2D'", "3L'", '2D ', ' R2', "2D'", ' F '))
                self._add_myperm2('CtrObl3[F@2L.3U>R@2F.3U>U@3R.2B]', (" R'", "3U'", "2F'", ' U2', '2F ', '3U ', "2F'", ' U2', '2F ', ' R '))
                
                self._add_myperm2('CtrObl3[F@3R.2U>U@2R.3F>R@2F.3U]', ('3B ', ' U2', "3B'", "2U'", '3B ', ' U2', "3B'", '2U '))
                self._add_myperm2('CtrObl3[F@2L.3U>U@2L.3B>R@3B.2U]', ("3U'", ' R2', '3U ', '2L ', "3U'", ' R2', '3U ', "2L'"))
                self._add_myperm2('CtrObl3[F@2L.3U>U@3R.2B>R@3B.2U]', ("2B'", "3D'", ' F2', '3D ', '2B ', "3D'", ' F2', '3D '))
                self._add_myperm2('CtrObl3[F@3R.2U>U@3L.2F>R@2F.3U]', ('2U ', '3R ', ' U2', "3R'", "2U'", '3R ', ' U2', "3R'"))

                self._add_myperm2('CtrObl3[F@3R.2U>R@2F.3U>U@2R.3F]', ("2U'", '3B ', ' U2', "3B'", '2U ', '3B ', ' U2', "3B'"))
                self._add_myperm2('CtrObl3[F@2L.3U>R@3B.2U>U@2L.3B]', ('2L ', "3U'", ' R2', '3U ', "2L'", "3U'", ' R2', '3U '))
                self._add_myperm2('CtrObl3[F@2L.3U>R@3B.2U>U@3R.2B]', ("3D'", ' F2', '3D ', "2B'", "3D'", ' F2', '3D ', '2B '))
                self._add_myperm2('CtrObl3[F@3R.2U>R@2F.3U>U@3L.2F]', ('3R ', ' U2', "3R'", '2U ', '3R ', ' U2', "3R'", "2U'"))

                self._add_myperm2('CtrObl3[F@3R.2U>R@3B.2U>U@3R.2B]', ('2B ', ' D ', '3R ', " D'", "2B'", ' D ', "3R'", " D'"))
                self._add_myperm2('CtrObl3[F@3R.2U>R@2F.3U>U@2L.3B]', ("2U'", ' L ', "3B'", " L'", '2U ', ' L ', '3B ', " L'"))
                self._add_myperm2('CtrObl3[F@3R.2U>U@3R.2B>R@2F.3U]', ('2U ', ' B ', "3R'", " B'", "2U'", ' B ', '3R ', " B'"))
                self._add_myperm2('CtrObl3[F@2L.3U>U@2L.3B>R@2F.3U]', ("2L'"," D ","3B "," D'","2L "," D ","3B'"," D'"))

                self._add_myperm2('CtrObl3[F@3R.2U>R@3B.2U>U@3L.2F]', (' U2', '2B ', ' D ', '3R ', " D'", "2B'", ' D ', "3R'", " D'", ' U2'))
                self._add_myperm2('CtrObl3[F@2L.3U>R@3B.2U>U@2R.3F]', (' F2', "2R'", ' B ', "3D'", " B'", '2R ', ' B ', '3D ', " B'", ' F2'))
                self._add_myperm2('CtrObl3[F@2L.3U>U@3L.2F>R@3B.2U]', (' R2', '2F ', ' L ', "3U'", " L'", "2F'", ' L ', '3U ', " L'", ' R2'))
                self._add_myperm2('CtrObl3[F@2L.3U>U@2R.3F>R@2F.3U]', (" U2","2L'"," D ","3B "," D'","2L "," D ","3B'"," D'"," U2"))

                self._add_myperm2('CtrObl3[F@3R.2U>L@3B.2D>R@2F.3U]', ('2U ', ' F ', '3D ', " F'", "2U'", ' F ', "3D'", " F'"))
                self._add_myperm2('CtrObl3[F@3R.2U>R@3B.2U>L@2F.3D]', ("2U'", ' F ', "3D'", " F'", '2U ', ' F ', '3D ', " F'"))
                self._add_myperm2('CtrObl3[F@3R.2U>L@2F.3D>R@3B.2U]', ('2U ', ' R ', '3F2', " R'", "2U'", ' R ', '3F2', " R'"))
                self._add_myperm2('CtrObl3[F@3R.2U>R@2F.3U>L@3B.2D]', ("2U'", ' L ', '3B2', " L'", '2U ', ' L ', '3B2', " L'"))

                self._add_myperm2('CtrObl3[F@2L.3U>R@2B.3D>L@2B.3U]', ('2D2', ' L ', '3U ', " L'", '2D2', ' L ', "3U'", " L'"))
                self._add_myperm2('CtrObl3[F@2L.3U>L@2B.3U>R@2B.3D]', ('2D2', ' R ', "3U'", " R'", '2D2', ' R ', '3U ', " R'"))
                self._add_myperm2('CtrObl3[F@3R.2U>L@2F.3D>R@2F.3U]', ('3F2', '2D ', '3R2', "2D'", ' R2', '2D ', '3R2', "2D'", ' R2', '3F2'))
                self._add_myperm2('CtrObl3[F@3R.2U>R@2F.3U>L@2F.3D]', ('3B2', "2D'", '3R2', '2D ', ' L2', "2D'", '3R2', '2D ', ' L2', '3B2'))
                self._add_myperm2('CtrObl3[F@3R.2U>R@2B.3D>L@2B.3U]', ('3B2', '2D ', '3R2', "2D'", ' R2', '2D ', '3R2', "2D'", ' R2', '3B2'))
                self._add_myperm2('CtrObl3[F@3R.2U>L@2B.3U>R@2B.3D]', ('3F2', "2D'", '3R2', '2D ', ' L2', "2D'", '3R2', '2D ', ' L2', '3F2'))

                self._add_myperm2('CtrObl3[F@2L.3U>R@2F.3U>L@2B.3U]', ('3B2', '2D ', " F'", "2D'", '3B2', '2D ', ' F ', "2D'"))
                self._add_myperm2('CtrObl3[B@2R.3U>R@2F.3U>L@2B.3U]', ('3B2', "2D'", " B'", '2D ', '3B2', "2D'", ' B ', '2D '))
                self._add_myperm2('CtrObl3[B@2R.3U>L@2B.3U>R@2F.3U]', ("2D'", " B'", '2D ', '3B2', "2D'", ' B ', '2D ', '3B2'))
                self._add_myperm2('CtrObl3[F@2L.3U>L@2B.3U>R@2F.3U]', ('2D ', " F'", "2D'", '3B2', '2D ', ' F ', "2D'", '3B2'))

                self._add_myperm2('CtrObl3[F@3R.2U>L@2B.3U>R@2F.3U]', ('2D ', ' F2', "2D'", '3B2', '2D ', ' F2', "2D'", '3B2'))
                self._add_myperm2('CtrObl3[F@3R.2U>R@2B.3D>L@2F.3D]', ("2D'", ' F2', '2D ', '3F2', "2D'", ' F2', '2D ', '3F2'))
                self._add_myperm2('CtrObl3[B@3L.2U>R@2F.3U>L@2B.3U]', ('3B2', "2D'", ' B2', '2D ', '3B2', "2D'", ' B2', '2D '))
                self._add_myperm2('CtrObl3[F@3R.2U>R@2F.3U>L@2B.3U]', ('3B2', '2D ', ' F2', "2D'", '3B2', '2D ', ' F2', "2D'"))

                self._add_myperm2('CtrObl3[F@3R.2U>L@3F.2U>R@2B.3D]', ('3R2', '2D ', ' B ', '3U ', " B'", "2D'", ' B ', "3U'", " B'", '3R2'))
                self._add_myperm2('CtrObl3[F@3R.2U>R@3F.2D>L@2B.3U]', ('3R2', "2D'", ' B ', "3U'", " B'", '2D ', ' B ', '3U ', " B'", '3R2'))
                self._add_myperm2('CtrObl3[F@3R.2U>L@2B.3U>R@3F.2D]', ('3R2', '2D ', ' R ', '3B2', " R'", "2D'", ' R ', '3B2', '3R2', " R'"))
                self._add_myperm2('CtrObl3[F@3R.2U>R@2B.3D>L@3F.2U]', ('3R2', "2D'", ' L ', '3F2', " L'", '2D ', ' L ', '3F2', '3R2', " L'"))
                self._add_myperm2('CtrObl3[F@2L.3U>R@2F.3U>L@2F.3D]', ('2L2', '2U2', ' L ', '3D ', " L'", '2U2', ' L ', "3D'", '2L2', " L'"))
                self._add_myperm2('CtrObl3[F@2L.3U>L@2F.3D>R@2F.3U]', ('2L2', '2U2', ' R ', "3D'", " R'", '2U2', ' R ', '3D ', '2L2', " R'"))

                self._add_myperm2('CtrObl3[F@2L.3U>R@2F.3U>L@3F.2U]', ('2L2', ' L ', '2U2', ' L ', '3D ', " L'", '2U2', ' L ', "3D'", '2L2', ' L2'))
                self._add_myperm2('CtrObl3[B@2R.3U>R@2F.3U>L@3F.2U]', ('2R2', ' L ', '2U2', ' L ', "3D'", " L'", '2U2', ' L ', '3D ', '2R2', ' L2'))
                self._add_myperm2('CtrObl3[B@2R.3U>L@3F.2U>R@2F.3U]', (' L2', '2R2', "3D'", " L'", '2U2', ' L ', '3D ', " L'", '2U2', " L'", '2R2'))
                self._add_myperm2('CtrObl3[F@2L.3U>L@3F.2U>R@2F.3U]', (' L2', '2L2', '3D ', " L'", '2U2', ' L ', "3D'", " L'", '2U2', " L'", '2L2'))



            self._add_myperm2('OuterCenterBar-A', (' R2', "2D'", ' R2', ' B2', ' L2', "2U'", ' L2', ' B2'))
            self._add_myperm2('OuterCenterBar-B', (' B2', ' L2', "2D'", ' L2', ' B2', ' R2', "2U'", ' R2'))
            
            self._add_myperm2('OuterCenterBar-C', (' R ', "2D'", ' R2', ' B2', ' L2', "2U'", ' L2', ' B2', ' R '))
            self._add_myperm2('OuterCenterBar-D', (' F ', '2D ', ' F2', ' L2', ' B2', '2U ', ' B2', ' L2', ' F '))

            self._add_myperm2('OuterCenterBar-E', (' F ', " R'", '2U ', ' R2', ' B2', ' L2', '2D ', ' L2', ' B2', " R'", " F'"))
            self._add_myperm2('OuterCenterBar-F', (' F ', ' R ', '2U ', ' R2', ' B2', ' L2', '2D ', ' L2', ' B2', ' R ', " F'"))
            self._add_myperm2('OuterCenterBar-G', (" F'", " R'", '2U ', ' R2', ' B2', ' L2', '2D ', ' L2', ' B2', " R'", ' F '))
            self._add_myperm2('OuterCenterBar-H', (' R ', " F'", "2U'", ' F2', ' L2', ' B2', "2D'", ' B2', ' L2', " F'", " R'"))
    
            self._add_myperm2('OuterCenterBar-W', ('2U ', ' R2', ' B2', ' L2', '2D ', '2U ', ' L2', ' B2', ' R2', '2D '))
            self._add_myperm2('OuterCenterBar-WW', (" F'", "2U'", ' F2', ' L2', ' B2', "2D'", "2U'", ' B2', ' L2', ' F2', "2D'", ' F '))

            self._add_myperm2('OuterCenterBar-KA', ("2D'", ' L2', ' B2', ' R2', '2U2', ' R2', ' B2', ' L2', "2D'"))
            self._add_myperm2('OuterCenterBar-KB', ('2D ', ' L2', ' B2', ' R2', '2U2', ' R2', ' B2', ' L2', '2D '))

            self._add_myperm2('OuterCenterBar-JA', ('2U ', ' F2', ' R2', ' B2', "2D'", ' B2', ' R2', ' F2', '2U2'))
            self._add_myperm2('OuterCenterBar-JB', ('2U2', ' F2', ' R2', ' B2', '2D ', ' B2', ' R2', ' F2', "2U'"))

            self._add_myperm2('OuterCenterBar-IA', (' L2', '2U ', ' F2', ' R2', ' B2', "2D'", ' B2', ' R2', ' F2', '2U2', ' L2'))
            self._add_myperm2('OuterCenterBar-IB', (' L2', '2U2', ' F2', ' R2', ' B2', '2D ', ' B2', ' R2', ' F2', "2U'", ' L2'))
            
            self._add_myperm2('OuterCenterBar-X', ('2D2', ' F2', ' R2', ' F2', '2D2', ' F2', ' R2', ' F2'))
            self._add_myperm2('OuterCenterBar-Y', (' U2', ' R2', ' U2', '2B2', ' U2', ' R2', ' U2', '2B2'))
            self._add_myperm2('OuterCenterBar-Z', (" R'", '2B2', ' D2', ' R2', ' D2', '2B2', ' D2', ' R2', ' D2', ' R '))
            self._add_myperm2('OuterCenterBar-XX', ('2U2', ' F2', ' B2', '2D2', ' F2', ' B2'))
            self._add_myperm2('OuterCenterBar-ZZ', (' R ', '2F2', ' U2', ' D2', '2B2', ' U2', ' D2', " R'"))

            

    def _register_myperms2_f2l_oll(self):
        """F2L/OLLやCenters条件に応じた手順群を登録する。"""
        # 命名メモ:
        # - OuterCenterBar / MidCenterBar は center の bar を動かす family。
        # - Adjacent3Center / Line3Center は 3面の center を動かす family。
        # - InOut / InIn / OutOut / Middle-* は各 center の相対位置関係を表す。
        if self.F2L or self.OLL:
            self.myperms2 = {}
            self.myperms2['Q1-'] = (" S "," E "," S'"," E'")
            self.myperms2['Q2-'] = (" S "," E2"," S'"," E2")
            self.myperms2['Q3-'] = (' S ', " U'", ' B ', " F'", ' L ', ' F ', " B'", ' U ', " F "," B'", ' R2', " F'"," B ")

            self.myperms2['CornerSwap00-'] = (" L'", ' D ', " R'", " D'", ' L ', ' D ', ' R ', " D'", " L'", " D'", ' L ', ' U2', " L'", ' D ', ' L ')
            self.myperms2['CornerSwap01-'] = (" L'", ' D2', " L'", ' U2', ' L ', ' D2', " L'", ' U2', ' L2', ' F ', " D'", " F'", ' U ', ' F ', ' D ', " F'")
            self.myperms2['CornerSwap02-'] = (' L ', " B'", " L'", " F'", ' L ', ' B ', " L'", ' F ', ' B2', ' L ', ' F ', " L'", ' B2', ' L ', " F'", " L'")
            self.myperms2['CornerSwap03-'] = (" B'", ' D ', ' B ', ' U2', " B'", " D'", ' B ')
            self.myperms2['CornerSwap04-'] = (' F2', " U'", ' F2', " U'", ' F2', ' U2', ' F2')
            self.myperms2['CornerSwap05-'] = (" U ",' F2', " U'", ' F2', " U'", ' F2', ' U2', ' F2')
            self.myperms2['CornerSwap06-'] = (" U2",' F2', " U'", ' F2', " U'", ' F2', ' U2', ' F2')
            self.myperms2['CornerSwap07-'] = (" U'",' F2', " U'", ' F2', " U'", ' F2', ' U2', ' F2')
            self.myperms2['CornerSwap08-'] = (" F'", ' D ', ' F ', ' U ', " F'", " D'", ' F ')
            self.myperms2['CornerSwap09-'] = (" U "," F'", ' D ', ' F ', ' U ', " F'", " D'", ' F ')
            self.myperms2['CornerSwap10-'] = (" F'", ' D ', ' F ', " U'", " F'", " D'", ' F ')
            self.myperms2['CornerSwap11-'] = (' B ', ' D ', " B'", ' U ', ' B ', " D'", " B'")
            self.myperms2['CornerSwap12-'] = (' F2', " L'", ' F2', ' U2', ' D2', ' B2', " R'", ' B2', ' D2')


            
            self.myperms2['F2L-A0'] = (" R "," U2"," R'"," U "," R "," U2"," R'"," U "," F'"," U'"," F ")
            self.myperms2['F2L-A1'] = (" U "," R "," U'"," R'"," F "," R'"," F'"," R ")
            self.myperms2['F2L-B1'] = (" R "," U'"," R'"," U "," F'"," U "," F ")
            self.myperms2['F2L-B2'] = (" U "," R "," U'"," R'") * 3
            self.myperms2['F2L-C'] = (" R "," U'"," R'"," U "," R "," U2"," R'"," U "," R "," U'"," R'")
            self.myperms2['F2L-D'] = (" R "," U'"," R'"," U'"," R "," U'"," R'"," U "," F'"," U'"," F ")
            self.myperms2['F2L-E'] = (" R "," U'"," R'"," U "," R "," U'"," R'")
            self.myperms2['F2L-F'] = (" R "," U "," R'"," U'"," R "," U "," R'")
            self.myperms2['F2L-G'] = (" U'"," R "," U'"," R'"," U2"," R "," U'"," R'")
            self.myperms2['F2L-H'] = (" U "," F'"," U'"," F "," U'"," R "," U "," R'")
            self.myperms2['F2L-I'] = (" R "," U'"," R'")
            self.myperms2['F2L-J'] = (" U'"," R "," U2"," R'"," U "," F'"," U'"," F ")
            self.myperms2['F2L-K'] = (" R "," U'"," R'"," U2"," F'"," U'"," F ")
            self.myperms2['F2L-L'] = (" U'"," R "," U'"," R'"," U "," R "," U "," R'")
            self.myperms2['F2L-M'] = (" U "," F "," R'"," F'"," R "," U "," R "," U "," R'")
            self.myperms2['F2L-N'] = (" R "," U2"," R'"," U'"," R "," U "," R'")

            self.myperms2['F2L-Q'] = (" R "," U "," R'"," U2"," R "," U'"," R'")
            self.myperms2['F2L-R'] = (" R "," U "," R'"," U "," R "," U "," R'")
            self.myperms2['F2L-S'] = (" R "," U2"," R'"," U2"," R "," U'"," R'")  
            self.myperms2['F2L-T'] = (" R "," U "," R'")
            self.myperms2['F2L-U'] = (" R "," U2"," R'"," U "," R "," U'"," R'")
            self.myperms2['F2L-V'] = (" R "," U "," R'"," U "," R "," U'"," R'")

            self.myperms2['OLL-Sune'] = (" R "," U2"," R'"," U'"," R "," U'"," R'")
            self.myperms2['OLL-8'] = (" B'"," R'"," F'"," R "," B "," R'"," F "," R ")
            self.myperms2['OLL-A1'] = (" R'"," F'"," R "," B'"," R'"," F "," R "," B ")
            self.myperms2['OLL-A2'] = (" R2"," D'"," R "," U2"," R'"," D "," R "," U2"," R ")
            self.myperms2['OLL-CrossH'] = (" F ",) + (" R "," U "," R'"," U'") * 3 + (" F'",)
            self.myperms2['OLL-CrossPi'] = (" R "," U2"," R2"," U'"," R2"," U'"," R2"," U2"," R ")

            self.myperms2['OLL-DotH'] = (" R "," U2"," R2"," F "," R "," F'"," U2"," R'"," F "," R "," F'")
            self.myperms2['OLL-DotT'] = (" F "," R "," U "," R'"," U'"," F'"," z "," B "," R "," U "," R'"," U'"," B'"," z'")
            self.myperms2['OLL-DotQ'] = (" z "," B "," R "," U "," R'"," U'"," B'"," z'"," U "," F "," R "," U "," R'"," U'"," F'")

            self.myperms2['OLL-Square'] = (" x "," L "," U2"," R'"," U'"," R "," U'"," L'"," x'")
            self.myperms2['OLL-SL'] = (" x "," L "," U "," R'"," U "," R "," U2"," L'"," x'")
            self.myperms2['OLL-SC'] = (" x "," L "," U "," R'"," U "," R'"," F "," R "," F'"," R "," U2"," L'"," x'")
            self.myperms2['OLL-Y'] = (" R "," U "," R'"," U'"," R'"," F "," R2"," U "," R'"," U'"," F'")
            self.myperms2['OLL-LargeLI'] = (" F "," U "," R "," U'"," R2"," F'"," R "," U "," R "," U'"," R'")
            self.myperms2['OLL-LargeLJ'] = (" x "," L "," U "," L'"," x'"," R "," U "," R'"," U'"," x "," L "," U'"," L'"," x'")
            self.myperms2['OLL-LC'] = (" F ",) + (" R "," U "," R'"," U'") * 2 + (" F'",)
            self.myperms2['OLL-LJ'] = (" x "," L "," U "," L'"," x'") + (" R "," U "," R'"," U'") * 2 + (" x "," L "," U'"," L'"," x'")
            self.myperms2['OLL-LL'] = (" x "," L "," U'"," x2"," L2"," U "," x2"," L2"," U "," x2"," L2"," U'"," x "," L ")

            self.myperms2['OLL-IC'] = (" F ",) + (" U "," R "," U'"," R'") * 2 + (" F'",)
            self.myperms2['OLL-IO'] = (" x "," L "," U "," L'"," x'") + (" U "," R "," U'"," R'") * 2 + (" x "," L "," U'"," L'"," x'")
            self.myperms2['OLL-ID'] = (" R'"," U'") + (" F "," R'"," F'"," R ") * 2 + (" U "," R ")
            self.myperms2['OLL-III'] = (" R "," U2"," R2"," U'"," R "," U'"," R'"," U2"," F "," R "," F'")

            self.myperms2['OLL-Diagonal'] = (" R "," U "," R'"," U "," R'"," F "," R ", " F'"," U2"," R'"," F "," R "," F'")
            self.myperms2['OLL-VU'] = (" F "," R'"," F'"," R "," U2"," F "," R'"," F'"," R "," U'"," R "," U'"," R'")
            self.myperms2['OLL-VV'] = (" x'"," L'"," R "," U "," R "," U "," R'"," U'"," x "," L "," R2"," F "," R "," F'")

            self.myperms2['OLL-SXI'] = (" R "," U "," R'"," U'"," R "," U'"," R'"," F'"," U'"," F "," R "," U "," R'")
            self.myperms2['OLL-SXJ'] = (" R "," U "," R'"," U "," R "," U2"," R'"," F "," R "," U "," R'"," U'"," F'")

            self.myperms2['OLL-PL'] = (" F "," U "," R "," U'"," R'"," F'")
            self.myperms2['OLL-PJ'] = (" R'"," U'"," F "," U "," R "," U'"," R'"," F'"," R ")

            self.myperms2['OLL-LargeS'] = (" R'"," F "," R "," U "," R'"," U'"," F'"," U "," R ")

            self.myperms2['OLL-TH'] = (" R "," U "," R'"," U'"," R'"," F "," R "," F'")
            self.myperms2['OLL-TU'] = (" F "," R "," U "," R'"," U'"," F'")

            self.myperms2['OLL-CT'] = (" R'"," U'"," R'"," F "," R "," F'"," U "," R ")
            self.myperms2['OLL-CU'] = (" R "," U "," R2"," U'"," R'"," F "," R "," U "," R "," U'"," F'")

            self.myperms2['OLL-SquareXU'] = (" R "," U2"," R2"," F "," R "," F'"," R "," U2"," R'")
            self.myperms2['OLL-SquareXV'] = (" F "," R "," U'"," R'"," U'"," R "," U "," R'"," F'")

            self.myperms2['OLL-W'] = (" R "," U "," R'"," U "," R "," U'"," R'"," U'"," R'"," F "," R "," F'")

            self.myperms2['OLL-X'] = (" R'", ' F2', ' R ', " L'", ' U2', ' L2', " R'", " F'", ' R ', " L'", ' U2', " R'", ' L ', ' F2', ' R ', " L'")
            self.myperms2['OLL-R'] = (" B'", " R'", ' F ', ' R ', ' B ', " F'", " U'", " F'", ' U ', ' F ')
            self.myperms2['OLL-H'] = (" F'", " U'", ' F ', ' U ', ' F ', " B'", " R'", " F'", ' R ', ' B ')


        if self.Centers:
            self.myperms2 = {k:self.myperms2[k] for k in self.myperms2 if k[:4] not in ['Edge','Swap'] and k[:7] not in ['MidEdge'] and k[:2] not in ['CP']}

            self._add_myperm2('CtrX8p[4x2]+W2-2s[FL@D<>UB@R]', ("2R2", ' B ', "2D'", ' B ', '2R ', ' B2', ' U2', ' B ', '2D ', " B'", ' U2', '2R2'))
            self._add_myperm2('CtrX8p[4x2]+W2-2s[FL@U<>UB@L]', ('2L2', ' B ', '2U ', ' B ', "2L'", ' B2', ' U2', ' B ', "2U'", " B'", ' U2', '2L2'))
            self._add_myperm2('CtrX8p[4x2]+W2-2s[FL@D<>UF@R]', ("2R'", ' B ', "2D'", ' B ', '2R ', ' B2', ' U2', ' B ', '2D ', " B'", ' U2', '2R '))
            self._add_myperm2('CtrX8p[4x2]+W2-2s[FL@U<>UF@L]', ('2L ', ' B ', '2U ', ' B ', "2L'", ' B2', ' U2', ' B ', "2U'", " B'", ' U2', "2L'")           )
            self._add_myperm2('CtrX8p[4x2]+W2-2s[DF@R<>FL@D]', (' B ', "2D'", ' B ', '2R ', ' B2', ' U2', ' B ', '2D ', " B'", ' U2'))
            self._add_myperm2('CtrX8p[4x2]+W2-2s[DF@L<>FL@U]', (' B ', '2U ', ' B ', "2L'", ' B2', ' U2', ' B ', "2U'", " B'", ' U2')     )
            self._add_myperm2('CtrX8p[4x2]+W2-2s[DB@R<>FL@D]', ('2R ', ' B ', "2D'", ' B ', '2R ', ' B2', ' U2', ' B ', '2D ', " B'", ' U2', "2R'"))
            self._add_myperm2('CtrX8p[4x2]+W2-2s[DB@L<>FL@U]', ("2L'", ' B ', '2U ', ' B ', "2L'", ' B2', ' U2', ' B ', "2U'", " B'", ' U2', '2L '))

            self.myperms2['Swap_A'] = ('2R ', ' D2', "2R'", ' D2', '2L ', ' D2', ' B2', '2L ', ' B2', "2L'", ' D2')
            self.myperms2['Swap_B'] = ("2R'", ' U ', "2B'", " U'", ' B2', ' U ', '2B ', ' U ', "2R'", ' U2', ' B2', '2R ')
            self.myperms2['Swap_I'] = ('2L ', ' F2', '2L ', '2R ', ' F2', "2R'", ' U2', '2R2', ' B2', '2R ', ' B2', '2R2', ' U2')
            self.myperms2['Swap_J'] = (" F'", "2D'", ' F ', ' U2', " F'", '2D ', " F'", "2R'", ' F2', ' U2')
            self.myperms2['Swap_K'] = ('2R2', " F'", "2D'", ' F ', ' U2', " F'", '2D ', " F'", "2R'", ' F2', ' U2', '2R2')

            
            

            

       

    def _expand_registered_myperms(self, names = None):
        """登録済みmyperms2を対称変換展開してmypermsへ写す。"""
        self.myperms2 = normalize_myperm_registry(self.myperms2)
        keys = tuple(self.myperms2.keys()) if names is None else tuple(names)
        for key in keys:
            if key not in self.myperms2:
                continue
            L = self.make_transformations(self.myperms2[key],tuple())
            if self.size < 6:
                Num = 48
            elif len([x for x in self.myperms2[key] if x[0] in ['2','3']]) != 0:
                Num = 96
            else:
                Num = 48
            for i in range(Num):
                self.myperms[make_myperm_key(key, i)] = L[0][i]

    def _reindex_myperms_by_points(self, names = None):
        """point最大の対称変換を各myperm系列の#00へ割り当てる。"""
        points_path = Path(__file__).resolve().parent.parent / "Points.txt"
        if not points_path.exists():
            self.myperm_transform_key_aliases = {}
            self.myperm_transform_points = {}
            return
        point_table = load_myperm_points(points_path)
        reindex_myperms_by_points(self, point_table, names = names)

    def apply_point_reindex(self, point_table = None):
        """Apply point-based transform reindexing to the current myperm registry."""
        if point_table is None:
            points_path = Path(__file__).resolve().parent.parent / "Points.txt"
            point_table = load_myperm_points(points_path)
        reindex_myperms_by_points(self, point_table)
        rename_myperms_by_effect(self)
        self._init_myperms_index()
        self._init_single_move_and_rotate()

    
    def _init_single_move_and_rotate(self):
        self.single_and_rotate = [
            key for key in self.myperms.keys()
            if myperm_base_key(key).startswith('SingleMove') or myperm_base_key(key).startswith('Rotate')
        ]
                
    def collect_single_move_and_rotate(self):
        return self.single_and_rotate

    def _init_cube_state_and_moves(self):
        """盤面初期化・move定義・piece番号表をまとめて構築する。"""
        face_keys = ['U','D','F','B','L','R']
        self._init_surface_size()
        self._init_state_colors()
        self._apply_state_masks()
        self.state_0 = self.state.copy()
        self._init_face_nums()
        face_turn_map = self._build_face_turn_map()
        self._init_move_tables(face_keys, face_turn_map)
        self._init_scramble_sets()
        side_strips = self._build_side_strips()
        self._apply_side_strips(side_strips)
        self._apply_axis_rotations(side_strips)
        self._finalize_axis_rotations()
        self._init_piece_metadata()

    def _init_surface_size(self):
        self.surface_num = self.size ** 2

    def _init_state_colors(self):
        self.state = np.zeros(self.surface_num * 6,dtype = str)
        self.state[0:self.surface_num] = 'R'
        self.state[self.surface_num:2 * self.surface_num] = 'O'
        self.state[2 * self.surface_num:3 * self.surface_num] = 'Y'
        self.state[3 * self.surface_num:4 * self.surface_num] = 'W'
        self.state[4 * self.surface_num:5 * self.surface_num] = 'G'
        self.state[5 * self.surface_num:6 * self.surface_num] = 'B'

    def _apply_state_masks(self):
        if self.F2L:
            self._mask_f2l_state()
        if self.OLL:
            self._mask_oll_state()
        if self.Cross:
            self._mask_cross_state()
        if self.Centers:
            self._mask_centers_state()
        if self.Edges:
            self._mask_edges_state()

    def _mask_f2l_state(self):
        self.state[0:9] = 'X'
        for i in range(2,6):
            self.state[i * 9 + 0] = 'X'
            self.state[i * 9 + 3] = 'X'
            self.state[i * 9 + 4] = 'X'

    def _mask_oll_state(self):
        self.state[0:9] = 'R'
        for i in range(2,6):
            self.state[i * 9 + 0] = 'X'
            self.state[i * 9 + 3] = 'X'
            self.state[i * 9 + 4] = 'X'

    def _mask_cross_state(self):
        self.state[0:8] = 'X'
        for i in range(2,6):
            self.state[i * 9 + 0] = 'X'
            self.state[i * 9 + 1] = 'X'
            self.state[i * 9 + 2] = 'X'
            self.state[i * 9 + 3] = 'X'
            self.state[i * 9 + 4] = 'X'
            self.state[i * 9 + 5] = 'X'
            self.state[i * 9 + 7] = 'X'
        self.state[9:13] = 'X'

    def _mask_centers_state(self):
        for i in range(6):
            start = i * self.surface_num + 4 * (self.size - 1)
            end = (i + 1) * self.surface_num
            self.state[start:end] = 'X'

    def _mask_edges_state(self):
        for i in range(6):
            end = i * self.surface_num + 4 * (self.size - 1)
            self.state[i * self.surface_num:end] = 'X'

    def _init_face_nums(self):
        self.Nums = {}
        self.Nums['R'] = R_Nums[self.size]
        self.Nums['O'] = self.Nums['R'][::-1,::-1] + self.surface_num
        self.Nums['Y'] = self.Nums['R'][::-1,::-1] + self.surface_num * 2
        self.Nums['W'] = self.Nums['R'] + self.surface_num * 3
        self.Nums['G'] = np.flip(self.Nums['R'].T,axis = 0) + self.surface_num * 4
        self.Nums['B'] = np.flip(self.Nums['R'].T,axis = 1) + self.surface_num * 5

    def _build_face_turn_map(self):
        face_turn_map = np.zeros(0,dtype = 'i')
        quarter_turn = np.array([3,0,1,2],dtype = 'i')
        for i in range(self.surface_num // 4):
            face_turn_map = np.r_[face_turn_map,quarter_turn + 4 * i]
        if self.size % 2 == 1:
            face_turn_map = np.r_[face_turn_map,np.array([self.surface_num - 1])]
        return face_turn_map

    def _init_move_tables(self, face_keys, face_turn_map):
        all_indices = np.arange(self.surface_num * 6,dtype = 'i')
        for j in range(6):
            for i in range(self.size // 2):
                key = self._layer_key(face_keys[j], i)
                self.move[key] = all_indices.copy()
            face_key = " " + face_keys[j] + " "
            self.move[face_key][self.surface_num * j:self.surface_num * (j+1)] = face_turn_map + self.surface_num * j

        if self.size % 2 == 1:
            self.move[" M "] = all_indices.copy()
            self.move[" S "] = all_indices.copy()
            self.move[" E "] = all_indices.copy()

        self.move[" x "] = all_indices.copy()
        self.move[" y "] = all_indices.copy()
        self.move[" z "] = all_indices.copy()

    def _layer_key(self, face_key, layer_index):
        if layer_index != 0:
            return str(layer_index + 1) + face_key + " "
        return " " + face_key + " "

    def _init_move_keys(self):
        face_keys = ["U","D","F","B","L","R"]
        self.move_keys = [" " + s + t for s in face_keys for t in [" ","'","2"]]
        self.move_keys += [str(i + 1) + s + t for i in range(1,self.size // 2) for s in face_keys for t in [" ","'","2"]]
        if self.size % 2 == 1:
            self.move_keys += [" E "," E'"," E2"," S "," S'"," S2"," M "," M'"," M2"]
        self.move_keys += [" y "," y'"," y2"," z "," z'"," z2"," x "," x'"," x2"]
        self.move_len = len(self.move_keys)
        self.key_to_num = {}
        for i in range(self.move_len):
            self.key_to_num[self.move_keys[i]] = i

    def _init_scramble_sets(self):
        self.my_scrambles2 = {0:{}}
        self.my_scramble_changed_piece_keys = {0:{}}
        for key in self.move_keys:
            self.my_scrambles2[0][key] = set([])
        self.my_scramble_changed_piece_keys[0] = {}
        self.counter = {1:{},2:{},3:{},4:{},5:{},6:{},7:{}}

    def _build_side_strips(self):
        side_strips = {}
        for i in range(self.size // 2):
            self._add_layer_side_strips(side_strips, i)
        if self.size % 2 == 1:
            self._add_slice_side_strips(side_strips)
        return side_strips

    def _add_layer_side_strips(self, side_strips, layer_index):
        key_prefix = " " if layer_index == 0 else str(layer_index + 1)
        i = layer_index
        side_strips[key_prefix + 'U' + " "] = [self.Nums['Y'][i,:],self.Nums['G'][:,-1-i],self.Nums['W'][-1-i,::-1],self.Nums['B'][::-1,i]]
        side_strips[key_prefix + 'D' + " "] = [self.Nums['Y'][-1-i,:],self.Nums['B'][::-1,-1-i],self.Nums['W'][i,::-1],self.Nums['G'][:,i]]
        side_strips[key_prefix + 'F' + " "] = [self.Nums['R'][-1-i,:],self.Nums['B'][-1-i,:],self.Nums['O'][-1-i,:],self.Nums['G'][-1-i,:]]
        side_strips[key_prefix + 'B' + " "] = [self.Nums['R'][i,:],self.Nums['G'][i,:],self.Nums['O'][i,:],self.Nums['B'][i,:]]
        side_strips[key_prefix + 'L' + " "] = [self.Nums['R'][:,i],self.Nums['Y'][:,i],self.Nums['O'][::-1,-1-i],self.Nums['W'][:,i]]
        side_strips[key_prefix + 'R' + " "] = [self.Nums['R'][:,-1-i],self.Nums['W'][:,-1-i],self.Nums['O'][::-1,i],self.Nums['Y'][:,-1-i]]

    def _add_slice_side_strips(self, side_strips):
        side_strips[" M "] = [self.Nums['R'][:,self.size // 2],self.Nums['Y'][:,self.size // 2],self.Nums['O'][::-1,self.size // 2],self.Nums['W'][:,self.size // 2]]
        side_strips[" S "] = [self.Nums['R'][self.size // 2,:],self.Nums['B'][self.size // 2,:],self.Nums['O'][self.size // 2,:],self.Nums['G'][self.size // 2,:]]
        side_strips[" E "] = [self.Nums['Y'][self.size // 2,:],self.Nums['B'][::-1,self.size // 2],self.Nums['W'][self.size // 2,::-1],self.Nums['G'][:,self.size // 2]]

    def _apply_side_strips(self, side_strips):
        for key in side_strips.keys():
            for i in range(4):
                for j in range(self.size):
                    self.move[key][side_strips[key][i][j]] = side_strips[key][i-1][j]
            self.move[key[:2] + "'"] = np.argsort(self.move[key])
            self.move[key[:2] + "2"] = self.move[key][self.move[key]]

    def _apply_axis_rotations(self, side_strips):
        for key in side_strips.keys():
            axis_key = " " + self.axis[key[1]] + " "
            if key[1] in ["R","U","F","S"]:
                self.move[axis_key] = self.move[axis_key][self.move[key]]
            else:
                self.move[axis_key] = self.move[axis_key][self.move[self.invert_str(key)]]

    def _finalize_axis_rotations(self):
        for key in [" x "," y "," z "]:
            self.move[key[:2] + "'"] = np.argsort(self.move[key])
            self.move[key[:2] + "2"] = self.move[key][self.move[key]]

    def _init_piece_metadata(self):
        """pieceの index 表・番号逆引き・完成色をまとめて初期化する。"""
        self._init_piece_indices()
        self._init_piece_lookup_tables()
        self._init_default_colors()

    def _init_piece_indices(self):
        """center / edge / corner の index 集合を作る。"""
        self.center_num = (self.size - 2) ** 2
        self.edge_pairs = self._build_edge_pairs()
        self.AB = AB[self.size]
        self.CL = self._build_corner_locations()
        self.center_index = self._build_center_indices()
        self.edge_index = self._build_edge_indices()
        self.corner_index = self._build_corner_indices()

    def _build_edge_pairs(self):
        """edge piece を構成する2面の基準位置を返す。"""
        return [((0,0),(2,0)),
                ((0,1),(4,0)),
                ((0,2),(3,0)),
                ((0,3),(5,0)),
                ((2,3),(4,1)),
                ((4,3),(3,1)),
                ((3,3),(5,1)),
                ((5,3),(2,1)),
                ((1,0),(3,2)),
                ((1,1),(4,2)),
                ((1,2),(2,2)),
                ((1,3),(5,2))]

    def _build_corner_locations(self):
        """corner piece を構成する3面の基準位置を返す。"""
        return [((0,0),(2,3),(4,0)),
                ((0,1),(4,3),(3,0)),
                ((0,2),(3,3),(5,0)),
                ((0,3),(5,3),(2,0)),
                ((1,0),(3,1),(4,2)),
                ((1,1),(4,1),(2,2)),
                ((1,2),(2,1),(5,2)),
                ((1,3),(5,1),(3,2))]

    def _build_center_indices(self):
        """center piece の index 一覧を返す。"""
        return [(i + self.surface_num * j,) for j in range(6) for i in range(4 * (self.size - 1),self.surface_num)]

    def _build_edge_indices(self):
        """edge piece の index 一覧を返す。"""
        return [(p[0][0] * self.surface_num + p[0][1] + 4 * ab[0],p[1][0] * self.surface_num + p[1][1] + 4 * ab[1]) for ab in self.AB for p in self.edge_pairs]

    def _build_corner_indices(self):
        """corner piece の index 一覧を返す。"""
        return [(cl[0][0] * self.surface_num + cl[0][1],cl[1][0] * self.surface_num + cl[1][1],cl[2][0] * self.surface_num + cl[2][1]) for cl in self.CL]

    def _init_piece_lookup_tables(self):
        """盤面 index から piece へ戻る逆引き表を作る。"""
        self.num_to_piece = {}
        for i in range(6 * self.surface_num):
            if i % self.surface_num < 4:
                self.num_to_piece[i] = [x for x in self.corner_index if i in x][0]
            elif i % self.surface_num < 4 * (self.size - 1):
                self.num_to_piece[i] = [x for x in self.edge_index if i in x][0]
            else:
                self.num_to_piece[i] = (i,)

    def _init_default_colors(self):
        """完成状態での各 piece の色並びを保存する。"""
        self.default_color = {}
        for x in self.center_index:
            self.default_color[x] = self.state_0[x[0]]
        for x in self.edge_index:
            self.default_color[x] = self.state_0[x[0]] + self.state_0[x[1]]
        for x in self.corner_index:
            self.default_color[x] = self.state_0[x[0]] + self.state_0[x[1]] + self.state_0[x[2]]

    def _init_color_keys_and_groups(self):
        """配色ID・入力次元・評価用グループベクトルを初期化する。"""
        self._init_piece_color_keys()
        self._apply_partial_solve_color_keys()
        self._init_piece_color_lists()
        self._init_input_vector_metadata()
        self._init_group_values()

    def _init_piece_color_keys(self):
        """edge / corner の色並びを整数IDへ変換する表を作る。"""
        # エッジ/コーナー配色の識別ID（色並び→番号）
        self.edge_key = {'RB': 0,'BR': 1,'RY': 2,'YR': 3,
                         'RG': 4,'GR': 5,'RW': 6,'WR': 7,
                         'BY': 8,'YB': 9,'YG':10,'GY':11,
                         'GW':12,'WG':13,'WB':14,'BW':15,
                         'OG':16,'GO':17,'OW':18,'WO':19,
                         'OB':20,'BO':21,'OY':22,'YO':23,
                         }

        self.corner_key = {'RBY': 0,'BYR': 1,'YRB': 2,
                           'RYG': 3,'YGR': 4,'GRY': 5,
                           'RGW': 6,'GWR': 7,'WRG': 8,
                           'RWB': 9,'WBR':10,'BRW':11,
                           'OGY':12,'GYO':13,'YOG':14,
                           'OYB':15,'YBO':16,'BOY':17,
                           'OBW':18,'BWO':19,'WOB':20,
                           'OWG':21,'WGO':22,'GOW':23,
                           }

    def _apply_partial_solve_color_keys(self):
        """F2L / OLL / Edges / Cross 条件に応じて配色IDを上書きする。"""
        if self.F2L or self.Edges or self.Cross:
            self.edge_key['XX'] = 0
            self.corner_key['XXX'] = 0

        if self.OLL:
            self.edge_key['RX'] = 0
            self.edge_key['XR'] = 1
            self.corner_key['RXX'] = 0
            self.corner_key['XRX'] = 1
            self.corner_key['XXR'] = 2

    def _init_piece_color_lists(self):
        """ID順に並べた色並びリストを作る。"""
        # ID順に色並びを並べたリスト
        self.edge_colors = sorted(self.edge_key.keys(),key = lambda x :self.edge_key[x])
        self.corner_colors = sorted(self.corner_key.keys(),key = lambda x :self.corner_key[x])

    def _init_input_vector_metadata(self):
        """入力次元と完成状態特徴量を計算する。"""
        # 入力ベクトルの総次元（盤面情報の固定長表現）
        self.ips = 36*self.surface_num + 144 * self.size - 240
        
        # 完全解状態の特徴量（教師データ基準）
        self.perfect_data = self.makedata()

    def _init_group_values(self):
        """評価用グループごとのマスクベクトルと総和を作る。"""
        base_vector = self._empty_group_vector()
        self.group_val = {}
        self.total_val = {}
        group_names = self._group_name_map()

        if self.size % 2 == 1:
            self._init_group_values_for_odd_size(group_names, base_vector)
        else:
            self._init_group_values_for_even_size(group_names, base_vector)
        
        self._init_center_group_values(group_names, base_vector)
        
        self._set_group_aliases(group_names)

        # 各グループのマスク総和（スコア正規化等に利用）
        for key in group_names.values():
            self.total_val[key] = np.sum(self.group_val[key])
        for key in group_names.keys():
            self.total_val[key] = self.total_val[group_names[key]]

    def _init_group_values_for_odd_size(self, group_names, base_vector):
        """奇数サイズ用の Corner / MidEdge / Wing グループを初期化する。"""
        center_feature_start = 36 * self.center_num
        self.group_val[group_names['A']] = self._group_vector_slice(-192, None)
        self.group_val[group_names['B']] = self._group_vector_slice(center_feature_start, center_feature_start + 288)
        if self.size >= 5:
            self.group_val[group_names['C']] = self._group_vector_slice(center_feature_start + 288, center_feature_start + 864)
            if self.size == 7:
                self.group_val[group_names['c']] = self._group_vector_slice(center_feature_start + 864, -192)
            else:
                self._set_empty_group(group_names['c'], base_vector)
        else:
            self._set_empty_group(group_names['C'], base_vector)
            self._set_empty_group(group_names['c'], base_vector)

    def _init_group_values_for_even_size(self, group_names, base_vector):
        """偶数サイズ用の Corner / MidEdge / Wing グループを初期化する。"""
        center_feature_start = 36 * self.center_num
        self.group_val[group_names['A']] = self._group_vector_slice(-192, None)
        self._set_empty_group(group_names['B'], base_vector)
        if self.size >= 4:
            self.group_val[group_names['C']] = self._group_vector_slice(center_feature_start, center_feature_start + 576)
            if self.size == 6:
                self.group_val[group_names['c']] = self._group_vector_slice(center_feature_start + 576, -192)
            else:
                self._set_empty_group(group_names['c'], base_vector)
        else:
            self._set_empty_group(group_names['C'], base_vector)
            self._set_empty_group(group_names['c'], base_vector)

    def _init_center_group_values(self, group_names, base_vector):
        """X / Plus / Oblique / CoreCenter の group mask を初期化する。"""
        for key in ['D','d','E','e','F','f','G']:
            self.group_val[group_names[key]] = self._center_group_vector(key, base_vector)

    def _center_group_vector(self, key, base_vector):
        """center 系 group key に対応する mask ベクトルを返す。"""
        if self.Centers:
            return base_vector.copy()
        group_vector = base_vector.copy()
        for face_index in range(6):
            for group_index in self.group_indices[key]:
                vector_index = face_index + 6 * (face_index * self.center_num + group_index - 4 * (self.size - 1))
                group_vector[0,vector_index] = 1
        return group_vector

    def _set_empty_group(self, group_name, base_vector):
        """指定した group に空ベクトルを代入する。"""
        self.group_val[group_name] = base_vector.copy()

    def _empty_group_vector(self):
        """評価用グループの空ベクトルを返す。"""
        return np.zeros((1,self.ips),dtype = 'f')

    def _group_vector_slice(self, start, end):
        """perfect_data の指定区間だけを立てたグループベクトルを返す。"""
        group_vector = self._empty_group_vector()
        group_vector[0,start:end] = self.perfect_data[start:end]
        return group_vector

    def _group_name_map(self):
        """短い group key と意味ベース名の対応を返す。"""
        return {
            'A': 'Corner',
            'B': 'MidEdge',
            'C': 'Wing-Layer2',
            'c': 'Wing-Layer3',
            'D': 'XCenter-Layer2',
            'd': 'XCenter-Layer3',
            'E': 'PlusCenter-Layer2',
            'e': 'PlusCenter-Layer3',
            'F': 'ObliqueCenter-A',
            'f': 'ObliqueCenter-B',
            'G': 'CoreCenter',
        }

    def _set_group_aliases(self, group_names):
        """既存コード互換のため、旧 short key でも同じベクトルを引けるようにする。"""
        for short_key, long_key in group_names.items():
            self.group_val[short_key] = self.group_val[long_key]
        
        
    


    def _init_myperms_index(self):
        """(piece, color) から候補 myperm 群を引く逆引き表を構築する。"""
        self._init_empty_myperms_index()
        self._register_myperms_index_entries()
        self._init_myperms_order()

    def _init_empty_myperms_index(self):
        """未一致色ごとの空の myperm 候補リストを用意する。"""
        self.myperms_dict = {}
        self.piece_color_counter = {}
        self._init_empty_center_myperms_index()
        self._init_empty_edge_myperms_index()
        self._init_empty_corner_myperms_index()

    def _init_empty_center_myperms_index(self):
        """center piece 用の逆引きキーを作る。"""
        for piece in self.center_index:
            for color in ['R','O','B','G','Y','W','X']:
                if self.default_color[piece] != color:
                    self.myperms_dict[(piece,color)] = []
                    self.piece_color_counter[(piece,color)] = 0

    def _init_empty_edge_myperms_index(self):
        """edge piece 用の逆引きキーを作る。"""
        for piece in self.edge_index:
            for color in self.edge_key:
                if self.default_color[piece] != color:
                    self.myperms_dict[(piece,color)] = []
                    self.piece_color_counter[(piece,color)] = 0

    def _init_empty_corner_myperms_index(self):
        """corner piece 用の逆引きキーを作る。"""
        for piece in self.corner_index:
            for color in self.corner_key:
                if self.default_color[piece] != color:
                    self.myperms_dict[(piece,color)] = []
                    self.piece_color_counter[(piece,color)] = 0

    def _register_myperms_index_entries(self):
        """各 myperm を1回ずつ適用し、変化する piece/color に登録する。"""
        for key, moves in self.myperms.items():
            if self._skip_myperms_index_key(key):
                continue
            self._register_single_myperms_index_entry(key, moves)

    def _skip_myperms_index_key(self, key):
        """逆引き登録から除外する myperm 名か判定する。"""
        base_key = myperm_base_key(key)
        return base_key[:3] in ["L2E","L4I","L4J"] or base_key[:5] in ['Super']

    def _register_single_myperms_index_entry(self, key, moves):
        """1つの myperm を適用して、変化した piece/color に key を追加する。"""
        self._apply_inverse_moves(moves)
        self._register_changed_center_entries(key)
        self._register_changed_edge_entries(key)
        self._register_changed_corner_entries(key)
        self._apply_moves(moves)

    def _apply_inverse_moves(self, moves):
        """逆順の move を適用して観測用の盤面へ移す。"""
        for move in self.invert_moves(moves):
            self.make_move(move)

    def _apply_moves(self, moves):
        """通常順の move を適用して盤面を元へ戻す。"""
        for move in moves:
            self.make_move(move)

    def _register_changed_center_entries(self, key):
        """色が変化した center piece に myperm key を登録する。"""
        for piece in self.center_index:
            color = self.state[piece[0]]
            if color != self.default_color[piece]:
                self.myperms_dict[(piece,color)].append(key)

    def _register_changed_edge_entries(self, key):
        """色が変化した edge piece に myperm key を登録する。"""
        for piece in self.edge_index:
            color = self.state[piece[0]] + self.state[piece[1]]
            if color != self.default_color[piece]:
                self.myperms_dict[(piece,color)].append(key)

    def _register_changed_corner_entries(self, key):
        """色が変化した corner piece に myperm key を登録する。"""
        for piece in self.corner_index:
            color = self.state[piece[0]] + self.state[piece[1]] + self.state[piece[2]]
            if color != self.default_color[piece]:
                self.myperms_dict[(piece,color)].append(key)



    def get_chenged_pieces_keys_from_moves(self,Moves):
        current_state = self.state.copy()
        self.reset()
        for m in Moves:
            self.make_move(m)

        S = self._get_changed_pieces_keys()
        self.state = current_state
        return S

    def _get_changed_pieces_keys(self):
        S = self._register_changed_center_keys()
        S += self._register_changed_edge_keys()
        S += self._register_changed_corner_keys()
    
        return S

    def _register_changed_center_keys(self):
        S = []
        for piece in self.center_index:
            color = self.state[piece[0]]
            if color != self.default_color[piece]:
                S.append((piece,color))
        
        return S
        

    def _register_changed_edge_keys(self):
        S = []
        for piece in self.edge_index:
            color = self.state[piece[0]] + self.state[piece[1]]
            if color != self.default_color[piece]:
                S.append((piece,color))

        return S

    def _register_changed_corner_keys(self):
        S = []
        for piece in self.corner_index:
            color = self.state[piece[0]] + self.state[piece[1]] + self.state[piece[2]]
            if color != self.default_color[piece]:
                S.append((piece,color))

        return S





    def _init_myperms_order(self):
        """評価用の group 順序インデックスを作る。"""
        self.myperms_order = {}
        group_names = self._group_name_map()
        for key in ['A','B','C','c','D','d','E','e','F','f','G']:
            indices = self._group_order_indices(key)
            self.myperms_order[group_names[key]] = indices
            self.myperms_order[key] = indices

    def _group_order_indices(self, key):
        """1つの group key に対応する盤面 index 順序を返す。"""
        indices = []
        for face_index in [0,1,2,3,4,5]:
            indices += list(np.array(self.group_indices[key]) + self.surface_num * face_index)
        return indices
                


    def myperms_dict_key(self,S):
        L = []
        for key in self.myperms_dict:
            if S in self.myperms_dict[key]:
                L.append(key)

        return L

    
    def create_new_set(self):
        i = len(self.my_scrambles2.keys())
        self.my_scrambles2[i] = {}
        self.my_scramble_changed_piece_keys[i] = {}
        for k in self.my_scrambles2[0].keys():
            self.my_scrambles2[i][k] = set([]) 

    def register_scramble_sequence(self, level, moves):
        """Register one scramble sequence and cache its changed-piece keys."""
        normalized_moves = tuple(moves)
        self.my_scrambles2[level][normalized_moves[-1]].add(normalized_moves)
        self.my_scramble_changed_piece_keys[level][normalized_moves] = tuple(
            self.get_chenged_pieces_keys_from_moves(normalized_moves)
        )

    def get_registered_scramble_changed_piece_keys(self, level, moves):
        """Return cached changed-piece keys for a registered scramble sequence."""
        normalized_moves = tuple(moves)
        return self.my_scramble_changed_piece_keys[level].get(normalized_moves)

    def make_move(self,key):
        self.state = self.state[self.move[key]]


    def scramble(self,N,Move = None,difficult_mode = False,scramble_mode = None,flip = None,rotate = None,swap = False,add_moves = None,transform_N = None,flip_inside = None,move_count_policy = 'prefer_rare'):
        if Move != None:
            return self._apply_moves_and_return(Move)

        if scramble_mode not in ['Centers','myperms','Edges','Slices','OLL']:
            return self._simple_scramble(N)

        move_count_policy = self.scramble_selector.resolve_move_count_policy(move_count_policy, add_moves)
        return self._guided_scramble(N,move_count_policy,transform_N,flip_inside)

    def _apply_moves_and_return(self, Move):
        for m in Move:
            self.make_move(m)
        return tuple(Move)

    def _simple_scramble(self, N):
        move_lis = self._generate_simple_scramble_moves(N)
        self._apply_scramble_moves(move_lis)
        return tuple(move_lis)

    def _generate_simple_scramble_moves(self, N):
        move_lis = []
        for _ in range(N):
            move_lis.append(random.choice(self.move_keys))
        return tuple(move_lis)

    def _guided_scramble(self, N, move_count_policy, transform_N, flip_inside):
        move_count = self._init_scramble_count()
        transform_index = self._resolve_transform_index(transform_N)
        use_flip_inside = self._resolve_flip_inside(flip_inside)

        move_lis = []
        for level_index in range(N):
            selected_moves = self._guided_scramble_moves(level_index,move_count,move_count_policy)
            transformed_moves = self._transform_scramble_moves(selected_moves,transform_index,use_flip_inside)
            self._append_scramble_moves(move_lis,transformed_moves)
            self._apply_scramble_moves(transformed_moves)

        return tuple(move_lis)

    def _init_scramble_count(self):
        return self.scramble_selector.init_move_count()

    def _guided_scramble_moves(self, level_index, move_count, move_count_policy):
        return self.scramble_selector.select(level_index, move_count, move_count_policy = move_count_policy)

    def _transform_scramble_moves(self, moves, transform_index, use_flip_inside):
        transformed_moves = self.transform(moves,transform_index)
        if use_flip_inside:
            transformed_moves = self.flip_inside_moves(transformed_moves)
        return transformed_moves

    def _append_scramble_moves(self, move_lis, moves):
        move_lis += list(moves)

    def _apply_scramble_moves(self, moves):
        for move in moves:
            self.make_move(move)

    def _resolve_transform_index(self, transform_N):
        if transform_N is not None:
            return transform_N
        if self.F2L or self.OLL:
            return random.choice([0])
        if self.size >= 6:
            return random.randrange(96)
        return random.randrange(48)

    def _resolve_flip_inside(self, flip_inside):
        if flip_inside is not None:
            return flip_inside
        return bool(random.randint(0,1))

    def _collect_scramble_candidates(self, level_index):
        level_index = self.scramble_selector.resolve_level(level_index)
        return self.scramble_selector.collect_candidates(level_index)

    def _select_scramble_candidate(self, candidates, Count, move_count_policy, level_index):
        if move_count_policy == 'prefer_frequent':
            return self.scramble_selector._select_candidate_max(candidates, Count, level_index)
        return self.scramble_selector._select_candidate_min(candidates, Count, level_index)

    def _select_candidate_max(self, candidates, Count, level_index):
        return self.scramble_selector._select_candidate_max(candidates, Count, level_index)

    def _select_candidate_min(self, candidates, Count, level_index):
        return self.scramble_selector._select_candidate_min(candidates, Count, level_index)

    def _evaluate_piece_color_value(self,changed_piece_keys):
        if not changed_piece_keys:
            return 0
        return sum(self.piece_color_counter[key] for key in changed_piece_keys)

    def _update_piece_color_counter(self,changed_piece_keys):
        self.scramble_selector.update_piece_color_counter(changed_piece_keys)

    def _update_count(self, Count, M):
        self.scramble_selector.update_count(Count, M)

    def _update_counter_stats(self, level_index, M):
        self.scramble_selector.update_counter_stats(level_index, M)

    def swap_2_3(self,move):
        if move[0] == "2":
            return "3" + move[1:]
        elif move[0] == "3":
            return "2" + move[1:]
        else:
            return move



    def flip_moves(self,Moves,axis = None):
        """指定軸の鏡映ルールで手順列を変換する。"""
        return self.move_ops.flip_moves(Moves,axis = axis)

    def rotate_moves(self,Moves,axis = None):
        """指定回転ルールで手順列を回転変換する。"""
        return self.move_ops.rotate_moves(Moves,axis = axis)

    def diag_flip_moves(self,Moves):
        """対角反転ルールで手順列を変換する。"""
        return self.move_ops.diag_flip_moves(Moves)

    def invert_str(self,s):
        """1手だけ逆回転に変換する。"""
        return self.move_ops.invert_str(s)

    def invert_moves(self,Moves):
        """手順列を逆順・逆回転にした列を返す。"""
        return self.move_ops.invert_moves(Moves)

    def swap_moves(self,Moves):
        """2層・3層の手を入れ替える補助変換を適用する。"""
        return self.move_ops.swap_moves(Moves)

    def flip_inside(self,s):
        """1手だけ内外反転ルールで変換する。"""
        return self.move_ops.flip_inside(s)

    def flip_inside_moves(self,Moves):
        """内外反転ルールで手順列を変換する。"""
        return self.move_ops.flip_inside_moves(Moves)
    


    def reduce(self,move_lis):
        """同一 state に戻るループを消して、手順列を state ベースで簡約する。"""
        reduced_moves = []
        visited_states = [''.join(self.state)]
        kept_indices = []

        for original_index, move in enumerate(move_lis):
            reduced_moves, visited_states, kept_indices = self._reduce_step(
                move,
                original_index,
                reduced_moves,
                visited_states,
                kept_indices,
            )

        self._restore_state_after_reduce(move_lis)
        return (tuple(reduced_moves),kept_indices)

    def _reduce_step(self, move, original_index, reduced_moves, visited_states, kept_indices):
        """1手進めて、既出状態なら巻き戻し、未出なら履歴へ追加する。"""
        self.make_move(move)
        state_key = ''.join(self.state)

        if state_key in visited_states:
            return self._trim_history_to_revisited_state(state_key, reduced_moves, visited_states, kept_indices)

        reduced_moves.append(move)
        visited_states.append(state_key)
        kept_indices.append(original_index)
        return reduced_moves, visited_states, kept_indices

    def _trim_history_to_revisited_state(self, state_key, reduced_moves, visited_states, kept_indices):
        """再訪した state の位置まで履歴を巻き戻して、ループ部分を消す。"""
        trim_index = visited_states.index(state_key)
        return (
            reduced_moves[:trim_index],
            visited_states[:trim_index + 1],
            kept_indices[:trim_index],
        )

    def _restore_state_after_reduce(self, move_lis):
        """reduce 中に進めた state を、元の state へ戻す。"""
        for move in self.invert_moves(move_lis):
            self.make_move(move)

    def simplify(self,move_lis):
        """同じ面・同じ層の連続手をまとめて手順列を簡約する。"""
        return self.move_ops.simplify(move_lis)

    def conjugate(self,A,B):
        """共役 A B A^-1 を作って簡約した手順列を返す。"""
        return self.move_ops.conjugate(A,B)

    def commutator(self,A,B):
        """交換子 A B A^-1 B^-1 を作って簡約した手順列を返す。"""
        return self.move_ops.commutator(A,B)
        
    def reset(self):
        self.state[:] = self.state_0

    def makedata(self):
        """現在 state を AI 入力ベクトルへ変換する。"""
        center_one_hot = self._center_one_hot()
        input_vector = np.zeros(self.ips,dtype = 'f')
        offset = self._write_center_features(input_vector, center_one_hot)
        offset = self._write_edge_features(input_vector, offset)
        self._write_corner_features(input_vector, offset)
        return input_vector

    def _center_one_hot(self):
        """center state を色ごとの one-hot 行列に変換する。"""
        centers = np.zeros(6 * self.center_num,dtype = str)
        for i in range(6):
            centers[self.center_num*i:self.center_num*(i+1)] = self.state[
                4 * (self.size-1)+self.surface_num * i:self.surface_num * (i+1)
            ]

        center_one_hot = np.zeros((6 * self.center_num,6),dtype = 'f')
        for i in range(6):
            center_one_hot[:,i][centers == self.colors[i]] = 1
        return center_one_hot

    def _write_center_features(self, input_vector, center_one_hot):
        """center one-hot を入力ベクトル先頭へ書き込み、次の offset を返す。"""
        offset = 36 * self.center_num
        input_vector[:offset] = center_one_hot.reshape(-1)
        return offset

    def _write_edge_features(self, input_vector, offset):
        """edge 特徴を入力ベクトルへ書き込み、次の offset を返す。"""
        for edge_indices in self.edge_index:
            edge_key = self.state[edge_indices[0]] + self.state[edge_indices[1]]
            if edge_key != 'XX':
                input_vector[offset + self.edge_key[edge_key]] = 1
                offset += 24
        return offset

    def _write_corner_features(self, input_vector, offset):
        """corner 特徴を入力ベクトルへ書き込む。"""
        for corner_indices in self.corner_index:
            corner_key = (
                self.state[corner_indices[0]]
                + self.state[corner_indices[1]]
                + self.state[corner_indices[2]]
            )
            if corner_key != 'XXX':
                input_vector[offset + self.corner_key[corner_key]] = 1
                offset += 24
        
    def is_perfect(self):
        return (self.state == self.state_0).all()


    def transform(self,s,i,flip_inside = False,invert = False):
        """変換indexに対応する対称変換を手順列へ適用する。"""
        return self.move_ops.transform(s,i,flip_inside = flip_inside,invert = invert)

    def _transformation_key(self, transform_index, invert = False):
        """変換indexから、実際に適用する変換手順列を取り出す。"""
        return self.move_ops._transformation_key(transform_index,invert = invert)

    def _apply_transform_step(self, moves, transform_step):
        """変換手順1つ分だけ手順列へ反映する。"""
        return self.move_ops._apply_transform_step(moves,transform_step)

    def make_transformations(self,s,Moves):
        """全ての対称変換について、scramble列とmove列の組を作る。"""
        return self.move_ops.make_transformations(s,Moves)

    def piece_display_name(self, piece_type, piece):
        """Return a position label using this cube's move faces and solved colors."""
        if piece_type == 'Center' and len(piece) == 1:
            return self._center_display_name(piece[0])
        if piece_type == 'Edge' and len(piece) == 2:
            return self._edge_display_name(piece)
        labels = ','.join(self._face_and_solved_color(index) for index in piece)
        return f'{piece_type}-({labels})'

    def _center_display_name(self, index):
        face_label, row_index, col_index = self._index_to_face_row_col(index)
        horizontal_label = self._coordinate_axis_label(face_label, col_index, axis = 'horizontal')
        vertical_label = self._coordinate_axis_label(face_label, row_index, axis = 'vertical')
        return f'Center-({self._face_and_solved_color(index)},{horizontal_label},{vertical_label})'

    def _edge_display_name(self, piece):
        face_labels = [self._face_and_solved_color(index) for index in piece]
        axis_label = self._edge_axis_label(piece)
        return f'Edge-({face_labels[0]},{face_labels[1]},{axis_label})'

    def _edge_axis_label(self, piece):
        face_labels = [self._move_face_label(index) for index in piece]
        incident_families = {self._axis_family(face_label) for face_label in face_labels}
        candidates = []
        for index in piece:
            face_label, row_index, col_index = self._index_to_face_row_col(index)
            if col_index not in (0, self.size - 1):
                label = self._coordinate_axis_label(face_label, col_index, axis = 'horizontal')
                if self._axis_family(label) not in incident_families:
                    candidates.append(label)
            if row_index not in (0, self.size - 1):
                label = self._coordinate_axis_label(face_label, row_index, axis = 'vertical')
                if self._axis_family(label) not in incident_families:
                    candidates.append(label)
        if candidates:
            return candidates[0]
        return '?'

    def _index_to_face_row_col(self, index):
        face_index = int(index // self.surface_num)
        face_color = RUBIKS_SOLVED_COLORS_BY_FACE_INDEX[face_index]
        face_label = RUBIKS_MOVE_FACE_LABELS_BY_INDEX[face_index]
        row_index, col_index = np.argwhere(self.Nums[face_color] == index)[0]
        return face_label, int(row_index), int(col_index)

    def _move_face_label(self, index):
        return RUBIKS_MOVE_FACE_LABELS_BY_INDEX[int(index // self.surface_num)]

    def _face_and_solved_color(self, index):
        color = str(self.state_0[index])
        return f'{self._move_face_label(index)}:{RUBIKS_COLOR_NAMES.get(color, color)}'

    def _coordinate_axis_label(self, face_label, coordinate, axis):
        positive_label, negative_label, toward_positive = RUBIKS_AXIS_INFO[face_label][axis]
        if toward_positive:
            positive_distance = self.size - coordinate
            negative_distance = coordinate + 1
        else:
            positive_distance = coordinate + 1
            negative_distance = self.size - coordinate

        axis_pair = frozenset({positive_label, negative_label})
        if positive_distance == negative_distance and axis_pair in RUBIKS_MIDDLE_AXIS_LABEL:
            return RUBIKS_MIDDLE_AXIS_LABEL[axis_pair]
        if positive_distance < negative_distance:
            return self._format_axis_distance(positive_label, positive_distance)
        return self._format_axis_distance(negative_label, negative_distance)

    def _format_axis_distance(self, axis_label, distance):
        if distance <= 1:
            return axis_label
        return f'{distance}{axis_label}'

    def _axis_family(self, label):
        axis_label = label[-1]
        return RUBIKS_AXIS_FAMILY[axis_label]
