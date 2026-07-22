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
        self.myperms2_source_aliases = {}
        self._init_group_indices()

    def _add_myperm2(self, name, moves, legacy = None):
        """Register one source myperm while keeping its previous source name."""
        self.myperms2[name] = moves
        if legacy and legacy != name:
            self.myperms2_source_aliases[legacy] = name
        return name

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
        self._register_myperms2_general()
        self._register_myperms2_f2l_oll()

    def _register_myperms2_base(self):
        """基本パターンと大分類の手順を登録する。"""
        # 命名メモ:
        # - X-Center / Plus-Center / Oblique-Center は動かす center の配置族。
        # - 4 / 6 は見た目上で動く center 数、末尾の英字は variant を表す。

        self.myperms2['SuperFlip'] = (" U "," R2"," F "," B "," R "," B2"," R "," U2"," L "," B2"," R "," U'"," D'"," R2"," F "," R'"," L "," B2"," U2"," F2")
        self.myperms2['SuperDiagFlip'] = (' U ', ' R2', ' F ', ' B ', ' R ', ' B2', ' R ', ' U2', ' L ', ' B2', ' R ', " U'", " D'", ' R2', ' F ', " L'", ' R ', ' U2', ' D2', ' B2', ' D2', ' B2')
        self.myperms2['SuperDiag'] = (" U2"," D2"," F2"," B2"," R2"," L2")

        self.myperms2['SuperTwist'] = (" U2"," B2"," D "," L2"," F'"," B'"," R2"," D "," F2"," D2"," B "," R2"," U'"," D'"," L2"," B ")
        self.myperms2['SuperTwist2'] = (" D'", ' L ', ' D ', ' R2', " D'", " L'", ' D ', ' R2', ' U2', " B'", ' D ', ' B ', ' U2', " B'", " D'", ' B ', " R'", ' U ', ' R ', ' D2', " R'", " U'", ' R ', ' D2', ' L2', " F'", ' R ', ' F ', ' L2', " F'", " R'", ' F ', ' R ', ' B2', " R'", ' D ', ' F2', " D'", ' R ', ' B2', " R'", ' D ', ' F2', " D'")
        self.myperms2['SuperRotate-A'] = (" L'"," R "," U "," D'"," F'"," B "," L'"," R ")
        self.myperms2['SuperRotate-B'] = (" L'"," R "," U2"," D2"," L'"," R "," F2"," B2")

        self.myperms2['Super-CubeInCube'] = (" F "," L "," F "," U'"," R "," U "," F2"," L2"," U'"," L'"," B "," D'"," B'"," L2"," U ")
        self.myperms2['Super-3Checker'] = (" F "," B2"," R'"," D2"," B "," R "," U "," D'"," R "," L'"," D'"," F'"," R2"," D "," F2"," B'") 
        self.myperms2['Super-6Checker'] = (" F "," B2"," R'"," D2"," B "," R "," U "," D'"," R "," L'"," D'"," F'"," R2"," D "," F2"," B'"," L2"," R2"," U2"," D2"," F2"," B2") 
        self.myperms2['Super-Cage'] = (" L "," U "," F2"," R "," L'"," U2"," B'"," U "," D "," B2"," L "," F "," B'"," R'"," L "," F'"," R ")
        self.myperms2['Super-Stripe'] = (" F "," U "," F "," R "," L2"," B "," D'"," R "," D2"," L "," D'"," B "," R2"," L "," F "," U "," F ")
        

        self.myperms2['Super-CrossA'] = (' R2', ' L2', " U'", ' R2', ' L2', ' U2', ' B2', ' F2', ' D ', ' B2', ' F2', ' U2')
        self.myperms2['Super-CrossB'] = (" L2"," U2"," D2"," F2"," U2"," D2"," L2"," R2"," B2"," R2")
        self.myperms2['Super-CrossC'] = (" R'", ' F2', ' B2', ' R ', ' D ', ' F2', ' B2', " D'", " U'", ' F2', ' B2', ' D ', ' B2', ' F2', ' R ', ' B2', ' F2', " L'")
        self.myperms2['Super-CrossD'] = (' U ', ' R2', ' U2', ' D2', ' B2', ' F2', ' L2', ' B2', ' F2', ' U ', ' D2')

        self.myperms2['Super-Crossa'] = self.myperms2['Super-CrossA'] + self.myperms2['SuperTwist']
        self.myperms2['Super-Crossb'] = self.myperms2['Super-CrossB'] + self.myperms2['SuperTwist']
        self.myperms2['Super-Crossc'] = self.myperms2['Super-CrossC'] + self.myperms2['SuperTwist']
        self.myperms2['Super-Crossd'] = self.myperms2['Super-CrossD'] + self.myperms2['SuperTwist']


        self.myperms2['BigQA'] = (" z'", ' L2', ' F ', " B'", " U'", ' B ', ' D ', ' B ', " D'", ' F ', " B'", ' L ', " B'", " L'", " F'", ' R ', ' F ', " R'", ' F2', ' R2', ' F ', ' B ', ' U ', " F'", " U'", " B'", ' R2', ' F ', " U'")
        self.myperms2['BigQB'] = (" R'", " F'", " D'", " B'", ' D ', " B'", ' F ', " L'", ' B ', ' L ', ' B ', ' R ', " B'", ' F ', " z'")
        self.myperms2['BigQG'] = (' F ', " L'", " U'", ' F ', ' U ', ' F2', " R'", ' F ', " D'", " B'", ' D ', " B'", ' F ', " L'", ' B ', ' L ', ' B ', ' F2', ' R ', " F'", " B'", ' F2', " z'", ' L ', ' D ', " F'", " D'", ' F ')
        self.myperms2['BigQH'] = (" L2"," R'", " F'", " D'", " B'", ' D ', " B'", ' F ', " L'", ' B ', ' L ', ' B ', ' R ', " B'", ' F ', " z'", " L2")

        self.myperms2['BigQC00-'] = (" F'"," L'"," R'", " F'", " D'", " B'", ' D ', " B'", ' F ', " L'", ' B ', ' L ', ' B ', ' R ', " B'", ' F ', " z'"," L "," F ")
        self.myperms2['BigQD00-'] = (" F2"," L'"," R'", " F'", " D'", " B'", ' D ', " B'", ' F ', " L'", ' B ', ' L ', ' B ', ' R ', " B'", ' F ', " z'"," L "," F2")
        self.myperms2['BigQE00-'] = (" F "," L'"," R'", " F'", " D'", " B'", ' D ', " B'", ' F ', " L'", ' B ', ' L ', ' B ', ' R ', " B'", ' F ', " z'"," L "," F'")
        self.myperms2['BigQF00-'] = (" L'"," R'", " F'", " D'", " B'", ' D ', " B'", ' F ', " L'", ' B ', ' L ', ' B ', ' R ', " B'", ' F ', " z'"," L ")

        self.myperms2['BigQC01-'] = self.invert_moves(self.myperms2['BigQC00-'])
        self.myperms2['BigQD01-'] = self.invert_moves(self.myperms2['BigQD00-'])
        self.myperms2['BigQE01-'] = self.invert_moves(self.myperms2['BigQE00-'])
        self.myperms2['BigQF01-'] = self.invert_moves(self.myperms2['BigQF00-'])


        self.myperms2['BigRA'] = (" U'", " F'", " U'", ' F ', " z'", " B'", ' U ', ' F ', ' U ', " R'", ' F ', " B'", ' D ', ' B ', ' D ', " B'", " D'", ' B ', " F'", ' R ', ' B ', " R'", " B'", ' U2', ' L ', ' U ', " L'", ' U ', ' B2', " D'", ' R ', ' D ', ' B2')
        self.myperms2['BigRB'] = (" U'", " z'", " L'", " F'", " B'", ' F2', " U'", " F'", ' B ', " L'", ' D2', ' L2', " B'", ' F ', " U'", ' L ', " F'", ' L ', ' F2', " R'", ' F ', ' R ', ' F ', ' L ', " F'", ' R2', ' F ', " L'", " F'", ' R2', ' F2')
        self.myperms2['BigRF'] = (" U'", " F'", " U'", " z'", ' F ', " B'", ' U ', ' F ', ' U ', " R'", " B'", ' F ', ' D ', " F'", ' B ', ' D ', " B'", " D'", ' B ', ' R ', ' B ', " R'", " B'", ' U2', ' L ', ' U ', " L'", ' U ', ' B2', " D'", ' R ', ' D ', ' B2')
        self.myperms2['BigRJ00-'] = (" L'", " F'", " U'", " B'", ' U ', ' F ', " B'", " R'", ' B ', ' R ', ' B ', ' L ', " B'", ' F ', " z'", ' D ', ' L ', " D'", " L'", " F'", " D'", " F'", ' D2', " R'", ' D2', ' R ', ' F2', ' R2', ' U ', " L'", " U'", ' R2', ' U ', ' L ', " U'")
        self.myperms2['BigRJ01-'] = (' F2', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F ', " L'", ' F ', ' R ', " F'", ' L ') + self.myperms2['BigQA']
        self.myperms2['BigRJ02-'] = (' U ', ' B ', ' R ', ' F ', " R'", ' F ', " B'", ' D ', " F'", " D'", " F'", " U'", " B'", ' F ', " U'", " z'", " U'", ' L ', ' U ', ' B ', " L'", " B'", " U'", ' F ', " U'", " F'", ' U2', " L'", ' U ', " R'", " U'", ' L ', ' U ', ' R ', " U'")
        self.myperms2['BigRK'] = (' U ', ' B ', ' R ', ' F ', " R'", " B'", ' F ', ' D ', " F'", " D'", " F'", " U'", ' F ', " B'", " z'", ' L ', ' U ', " L'", " U'", " F'", ' L ', ' F ', ' U ', " B'", ' U ', ' B ', ' U2', ' R2', ' B2', ' R ', ' F2', " R'", ' B2', ' R ', ' F2', ' R ')

    
    def _register_myperms2_x_perms(self):
        """ParitySwap系とその派生手順を登録する。"""
        # 命名メモ:
        # - ParitySwap-* は corner 2つ + midedge 2つの同時 swap。
        # - ParityCycle-* は corner 4つ + edge 2つの置換。
        # - A/B/F/J/K は corner 配置 family、末尾番号は family 内 variant。
        if self.size <= 3:
            PLLParity = ()
        elif self.size <= 5:
            PLLParity = ("2F2"," R2"," U2","2F2"," U2"," R2","2F2")
        elif self.size <= 7:
            PLLParity = ("2F2","3F2"," R2"," U2","2F2","3F2"," U2"," R2","2F2","3F2")


        self.myperms2['ParitySwap-A0-'] = (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F2', ' R ') + PLLParity
        self.myperms2['ParitySwap-A1-'] = (" R'", ' F2', ' D2', " B'", " L'", ' B ', ' D2', " F'", ' R ', " F'") + PLLParity
        self.myperms2['ParitySwap-A2-'] = PLLParity + (" R'", ' F2', ' D2', " B'", " L'", ' B ', ' L ', ' D2', " L'", " F'", ' R ', ' F2', " L'", " F'", ' L ', ' F2')
        self.myperms2['ParitySwap-A3-'] = PLLParity + (' F2', " R'", ' F2', ' D2', " B'", " L'", ' B ', ' L ', ' D2', " L'", " F'", ' R ', ' F2', " L'", " F'", ' L ')
        self.myperms2['ParitySwap-A4-'] = PLLParity + (' F2', ' U2', ' F2', ' U2', ' F ', ' R ', " L'", ' U2', " R'", ' L ', " F'", ' B ', ' U ', " B'", ' U ', ' R2', " D'", ' F ', ' D ', ' R2')
        self.myperms2['ParitySwap-A5-'] = PLLParity + (' U2', ' F2', ' U2', ' F ', ' R ', " L'", ' U2', " R'", ' L ', " F'", ' B ', ' U ', " B'", ' U ', ' R2', " D'", ' F ', ' D ', ' R2', " F2")
        
        
        self.myperms2['ParitySwap-B0-'] = PLLParity + (" L2"," F2"," U2"," L'"," U2"," L2"," F2"," L'"," U2"," L2"," U2"," F2"," L'"," F2")
        self.myperms2['ParitySwap-B1-'] = PLLParity + (" F2"," L "," F2"," U2"," L2"," U2"," L "," F2"," L2"," U2"," L "," U2"," F2"," L2")
        
        self.myperms2['ParitySwap-B2-'] = (" R2", " U2", " B2", " R'", " B2", " R2", " U2", " R ", " B2", " R2", " B2", " U2", " R ", " U2") + PLLParity
        self.myperms2['ParitySwap-B3-'] = (" U2", " R'", " U2", " B2", " R2", " B2", " R'", " U2", " R2", " B2", " R ", " B2", " U2", " R2") + PLLParity

        self.myperms2['ParitySwap-F0-'] = PLLParity + (" R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F2', ' R ', " F ")
        self.myperms2['ParitySwap-F1-'] = PLLParity + (" F'", " R'", " F2", " D2", " B'", " L'", " B ", " D2", " F'", " R ")      
        self.myperms2['ParitySwap-F2-'] = (" B ", " L'", ' B2', ' D2', " F'", " R'", ' F ', ' D2', " B'", ' L ', " B2") + PLLParity
        self.myperms2['ParitySwap-F3-'] = (' B2', " L'", ' B ', ' D2', " F'", ' R ', ' F ', ' D2', ' B2', ' L ', " B'") + PLLParity
        self.myperms2['ParitySwap-F4-'] = (" U2", " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F2', ' R ', " F ", " U2") + PLLParity
        self.myperms2['ParitySwap-F5-'] = (" U2", " F'", " R'", " F2", " D2", " B'", " L'", " B ", " D2", " F'", " R ", " U2") + PLLParity




        
        self.myperms2['ParitySwap-J0-'] = (' B2', " L'", ' B2', ' D2', " F'", " R'", ' F ', ' D2', " B'", ' L ', ' B ') + PLLParity
        self.myperms2['ParitySwap-J1-'] = (" B'", " L'", ' B ', ' D2', " F'", ' R ', ' F ', ' D2', ' B2', ' L ', ' B2') + PLLParity

        self.myperms2['ParitySwap-J2-'] = (' F2', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F ', " L'", ' F ', ' R ', " F'", ' L ') + PLLParity
        self.myperms2['ParitySwap-J3-'] = (" L'", ' F ', " R'", " F'", ' L ', " F'", ' D2', " B'", " L'", ' B ', ' D2', " F'", ' R ', ' F2') + PLLParity

        self.myperms2['ParitySwap-J4-'] = (' B2', ' L2', ' D2', ' F2', ' D2', ' L2', " B'", ' U2', ' L2', ' D ', ' F ', " D'", ' L2', ' U ', " B'", " U'") + PLLParity
        self.myperms2['ParitySwap-J5-'] = (' U ', ' B ', " U'", ' L2', ' D ', " F'", " D'", ' L2', ' U2', ' B ', ' L2', ' D2', ' F2', ' D2', ' L2', ' B2') + PLLParity
    
        self.myperms2['ParitySwap-K0-'] = (" R'", ' U2', ' L ', ' F2', " L'", ' F2', ' R2', ' U2', ' R ', ' U2', " R'", ' U2', ' F2', ' R2', ' F2') + PLLParity
        self.myperms2['ParitySwap-K1-'] = (' R2', ' F2', ' U2', ' R ', ' U2', " R'", ' U2', ' R2', ' F2', ' L ', ' F2', " L'", ' U2', ' R ', ' F2') + PLLParity

        self.myperms2['ParityCycle-QA0-'] = (" F "," R "," U "," R'"," U'"," R "," U'"," R'"," U'"," R "," U "," R'"," F'") + PLLParity
        self.myperms2['ParityCycle-QA1-'] = (" F "," R "," U'"," R'"," U "," R "," U "," R'"," U "," R "," U'"," R'"," F'") + PLLParity
        self.myperms2['ParityCycle-QB0-'] = (" F'"," R "," U "," R'"," U'"," R "," U'"," R'"," U'"," R "," U "," R'"," F ") + PLLParity
        self.myperms2['ParityCycle-QB1-'] = (" F'"," R "," U'"," R'"," U "," R "," U "," R'"," U "," R "," U'"," R'"," F ") + PLLParity
        self.myperms2['ParityCycle-QC0-'] = (" R "," U "," R'"," U'"," R "," U'"," R'"," U'"," R "," U "," R'") + PLLParity
        self.myperms2['ParityCycle-QC1-'] = (" R "," U'"," R'"," U "," R "," U "," R'"," U "," R "," U'"," R'") + PLLParity

 


        if self.size % 2 == 1:
            self._add_myperm2('CtrCore6p[3x2][B>R>D;F>L>U]', (" M "," E "," M'"," E'"), legacy = 'CenterX00')
            self._add_myperm2('CtrCore4s[B<>F;D<>U]', (" M "," E2"," M'"," E2"), legacy = 'CenterX01')


        self.myperms2['ParitySwap-XB-'] = (' U ', " F'", ' R ') + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', " F'", " U'")
        self.myperms2['ParitySwap-XC-'] = (' F2', ' R ') + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2')
        self.myperms2['ParitySwap-XD-'] = (' F ', ' R ') + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F ')
        self.myperms2['ParitySwap-XE-'] = (' R ',) + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', " F2")
        self.myperms2['ParitySwap-XF-'] = (" F'", ' R ') + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', " F'")
        self.myperms2['ParitySwap-XG-'] = self.conjugate((" R2",),self.myperms2['ParitySwap-A0-'])
        self.myperms2['ParitySwap-XH-'] = self.conjugate((" U'"," F'"," R "),self.myperms2['ParitySwap-A0-'])
        

        self.myperms2['ParitySwap-YA-'] = PLLParity + (" R "," U "," R'"," U'"," R'"," F "," R2"," U'"," R'"," U'"," R "," U "," R'"," F'")
        self.myperms2['ParitySwap-YB-'] = self.conjugate((" U "," F'"," R "),self.myperms2['ParitySwap-YA-'])
        self.myperms2['ParitySwap-YC-'] = self.conjugate((" F2"," R "),self.myperms2['ParitySwap-YA-'])
        self.myperms2['ParitySwap-YD-'] = self.conjugate((" F "," R "),self.myperms2['ParitySwap-YA-'])
        self.myperms2['ParitySwap-YE-'] = self.conjugate((" R ",),self.myperms2['ParitySwap-YA-'])
        self.myperms2['ParitySwap-YF-'] = self.conjugate((" F'"," R "),self.myperms2['ParitySwap-YA-'])
        self.myperms2['ParitySwap-YG-'] = self.conjugate((" R2",),self.myperms2['ParitySwap-YA-'])
        self.myperms2['ParitySwap-YH-'] = self.conjugate((" R'"," U'"," F "," U "),self.myperms2['ParitySwap-YA-'])


        
        self.myperms2['ParitySwap-ZA-'] = PLLParity + (' U2', " B'", ' U2', ' B ', ' U2',' D2', " R'", " B'", ' R ', ' D2', " L'", ' F ', " L'", " F'", ' L2')
        self.myperms2['ParitySwap-ZB-'] = self.conjugate((" F'"," U "," L'"),self.myperms2['ParitySwap-ZA-'])
        self.myperms2['ParitySwap-ZC-'] = self.conjugate((" U2"," L "),self.myperms2['ParitySwap-ZA-'])
        self.myperms2['ParitySwap-ZD-'] = self.conjugate((" U "," L "),self.myperms2['ParitySwap-ZA-'])
        self.myperms2['ParitySwap-ZE-'] = self.conjugate((" L ",),self.myperms2['ParitySwap-ZA-'])
        self.myperms2['ParitySwap-ZF-'] = self.conjugate((" U'"," L "),self.myperms2['ParitySwap-ZA-'])
        self.myperms2['ParitySwap-ZG-'] = self.conjugate((" L2",),self.myperms2['ParitySwap-ZA-'])
        self.myperms2['ParitySwap-ZH-'] = self.conjugate((" F "," U "," L'"),self.myperms2['ParitySwap-ZA-'])

        
        self.myperms2['ParitySwap-JXB-'] = (" R2", ' U ', " F'", ' R ') + PLLParity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', " F'", " U'", " R2")
        self.myperms2['ParitySwap-JYB-'] = (" R2", ' U ', " F'", ' R ') + PLLParity + (' R ', ' U ', " R'", " U'", " R'", ' F ', ' R2', " U'", " R'", " U'", ' R ', ' U ', " R'", " F'", " R'", ' F ', " U'", " R2")
        self.myperms2['ParitySwap-JZB-'] = (" U'", ' B ', " R'") + PLLParity + (" B'", ' R ', " B'", ' D2', ' F ', " L'", " F'", ' D2', ' B ', ' U ', ' L ', ' U2', " L'", ' D ', ' L ', ' U2', " L'", " D'")

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
            self._add_myperm2('CtrCore4[D>L>U>R]+ME2s[UL<>UR]', (' S ', ' D ', ' S ', " D'", ' S ', " D'", ' S ', ' D ', ' S2', " D'", ' S ', ' D2', ' L2', " S'", " D'", " S'", ' D ', ' L2', " D'"), legacy = 'CenterMidEdgeSwap-QA')
            self._add_myperm2('CtrCore4[D>L>U>R]+ME2[UL>RU]', (' S ', " D'", ' S ', ' D2', ' L2', " D'", ' S ', " D'", ' S ', ' D2', ' L2', " D'", ' S '), legacy = 'CenterMidEdgeSwap-QB')
            self._add_myperm2('CtrCore4[D>L>U>R]+ME2s[UF<>UR]', (' R2', " S'", ' R2', ' S2', " U'", ' S ', ' U ', ' S ', ' U ', ' S ', " U'", ' S2', ' D ', ' M ', ' D2', " M'", ' D ', " S'"), legacy = 'CenterMidEdgeSwap-QC00')
            self._add_myperm2('CtrCore4[D>L>U>R]+ME2[UF>RU]', (' R2', " S'", ' R2', ' S2', " U'", ' R2', ' S ', ' U ', ' S ', ' U ', ' S ', " U'", ' S ', " U'", ' R2', ' U '), legacy = 'CenterMidEdgeSwap-QD00')
            self._add_myperm2('CtrCore4[D>L>U>R]+ME2s[DF<>UR]', (' S ', " D'", ' S ', ' D ', ' S ', ' D ', ' S ', " D'", ' S ', ' M ', ' U ', ' M ', ' U2', " M'", ' U ', " M'"), legacy = 'CenterMidEdgeSwap-QE00')
            self._add_myperm2('CtrCore4[D>L>U>R]+ME2[DF>RU]', (' S ', ' D ', ' S ', ' D2', ' L2', ' D ', ' S ', ' D ', ' S ', " D'", ' S ', " D'", ' L2', ' D '), legacy = 'CenterMidEdgeSwap-QF00')
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2s[UF<>UR]', self.invert_moves(self.myperms2['CtrCore4[D>L>U>R]+ME2s[UF<>UR]']), legacy = 'CenterMidEdgeSwap-QC01')
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2[UF>RU]', self.invert_moves(self.myperms2['CtrCore4[D>L>U>R]+ME2[UF>RU]']), legacy = 'CenterMidEdgeSwap-QD01')
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2s[DF<>UR]', self.invert_moves(self.myperms2['CtrCore4[D>L>U>R]+ME2s[DF<>UR]']), legacy = 'CenterMidEdgeSwap-QE01')
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2[DF>RU]', self.invert_moves(self.myperms2['CtrCore4[D>L>U>R]+ME2[DF>RU]']), legacy = 'CenterMidEdgeSwap-QF01')
            self._add_myperm2('CtrCore4[D>L>U>R]+ME2s[DL<>UR]', (' S ', ' R ', ' S ', " R'", ' S ', " R'", ' L2', ' S ', ' R ', ' L2', ' S2', " R'", ' U2', ' R ', " S'", " R'", ' U2', ' R '), legacy = 'CenterMidEdgeSwap-QG')
            self._add_myperm2('CtrCore4[D>L>U>R]+ME2[DL>RU]', (' D ', ' S ', " D'", ' S ', ' D ', ' S ', ' D ', ' S ', " D'", ' S ', " D'", ' S ', ' U2', " S'", ' U2'), legacy = 'CenterMidEdgeSwap-QH')


            self._add_myperm2('CtrCore4[D>R>U>L]+ME2s[UB<>UF]', (" S'", " U'", " S'", ' U ', " S'", ' U ', " S'", " U'", ' S2', ' U ', ' R2', " U'", ' S ', ' U ', " S'", ' R2', ' U2', ' S ', ' U '), legacy = 'CenterMidEdgeSwap-PA')
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2[UB>FU]', (" E'", " M'", ' E ', ' D ', " M'", ' D2', ' F2', ' D ', " M'", ' D ', " M'", ' D2', ' F2', ' D ', " M'"), legacy = 'CenterMidEdgeSwap-PB')
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2[FL>FR]', (" E'", ' B ', " E'", " B'", " E'", " B'", " E'", ' B ', ' M ', " E'", " F'", ' M ', ' F2', ' D2', " M'", " F'", " M'", ' F ', ' D2', " F'"), legacy = 'CenterMidEdgeSwap-PX')
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2s[FL<>RF]', (' E ', ' R2', " E'", ' R ', " S'", ' L ', " S'", " L'", " S'", " L'", " S'", ' L ', " S'", ' R '), legacy = 'CenterMidEdgeSwap-PY')
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2s[RF<>UF]', (' R ', " S'", ' U ', " S'", " U'", ' S ', ' U2', " S'", ' U2', " S'", " U'", " S'", ' U ', ' S2', ' R2', ' S ', ' R '), legacy = 'CenterMidEdgeSwap-PC')
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2[RF>FU]', (" F'", ' U2', ' F ', ' M ', ' F ', ' M ', " F'", ' M ', " F'", ' M ', ' U2', ' F ', ' E ', ' M ', " E'", ' M ', ' U2', " M'", ' U2'), legacy = 'CenterMidEdgeSwap-PD')
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2[BR>FD]', (' M ', " E'", " M'", ' F ', " E'", " F'", " E'", " F'", " E'", ' F ', " E'", " M'", " B'", " M'", ' B2', ' M ', " B'", ' M '), legacy = 'CenterMidEdgeSwap-PE')
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2s[BR<>DF]', (' M ', " E'", " M'", ' F ', " E'", " F'", " E'", " F'", " E'", ' F ', " E'", ' B ', ' M ', ' B2', " M'", ' B '), legacy = 'CenterMidEdgeSwap-PF')
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2s[DB<>UF]', (" D'", " S'", ' U ', " S'", " U'", " S'", " U'", " S'", ' U ', " S'", " D'", " B'", ' M ', ' B ', ' D2', " B'", " M'", ' B '), legacy = 'CenterMidEdgeSwap-PG')
            self._add_myperm2('CtrCore4[D>R>U>L]+ME2[DB>FU]', (" D'", " S'", ' U ', " S'", " U'", " S'", " U'", " S'", ' U ', " S'", ' D ', ' M ', ' D2', " M'", ' D2'), legacy = 'CenterMidEdgeSwap-PH')


            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2s[UL<>UR]', (' S ', ' D ', ' S ', " D'", ' S ', " D'", ' S ', ' D ', ' S2', " D'", ' S ', ' D2', ' L2', " S'", " D'", " S'", ' D ', ' L2', " D'"," M'"," S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-SA')
            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2[UL>RU]', (" M'", ' S2', ' M ', ' S ', ' D ', " S'", ' D2', ' R2', ' D ', " S'", ' D ', " S'", ' D2', ' R2', ' D ', " S'"), legacy = 'CenterMidEdgeSwap-SB')
            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2s[UF<>UR]', (' R2', " S'", ' R2', ' S2', " U'", ' S ', ' U ', ' S ', ' U ', ' S ', " U'", ' S2', ' D ', ' M ', ' D2', " M'", ' D ', " S'"," M'"," S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-SC00')
            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2[UF>RU]', (' R2', " S'", ' R2', ' S2', " U'", ' R2', ' S ', ' U ', ' S ', ' U ', ' S ', " U'", ' S ', " U'", ' R2', ' U '," M'"," S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-SD00')
            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2s[DF<>UR]', (' S ', " D'", ' S ', ' D ', ' S ', ' D ', ' S ', " D'", ' S ', ' M ', ' U ', ' M ', ' U2', " M'", ' U ', " M2"," S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-SE00')
            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2[DF>RU]', (' S ', ' D ', ' S ', ' D2', ' L2', ' D ', ' S ', ' D ', ' S ', " D'", ' S ', " D'", ' L2', ' D '," M'"," S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-SF00')
            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2s[DL<>UR]', (' S ', ' R ', ' S ', " R'", ' S ', " R'", ' L2', ' S ', ' R ', ' L2', ' S2', " R'", ' U2', ' R ', " S'", " R'", ' U2', ' R '," M'"," S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-SG00')
            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2[DL>RU]', (" L'", " S'", ' L ', " S'", " L'", " S'", " L'", " S'", ' L ', " S'", ' L ', " E'", ' S2', ' E ', ' S ', ' R2', ' S ', ' R2'), legacy = 'CenterMidEdgeSwap-SH00')

            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2s[UF<>UR]', (' R2', " S'", ' R2', ' S2', " U'", ' S ', ' U ', ' S ', ' U ', ' S ', " U'", ' S2', ' D ', ' M ', ' D2', " M'", ' D ', " S'"," E "," S2"," E'"," S2"), legacy = 'CenterMidEdgeSwap-SC01')
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2[UF>RU]', (' R2', " S'", ' R2', ' S2', " U'", ' R2', ' S ', ' U ', ' S ', ' U ', ' S ', " U'", ' S ', " U'", ' R2', ' U '," E "," S2"," E'"," S2"), legacy = 'CenterMidEdgeSwap-SD01')
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2s[DF<>UR]', (' S ', " D'", ' S ', ' D ', ' S ', ' D ', ' S ', " D'", ' S ', ' M ', ' U ', ' M ', ' U2', " M'", ' U ', " M'"," E "," S2"," E'"," S2"), legacy = 'CenterMidEdgeSwap-SE01')
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2[DF>RU]', (' S ', ' D ', ' S ', ' D2', ' L2', ' D ', ' S ', ' D ', ' S ', " D'", ' S ', " D'", ' L2', ' D '," E "," S2"," E'"," S2"), legacy = 'CenterMidEdgeSwap-SF01')
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2s[DL<>UR]', (' S ', ' R ', ' S ', " R'", ' S ', " R'", ' L2', ' S ', ' R ', ' L2', ' S2', " R'", ' U2', ' R ', " S'", " R'", ' U2', ' R '," E "," S2"," E'"," S2"), legacy = 'CenterMidEdgeSwap-SG01')
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2[DL>RU]', (" L'", " S'", ' L ', " S'", " L'", " S'", " L'", " S'", ' L ', " S'", ' L ', " M'", ' S2', ' M ', ' S ', ' R2', ' S ', ' R2'), legacy = 'CenterMidEdgeSwap-SH01')
            
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2s[UB<>UF]', (" S'", " U'", " S'", ' U ', " S'", ' U ', " S'", " U'", ' S2', ' U ', ' R2', " U'", ' S ', ' U ', " S'", ' R2', ' U2', ' S ', ' U '," M'"," S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-RA')
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2[UB>FU]', (" E'", " M'", ' E ', ' D ', " M'", ' D2', ' F2', ' D ', " M'", ' D ', " M'", ' D2', ' F2', ' D ', " M2"," S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-RB')
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2[FL>FR]', (" E'", ' B ', " E'", " B'", " E'", " B'", " E'", ' B ', ' M ', " E'", " F'", ' M ', ' F2', ' D2', " M'", " F'", " M'", ' F ', ' D2', " F'"," M'"," S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-RX')
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2s[FL<>RF]', (' E ', ' R2', " E'", ' R ', " S'", ' L ', " S'", " L'", " S'", " L'", " S'", ' L ', " S'", ' R '," M'"," S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-RY')
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2s[RF<>UF]', (' R ', " S'", ' U ', " S'", " U'", ' S ', ' U2', " S'", ' U2', " S'", " U'", " S'", ' U ', ' S2', ' R2', ' S ', ' R '," M'"," S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-RC00')
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2[RF>FU]', (" F'", ' U2', ' F ', ' M ', ' F ', ' M ', " F'", ' M ', " F'", ' M ', ' U2', ' F ', ' E ', ' M ', " E'", ' M ', ' U2', " M'", ' U2'," M'"," S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-RD00')
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2[BR>FD]', (' M ', " E'", " M'", ' F ', " E'", " F'", " E'", " F'", " E'", ' F ', " E'", " M'", " B'", " M'", ' B2', ' M ', " B'", " S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-RE00')
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2s[BR<>DF]', (' M ', " E'", " M'", ' F ', " E'", " F'", " E'", " F'", " E'", ' F ', " E'", ' B ', ' M ', ' B2', " M'", ' B '," M'"," S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-RF00')
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2s[DB<>UF]', (" D'", " S'", ' U ', " S'", " U'", " S'", " U'", " S'", ' U ', " S'", " D'", " M'", " F'", " M'", ' F ', ' D2', " F'", ' M ', ' F ', " S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-RG')
            self._add_myperm2('CtrCore6s[B<>F;D<>R;L<>U]+ME2[DB>FU]', (" D'", " S'", ' U ', " S'", " U'", " S'", " U'", " S'", ' U ', " S'", ' D ', ' M ', ' D2', " M'", ' D2'," M'"," S2"," M "," S2"), legacy = 'CenterMidEdgeSwap-RH')
            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2s[RF<>UF]', (' R ', " S'", ' U ', " S'", " U'", ' S ', ' U2', " S'", ' U2', " S'", " U'", " S'", ' U ', ' S2', ' R2', ' S ', ' R '," E "," S2"," E'"," S2"), legacy = 'CenterMidEdgeSwap-RC01')
            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2[RF>FU]', (" F'", ' U2', ' F ', ' M ', ' F ', ' M ', " F'", ' M ', " F'", ' M ', ' U2', ' F ', ' E ', ' M ', " E'", ' M ', ' U2', " M'", ' U2'," E "," S2"," E'"," S2"), legacy = 'CenterMidEdgeSwap-RD01')
            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2[BR>FD]', (' M ', " E'", " M'", ' F ', " E'", " F'", " E'", " F'", " E'", ' F ', " E'", " M'", " B'", " M'", ' B2', ' M ', " B'", ' M '," E "," S2"," E'"," S2"), legacy = 'CenterMidEdgeSwap-RE01')
            self._add_myperm2('CtrCore6s[B<>F;D<>L;R<>U]+ME2s[BR<>DF]', (' M ', " E'", " M'", ' F ', " E'", " F'", " E'", " F'", " E'", ' F ', " E'", ' B ', ' M ', ' B2', " M'", ' B '," E "," S2"," E'"," S2"), legacy = 'CenterMidEdgeSwap-RF01')




            self._add_myperm2('C2[UBR>RFU]+CtrCore4[D>L>U>R]~v01', self.myperms2['ParitySwap-A0-'] + self.myperms2['CtrCore4[D>L>U>R]+ME2s[UL<>UR]'], legacy = 'CenterCornerSwap-A00')
            self._add_myperm2('C2[UBR>RFU]+CtrCore4[D>L>U>R]~v02', self.myperms2['ParitySwap-A1-'] + self.myperms2['CtrCore4[D>L>U>R]+ME2s[UL<>UR]'], legacy = 'CenterCornerSwap-A01')
            self._add_myperm2('C2s[UBR<>UFL]+CtrCore4[D>L>U>R]~v01', self.myperms2['ParitySwap-B0-'] + self.myperms2['CtrCore4[D>L>U>R]+ME2s[UL<>UR]'], legacy = 'CenterCornerSwap-B00')
            self._add_myperm2('C2s[UBR<>UFL]+CtrCore4[D>L>U>R]~v02', self.myperms2['ParitySwap-B1-'] + self.myperms2['CtrCore4[D>L>U>R]+ME2s[UL<>UR]'], legacy = 'CenterCornerSwap-B01')
            self._add_myperm2('C2[DFR>BRU]+CtrCore4[D>L>U>R]~v01', self.myperms2['ParitySwap-F0-'] + self.myperms2['CtrCore4[D>L>U>R]+ME2s[UL<>UR]'], legacy = 'CenterCornerSwap-F00')
            self._add_myperm2('C2[DFR>BRU]+CtrCore4[D>L>U>R]~v02', self.myperms2['ParitySwap-F1-'] + self.myperms2['CtrCore4[D>L>U>R]+ME2s[UL<>UR]']           , legacy = 'CenterCornerSwap-F01')
            self._add_myperm2('C2[DRB>LUF]+CtrCore4[D>L>U>R]~v01', self.myperms2['ParitySwap-J0-'] + self.myperms2['CtrCore4[D>L>U>R]+ME2s[UL<>UR]'], legacy = 'CenterCornerSwap-J00')
            self._add_myperm2('C2[DRB>LUF]+CtrCore4[D>L>U>R]~v02', self.myperms2['ParitySwap-J1-'] + self.myperms2['CtrCore4[D>L>U>R]+ME2s[UL<>UR]'], legacy = 'CenterCornerSwap-J01')
            self._add_myperm2('C2s[DLF<>UBR]+CtrCore4[D>L>U>R]~v01', self.myperms2['ParitySwap-J2-'] + self.myperms2['CtrCore4[D>L>U>R]+ME2s[UL<>UR]'], legacy = 'CenterCornerSwap-J02')
            self._add_myperm2('C2s[DLF<>UBR]+CtrCore4[D>L>U>R]~v02', self.myperms2['ParitySwap-J3-'] + self.myperms2['CtrCore4[D>L>U>R]+ME2s[UL<>UR]'], legacy = 'CenterCornerSwap-J03')
            self._add_myperm2('C2[DRB>FLU]+CtrCore4[D>L>U>R]~v01', self.myperms2['ParitySwap-J4-'] + self.myperms2['CtrCore4[D>L>U>R]+ME2s[UL<>UR]'], legacy = 'CenterCornerSwap-J04')
            self._add_myperm2('C2[DRB>FLU]+CtrCore4[D>L>U>R]~v02', self.myperms2['ParitySwap-J5-'] + self.myperms2['CtrCore4[D>L>U>R]+ME2s[UL<>UR]'], legacy = 'CenterCornerSwap-J05')
            self._add_myperm2('C2[UBR>FUR]+CtrCore4[D>L>U>R]~v01', self.myperms2['ParitySwap-ZA-'] + self.myperms2['CtrCore4[D>L>U>R]+ME2s[UL<>UR]'], legacy = 'CenterCornerSwap-K00')
            self._add_myperm2('C2[UBR>FUR]+CtrCore4[D>L>U>R]~v02', self.myperms2['ParitySwap-ZA-'] + self.myperms2['CtrCore4[D>L>U>R]+ME2s[UL<>UR]'], legacy = 'CenterCornerSwap-K01')

            

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

        
        self._add_myperm2('C4[DLF>FLU;UBR>RFU;UFL>LFD;URF>UBR]+EAll3[FL>FU>RU]', (" F "," U "," F'"," U'"), legacy = 'OutCommutator00')
        self._add_myperm2('C4[DLF>BUL;UFL>FUR;ULB>FDL;URF>UFL]+EAll3[FL>FU>LU]', (" F "," U'"," F'"," U "), legacy = 'OutCommutator01')
        self._add_myperm2('C4[DFR>FDL;DLF>DFR;UBR>LUF;UFL>RUB]+EAll3[DF>LF>UR]', (" F2"," U "," F'"," U'"," F'"), legacy = 'OutCommutator02')
        self._add_myperm2('C4[DFR>LBU;DLF>FLU;UFL>LFD;ULB>DFR]+EAll3[DF>LF>UL]', (" F2"," U'"," F'"," U "," F'")    , legacy = 'OutCommutator03')
        self._add_myperm2('C4[DFR>RFU;DLF>UBR;UBR>LFD;URF>FRD]+EAll3[DF>UR>RF]', (" F'"," U "," F'"," U'"," F2"), legacy = 'OutCommutator04')
        self._add_myperm2('C4[DFR>FDL;DLF>DFR;ULB>FUR;URF>BUL]+EAll3[DF>UL>RF]', (" F'"," U'"," F'"," U "," F2"), legacy = 'OutCommutator05')

        self._add_myperm2('C5[DFR>UFL>URF>UBR>DLF]+EAll3[DF>UF>UR]', (" F2"," U "," F2"," U'") , legacy = 'OutCommutator06')
        self._add_myperm2('C5[DFR>DLF>UBR>URF>UFL]+EAll3[DF>UR>UF]', (" U "," F2"," U'"," F2") , legacy = 'OutCommutator07')
        self._add_myperm2('C5[DFR>RUB>UFL>LFD>RFU]+EAll3[FL>FR>RU]', (" F "," U "," F2"," U'"," F ") , legacy = 'OutCommutator08')
        self._add_myperm2('C5[DFR>RFU>BUL>LFD>UFL]+EAll3[FL>FR>LU]', (" F "," U'"," F2"," U "," F ")     , legacy = 'OutCommutator09')
        self._add_myperm2('C5[DFR>RFU>LFD>UFL>RUB]+EAll3[FL>RU>FR]', (" F'"," U "," F2"," U'"," F'") , legacy = 'OutCommutator10')
        self._add_myperm2('C5[DFR>UFL>LFD>BUL>RFU]+EAll3[FL>LU>FR]', (" F'"," U'"," F2"," U "," F'") , legacy = 'OutCommutator11')

        if self.size >= 4:
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v01', (" F ","2U "," F'","2U'"), legacy = 'SideCommutator00')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v02', (" F ","2U'"," F'","2U "), legacy = 'SideCommutator01')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v03', ("2U "," F'","2U'"," F "), legacy = 'SideCommutator02')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v04', ("2U'"," F'","2U "," F "), legacy = 'SideCommutator03')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v05', (" F2","2U "," F'","2U'"," F'"), legacy = 'SideCommutator04')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v06', (" F2","2U'"," F'","2U "," F'")    , legacy = 'SideCommutator05')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v07', (" F'","2U "," F'","2U'"," F2"), legacy = 'SideCommutator06')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v08', (" F'","2U'"," F'","2U "," F2"), legacy = 'SideCommutator07')

            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v01', (" F2","2U "," F2","2U'") , legacy = 'SideCommutator08')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v02', (" F2","2U'"," F2","2U ") , legacy = 'SideCommutator09')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v03', ("2U "," F2","2U'"," F2"), legacy = 'SideCommutator10')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v04', ("2U'"," F2","2U "," F2"), legacy = 'SideCommutator11')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v05', (" F ","2U "," F2","2U'"," F ") , legacy = 'SideCommutator12')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v06', (" F ","2U'"," F2","2U "," F ")     , legacy = 'SideCommutator13')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v07', (" F'","2U "," F2","2U'"," F'") , legacy = 'SideCommutator14')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-5~v08', (" F'","2U'"," F2","2U "," F'") , legacy = 'SideCommutator15')

            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-6p[3x2]~v01', (" F ","2U2"," F'","2U2"), legacy = 'SideCommutator16')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-6p[3x2]~v02', ("2U2"," F'","2U2"," F "), legacy = 'SideCommutator17')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-6p[3x2]~v03', (" F2","2U2"," F'","2U2"," F'")   , legacy = 'SideCommutator18')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-6p[3x2]~v04', (" F'","2U2"," F'","2U2"," F2"), legacy = 'SideCommutator19')

            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-6p[3x2]~v01', (" F2","2U2"," F2","2U2") , legacy = 'SideCommutator20')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-6p[3x2]~v02', ("2U2"," F2","2U2"," F2") , legacy = 'SideCommutator21')
            self._add_myperm2('CtrObl6p[3x2]+CtrPlus3+CtrX6p[3x2]+W2-6p[3x2]~v03', (" F ","2U2"," F2","2U2"," F ")    , legacy = 'SideCommutator22')

        if self.size % 2 == 1:
            self._add_myperm2('CtrPlus12p[3x4]+ME5[DF>FU>FR>LF>BL]', (" F "," E "," F'"," E'"), legacy = 'MidCommutator00')
            self._add_myperm2('CtrPlus12p[3x4]+ME5[DF>FU>LB>FL>RF]', (" E "," F'"," E'"," F "), legacy = 'MidCommutator01')
            self._add_myperm2('CtrPlus12p[3x4]+ME5[DF>BL>RF>FL>FU]', (" F2"," E "," F'"," E'"," F'"), legacy = 'MidCommutator02')
            self._add_myperm2('CtrPlus12p[3x4]+ME5[DF>LF>FR>LB>FU]', (" F'"," E "," F'"," E'"," F2"), legacy = 'MidCommutator03')

            self._add_myperm2('CtrPlus12p[3x4]+ME6[BR>FD>FL;LB>UF>RF]', (" F "," E2"," F'"," E2"), legacy = 'MidCommutator04')
            self._add_myperm2('CtrPlus12p[3x4]+ME6[BR>FL>FU;DF>LB>RF]', (" E2"," F'"," E2"," F "), legacy = 'MidCommutator05')
            self._add_myperm2('CtrPlus12p[3x4]+ME6[BR>FR>FD;FL>FU>BL]', (" F2"," E2"," F'"," E2"," F'"), legacy = 'MidCommutator06')
            self._add_myperm2('CtrPlus12p[3x4]+ME6[BR>FU>FR;DF>LF>LB]', (" F'"," E2"," F'"," E2"," F2"), legacy = 'MidCommutator07')



    def _register_myperms2_midedge_general(self):
        """奇数サイズ向けのMidEdge系手順を登録する。"""
        # 命名メモ:
        # - MidEdge3-* は midedge 3個の cycle。
        # - MidEdge4-* は midedge 4個の cycle / 2-2 swap 型。
        # - family 文字は位置関係、末尾 A/B/C... は向き違い。
        # - MidEdgeFlip2/4-* は midedge の flip 用 family。
        if self.size % 2 == 1:           
            self._add_myperm2('ME3[DB>DF>UF]', (' M ', ' D2', " M'", ' D2'), legacy = 'MidEdge3-I-A')
            self._add_myperm2('ME3[DB>FD>FU]', (" U'", " M'", ' U ', ' F2', " U'", ' M ', ' U ', ' F2'), legacy = 'MidEdge3-I-B')
            self._add_myperm2('ME3[DB>FD>UF]', (" F'", " E'", " F'", ' D2', ' F ', ' E ', " F'", ' D2', ' F2'), legacy = 'MidEdge3-I-C')
            self._add_myperm2('ME3[DB>DF>FU]', (' D2', " B'", ' M ', ' B ', ' D2', " B'", " M'", ' B '), legacy = 'MidEdge3-I-D')

            self._add_myperm2('ME4[DB>DF;DF>FU;UB>BU;UF>DB]', (' M2', ' D2', ' M2', " D'", " M'", " D'", ' B2', ' D ', ' M ', " D'", ' B2'), legacy = 'MidEdge4-II-A')
            self._add_myperm2('ME4[DB>FD;DF>FU;UB>BU;UF>BD]', (" U'", " M'", ' U ', ' F2', " U'", ' M ', ' U ', " M'", " U'", " M'", ' U ', ' F2', " U'", ' M ', " U'", ' M ', ' U2'), legacy = 'MidEdge4-II-B')
            self._add_myperm2('ME4[DB>FD;DF>UF;UB>BU;UF>DB]', (' U2', ' M ', ' U2', ' D2', " B'", ' M ', ' B ', ' D2', " B'", " M'", ' B ', " M'"), legacy = 'MidEdge4-II-C')
            self._add_myperm2('ME4[DB>DF;DF>UF;UB>BU;UF>BD]', (' M ', ' D2', ' B ', ' M ', " B'", ' D2', ' B ', " M'", ' B ', " M'", ' B2'), legacy = 'MidEdge4-II-D')


            self._add_myperm2('ME4s[UB<>UF;UL<>UR]', (" M2"," U "," M2"," U2"," M2"," U "," M2"), legacy = 'MidEdge4-H-A')
            self._add_myperm2('ME4[UB>FU;UL>UR]', (' M2', " U'", ' M ', ' U2', " M'", " U'", ' M ', ' U ', ' B2', " U'", " M'", ' U ', ' B2', " U'", ' M2'), legacy = 'MidEdge4-H-B')
            self._add_myperm2('ME4[UB>FU;UF>UB;UL>UR;UR>LU]', (' S2', " U'", ' S ', ' U2', ' L2', " S'", " U'", ' S ', ' U ', ' L2', " U'", ' S '), legacy = 'MidEdge4-H-C')
            self._add_myperm2('ME4[UB>FU;UL>RU]', (" M'", ' U ', ' B2', " U'", " M'", ' U ', ' M2', ' B2', ' M2', ' B2', " U'", ' M ', ' U ', ' B2', " U'", ' M '), legacy = 'MidEdge4-H-D')

            self._add_myperm2('ME4s[DL<>DR;UB<>UF]', (" U'", ' S2', ' U2', ' S2', " U'"), legacy = 'MidEdge4-T-A')
            self._add_myperm2('ME4[DL>DR;UB>FU]', (' D ', ' M ', ' D2', " M'", ' D ', ' M ', " D'", ' F2', ' D ', " M'", " D'", ' F2', ' D '), legacy = 'MidEdge4-T-B')
            self._add_myperm2('ME4[DL>DR;DR>LD;UB>FU;UF>UB]', (' S ', " U'", ' L2', ' U ', " S'", " U'", ' S ', ' L2', ' U2', " S'", " U'"), legacy = 'MidEdge4-T-C')
            self._add_myperm2('ME4[DL>RD;UB>FU]', (' S ', ' U ', ' L2', " U'", ' S2', ' U ', ' R2', " U'", ' S ', ' U ', ' L2', ' R2', " U'"), legacy = 'MidEdge4-T-D')



            self._add_myperm2('ME3[UF>UL>UR]', (" M2"," U'"," M "," U2"," M'"," U'"," M2"), legacy = 'MidEdge3-U-A')
            self._add_myperm2('ME3[UF>LU>RU]', (" M ", ' U ', " M'", ' U2', ' M ', ' U ', " M'"), legacy = 'MidEdge3-U-B')
            self._add_myperm2('ME3[UF>UL>RU]', (' S2', " U'", ' R2', ' U ', " S'", " U'", ' R2', ' U ', " S'"), legacy = 'MidEdge3-U-C')
            self._add_myperm2('ME3[UF>LU>UR]', (" S'", ' U ', ' L2', " U'", " S'", ' U ', ' L2', " U'", ' S2'), legacy = 'MidEdge3-U-D')

            self._add_myperm2('ME3[DL>DR>UF]', (" D'", ' M ', ' D2', " M'", " D'"), legacy = 'MidEdge3-V-A')
            self._add_myperm2('ME3[DL>DR>FU]', (" M'", ' D ', " M'", ' D2', ' M ', ' D ', ' M '), legacy = 'MidEdge3-V-B')
            self._add_myperm2('ME3[DL>RD>UF]', (' U ', ' L2', " U'", ' S ', ' U ', ' L2', " U'", " S'"), legacy = 'MidEdge3-V-C')
            self._add_myperm2('ME3[DL>RD>FU]', (" S'", " U'", ' R2', ' U ', ' S ', " U'", ' R2', ' U '), legacy = 'MidEdge3-V-D')


            self._add_myperm2('ME4[UB>BU;UF>UL;UL>UR;UR>FU]', (" M'", " U'", ' B2', ' U ', " M'", " U'", ' B2', ' M2', ' U2', ' M2', " U'", ' M2'), legacy = 'MidEdge4-UU-A')
            self._add_myperm2('ME4[UB>BU;UF>LU;UL>UR;UR>UF]', (' M2', " U'", ' M2', ' U2', ' M2', ' B2', " U'", ' M ', ' U ', ' B2', " U'", ' M '), legacy = 'MidEdge4-UU-B')
            self._add_myperm2('ME4[UB>BU;UF>UL;UL>RU;UR>UF]', (" S'", " U'", ' L2', ' U ', " S'", " U'", ' L2', " S'", ' U2', ' S ', " U'", ' S2'), legacy = 'MidEdge4-UU-C')
            self._add_myperm2('ME4[UB>BU;UF>LU;UL>RU;UR>FU]', (' S2', ' U ', ' R2', " U'", " S'", ' U ', ' R2', " U'", ' S2', ' U ', ' S ', ' U2', " S'", ' U ', ' S '), legacy = 'MidEdge4-UU-D')


            self._add_myperm2('ME4[DL>DR;DR>FU;UB>BU;UF>DL]', (' M ', " D'", ' F2', ' D ', " M'", " D'", ' F2', ' M2', ' D2', ' M2', " D'"), legacy = 'MidEdge4-VV-A')
            self._add_myperm2('ME4[DL>DR;DR>UF;UB>BU;UF>LD]', (" D'", ' M2', ' D2', ' M2', ' F2', " D'", ' M ', ' D ', ' F2', " D'", " M'"), legacy = 'MidEdge4-VV-B')
            self._add_myperm2('ME4[DL>RD;DR>UF;UB>BU;UF>DL]', (" S'", ' U ', ' R2', " U'", ' S ', ' U ', ' R2', ' S ', ' U2', " S'", ' U '), legacy = 'MidEdge4-VV-C')
            self._add_myperm2('ME4[DL>RD;DR>FU;UB>BU;UF>LD]', (' S ', " U'", ' S ', ' U2', " S'", " U'", ' S2', ' U ', ' R2', " U'", ' S ', ' U ', ' R2', " U'"), legacy = 'MidEdge4-VV-D')

            self._add_myperm2('ME3[FL>LU>RU]', (' S ', ' R ', ' E ', ' R2', " E'", ' R ', " S'"), legacy = 'MidEdge3-P-A')
            self._add_myperm2('ME3[FL>UL>UR]', (" R'"," E "," R2"," E'"," R "," S "," R2"," S'"), legacy = 'MidEdge3-P-B')
            self._add_myperm2('ME3[FL>LU>UR]', (' S ', " L'", ' U2', ' L ', " S'", " L'", ' U2', ' L '), legacy = 'MidEdge3-P-C')
            self._add_myperm2('ME3[FL>UL>RU]', (" L'", ' U ', " M'", " U'", ' L ', ' U ', ' M ', " U'"), legacy = 'MidEdge3-P-D')
            self._add_myperm2('ME3[FL>RU>LU]', self.invert_moves(self.myperms2['ME3[FL>LU>RU]']), legacy = 'MidEdge3-P-E')
            self._add_myperm2('ME3[FL>UR>UL]', self.invert_moves(self.myperms2['ME3[FL>UL>UR]']), legacy = 'MidEdge3-P-F')
            self._add_myperm2('ME3[FL>UR>LU]', self.invert_moves(self.myperms2['ME3[FL>LU>UR]']), legacy = 'MidEdge3-P-G')
            self._add_myperm2('ME3[FL>RU>UL]', self.invert_moves(self.myperms2['ME3[FL>UL>RU]']), legacy = 'MidEdge3-P-H')

            self._add_myperm2('ME3[DR>UF>UL]', (' L2', " D'", ' M ', ' D2', " M'", " D'", ' L2'), legacy = 'MidEdge3-R-A')
            self._add_myperm2('ME3[DR>FU>UL]', (' S2', ' R ', ' F ', " R'", ' S2', ' R ', " F'", " R'"), legacy = 'MidEdge3-R-B')
            self._add_myperm2('ME3[DR>FU>LU]', (' L2', ' U ', ' L2', " U'", ' S ', ' U ', ' L2', " U'", " S'", ' L2'), legacy = 'MidEdge3-R-C')
            self._add_myperm2('ME3[DR>UF>LU]', (" D'", ' M ', " D'", " S'", ' D2', ' S ', " D'", " M'", ' D '), legacy = 'MidEdge3-R-D')
            self._add_myperm2('ME3[DR>UL>UF]', self.invert_moves(self.myperms2['ME3[DR>UF>UL]']), legacy = 'MidEdge3-R-E')
            self._add_myperm2('ME3[DR>UL>FU]', self.invert_moves(self.myperms2['ME3[DR>FU>UL]']), legacy = 'MidEdge3-R-F')
            self._add_myperm2('ME3[DR>LU>FU]', self.invert_moves(self.myperms2['ME3[DR>FU>LU]']), legacy = 'MidEdge3-R-G')
            self._add_myperm2('ME3[DR>LU>UF]', self.invert_moves(self.myperms2['ME3[DR>UF>LU]']), legacy = 'MidEdge3-R-H')


            self._add_myperm2('ME3[RF>FU>LU]', (' U ', " L'", ' E2', ' L ', " U'", " L'", ' E2', ' L '), legacy = 'MidEdge3-N-A')
            self._add_myperm2('ME3[RF>UF>LU]', (" R'", ' S2', ' R ', ' F ', " R'", ' S2', ' R ', " F'"), legacy = 'MidEdge3-N-B')
            self._add_myperm2('ME3[RF>UF>UL]', (' R ', ' S ', " R'", ' F ', ' R ', " S'", " R'", " F'"), legacy = 'MidEdge3-N-C')
            self._add_myperm2('ME3[RF>FU>UL]', (' E ', " L'", ' B ', " M'", ' B2', ' M ', ' B ', ' L ', " E'"), legacy = 'MidEdge3-N-D')

            
            self._add_myperm2('ME3[BR>UF>UL]', (' U ', ' L ', ' E2', " L'", " U'", ' L ', ' E2', " L'"), legacy = 'MidEdge3-Q-A')
            self._add_myperm2('ME3[BR>FU>UL]', (" L'", ' B ', " M'", ' B2', ' M ', ' B ', ' L '), legacy = 'MidEdge3-Q-B')
            self._add_myperm2('ME3[BR>FU>LU]', (" F'", ' E2', ' F ', " U'", " F'", ' E2', ' F ', ' U '), legacy = 'MidEdge3-Q-C')
            self._add_myperm2('ME3[BR>UF>LU]', (" S'", ' R ', " D'", ' M ', ' D2', " M'", " D'", " R'", ' S '), legacy = 'MidEdge3-Q-D')


            self._add_myperm2('ME3[RF>FU>UR]', (" S2"," L'"," E "," R "," U'"," R'"," E'"," R "," U "," R'"," L "," S2"), legacy = 'MidEdge3-Y-A')
            self._add_myperm2('ME3[RF>UF>RU]', (' E2', ' R ', " B'", " M'", ' B2', ' M ', " B'", " R'", ' E2'), legacy = 'MidEdge3-Y-B')
            self._add_myperm2('ME3[BR>DF>LU]', (" R'", ' S ', ' D ', ' R2', " D'", " S'", ' D ', ' R2', " D'", ' R '), legacy = 'MidEdge3-O-A')
            self._add_myperm2('ME3[DR>FL>BU]', (' B ', " L'", ' S ', ' L2', " S'", " L'", " B'"), legacy = 'MidEdge3-O-B')


            self._add_myperm2('ME4s[UB<>UL;UF<>UR]', (" M2"," U'"," F2"," M2"," F2"," M2"," U "," M2"), legacy = 'MidEdge4-Z-A')
            self._add_myperm2('ME4[UB>UL;UF>UR;UL>BU;UR>FU]', (' M2', ' U ', ' M ', ' U2', " M'", ' U ', " M'", " U'", ' F2', ' U ', ' M ', " U'", ' F2', ' U ', ' M2'), legacy = 'MidEdge4-Z-B')
            self._add_myperm2('ME4[UB>LU;UF>UR;UL>UB;UR>FU]', (' M2', " U'", " M'", ' F2', ' U2', ' M ', " U'", " M'", " U'", ' F2', ' U ', " M'"), legacy = 'MidEdge4-Z-C')
            self._add_myperm2('ME4[UB>UL;UF>RU]', (' M2', " U'", " M'", ' U2', ' M ', " U'", " M'", " U'", " M'", ' U2', ' M ', " U'", " M'"), legacy = 'MidEdge4-Z-D')
            self._add_myperm2('ME4[UB>LU;UF>RU]', (" S'", " U'", ' S ', ' U2', " S'", " U'", ' S2', ' U ', " S'", ' U2', ' S ', ' U ', " S'"), legacy = 'MidEdge4-Z-E')


            self._add_myperm2('ME4s[DL<>UB;DR<>UF]', (" D'"," F2"," M2"," F2"," M2"," D "), legacy = 'MidEdge4-S-A')
            self._add_myperm2('ME4[DL>BU;DR>FU;UB>DL;UF>DR]', (' D ', ' M ', ' D2', " M'", ' D ', " M'", " D'", ' B2', ' D ', ' M ', " D'", ' B2', ' D '), legacy = 'MidEdge4-S-B')
            self._add_myperm2('ME4[DL>UB;DR>FU;UB>LD;UF>DR]', (' U ', ' S ', ' U2', " S'", ' U ', " M'", ' D ', " M'", ' D2', ' M ', ' D ', ' M '), legacy = 'MidEdge4-S-C')
            self._add_myperm2('ME4[DL>UB;DR>FU]', (" M'", ' D ', " M'", ' D2', ' M2', ' B2', " M'", ' D ', ' M ', " D'", ' B2', ' D '), legacy = 'MidEdge4-S-D')
            self._add_myperm2('ME4[DL>BU;DR>FU]', (" M'", ' D ', " M'", ' D2', ' M ', ' D ', ' M2', " D'", ' M ', ' D2', " M'", " D'", " M'"), legacy = 'MidEdge4-S-E')


            self._add_myperm2('ME4s[DB<>DF;UB<>UF]', (" M2"," U2"," M2"," U2"), legacy = 'MidEdge4-F-A')
            self._add_myperm2('ME4[DB>DF;UB>FU]', (' M ', ' D2', ' B ', ' M ', " B'", ' D2', ' B ', " M'", " B'", ' D2', " M'", ' D2'), legacy = 'MidEdge4-F-B')
            self._add_myperm2('ME4[DB>FD;DF>DB;UB>FU;UF>UB]', (' D2', ' M ', " B'", ' M ', ' B ', ' D2', " B'", " M'", ' B ', ' F2', " M'", ' F2'), legacy = 'MidEdge4-F-C')
            self._add_myperm2('ME4[DB>FD;DF>DB;UB>UF;UF>BU]', (" M'", " B'", " M'", ' B ', ' U2', " B'", ' M ', ' B ', ' F2', ' M ', ' F2', ' U2'), legacy = 'MidEdge4-F-D')
            self._add_myperm2('ME4[DB>FD;UB>FU]', (' F2', " D'", ' M ', ' D ', ' F2', " D'", " M'", ' D ', ' U ', " M'", " U'", ' F2', ' U ', ' M ', " U'", ' F2')        , legacy = 'MidEdge4-F-E')

            self._add_myperm2('ME4s[DB<>UF;DF<>UB]', (" F2"," M2"," U2"," M2"," U2"," F2"), legacy = 'MidEdge4-X-A')
            self._add_myperm2('ME4[DB>UF;DF>BU]', (" F'", " M'", ' F ', ' D2', " F'", ' M ', ' F ', ' D2', ' M ', ' D2', " M'", ' D2'), legacy = 'MidEdge4-X-B')
            self._add_myperm2('ME4[DB>FU;DF>UB;UB>FD;UF>DB]', (" M'", ' F2', ' U ', " M'", " U'", ' F2', ' U ', ' M ', " U'", ' B2', ' M ', ' B2'), legacy = 'MidEdge4-X-C')
            self._add_myperm2('ME4[DB>FU;DF>BU]', (' B ', ' M ', " B'", ' D2', ' B ', " M'", " B'", ' F ', " M'", " F'", ' D2', ' F ', ' M ', " F'"), legacy = 'MidEdge4-X-D')
        

            self._add_myperm2('ME4s[DF<>DR;UB<>UL]', (' L2', " D'", " M'", ' D2', ' M ', " D'", ' L2', ' R2', " U'", " M'", ' U2', ' M ', " U'", ' R2'), legacy = 'MidEdge4-B-A')
            self._add_myperm2('ME4[DF>DR;DR>FD;UB>UL;UL>BU]', (" U'", " M'", ' U2', ' M ', " U'", " S'", ' U2', ' S ', ' U2', ' M ', ' U ', ' B2', " U'", ' M ', ' U ', " M'", ' B2', ' U2', ' M ', ' U ', ' M2'), legacy = 'MidEdge4-B-B')
            self._add_myperm2('ME4[DF>DR;DR>FD;UB>LU;UL>UB]', (' M ', " U'", " S'", ' U2', ' S ', " U'", ' B2', " M'", ' B2', ' L2', " D'", " M'", ' D2', ' M ', " D'", ' L2'), legacy = 'MidEdge4-B-C')
            self._add_myperm2('ME4[DF>RD;UB>UL]', (" M'", " U'", " S'", ' U2', ' S ', ' U ', ' M ', ' U2', ' S2', " U'", " S'", ' U2', ' S ', " U'", ' S2'), legacy = 'MidEdge4-B-D')
            self._add_myperm2('ME4[DF>RD;UB>LU]', (" M'", " D'", ' M ', ' D2', " M'", " D'", ' M ', " S'", " D'", " M'", ' D2', ' M ', ' D ', ' S ', ' D2'), legacy = 'MidEdge4-B-E')
            
            self._add_myperm2('ME4s[DF<>UR;DL<>UB]', (' U ', " M'", ' U2', ' M ', ' U ', ' S2', ' D2', ' S2', " D'", " M'", ' D2', ' M ', ' D '), legacy = 'MidEdge4-C-A')
            self._add_myperm2('ME4[DF>UR;DL>BU;UB>DL;UR>FD]', (' L2', ' D ', ' R2', " D'", ' S ', ' D ', ' R2', " D'", " S'", ' L2', " U'", ' S ', ' U2', " S'", " U'", " M'", ' U2', ' M ', ' U2'), legacy = 'MidEdge4-C-B')
            self._add_myperm2('ME4[DF>UR;DL>UB;UB>LD;UR>FD]', (' S ', ' U ', " S'", ' U2', ' S ', ' U ', " S'", " M'", ' U2', ' M ', ' U2', ' R2', " D'", " M'", ' D2', ' M ', " D'", ' R2'), legacy = 'MidEdge4-C-C')
            self._add_myperm2('ME4[DF>RU;DL>UB]', (' M ', " D'", ' S ', ' D2', " S'", " D'", ' F2', " M'", ' U ', ' S ', ' U2', " S'", ' U ', ' F2'), legacy = 'MidEdge4-C-D')
            self._add_myperm2('ME4[DF>RU;DL>BU]', (" S'", " D'", " S'", ' D2', ' S ', " D'", ' U ', " S'", ' U2', ' S ', ' U ', ' S ', ' M2', ' D2', ' M2', ' D2'), legacy = 'MidEdge4-C-E')

            self._add_myperm2('ME4s[DF<>UF;DR<>UR]', (" S'", ' U2', ' S ', ' U2', ' M ', " U'", ' M2', ' U2', ' M2', " U'", " M'"), legacy = 'MidEdge4-A-A')
            self._add_myperm2('ME4[DF>FU;DR>RU;UF>DF;UR>DR]', (' S2', " D'", ' L2', ' D ', ' S ', " D'", ' L2', ' D ', ' S ', " M'", " U'", " S'", ' U2', ' S ', " U'", ' M '), legacy = 'MidEdge4-A-B')
            self._add_myperm2('ME4[DF>FU;DR>UR;UF>DF;UR>RD]', (" D2"," L2"," B2",' S2', " U'", ' S ', ' U2', ' L2', " S'", " U'", ' S ', ' U ', ' L2', " U'", ' S '," B2"," L2"," D2"), legacy = 'MidEdge4-A-C')
            self._add_myperm2('ME4[DF>FU;DR>UR]', (" S'", ' U2', ' S ', ' U2', ' S2', ' U ', ' S2', ' U2', ' S2', ' U ', ' S2', ' B ', " M'", " B'", ' U2', ' B ', ' M ', " B'", ' U2'), legacy = 'MidEdge4-A-D')
            self._add_myperm2('ME4[DF>FU;DR>RU]', (" D2"," L2"," B2"," M'", ' U ', ' B2', " U'", " M'", ' U ', ' M2', ' B2', ' M2', ' B2', " U'", ' M ', ' U ', ' B2', " U'", ' M '," B2"," L2"," D2"), legacy = 'MidEdge4-A-E')

            self._add_myperm2('ME4s[DB<>UF;DL<>UR]', (" L2"," B2"," M2"," U "," M2"," U2"," M2"," U "," M2"," B2"," L2"), legacy = 'MidEdge4-D-A')
            self._add_myperm2('ME4[DB>FU;DL>RU;UF>DB;UR>DL]', (" L2"," B2",' S2', " U'", ' S ', ' U2', ' L2', " S'", " U'", ' S ', ' U ', ' L2', " U'", ' S '," B2"," L2", " S'", ' R ', " S'", " R'", ' D2', ' R ', ' S ', ' R ', ' S ', ' R2', ' D2'), legacy = 'MidEdge4-D-B')
            self._add_myperm2('ME4[DB>FU;DL>UR;UF>DB;UR>LD]', (" L2"," B2",' S2', " U'", ' S ', ' U2', ' L2', " S'", " U'", ' S ', ' U ', ' L2', " U'", ' S '," B2"," L2"), legacy = 'MidEdge4-D-C')
            self._add_myperm2('ME4[DB>FU;DL>UR]', (" L2"," B2",' M2', " U'", ' M ', ' U2', " M'", " U'", ' M ', ' U ', ' B2', " U'", " M'", ' U ', ' B2', " U'", ' M2'," B2"," L2"), legacy = 'MidEdge4-D-D')
            self._add_myperm2('ME4[DB>FU;DL>RU]', (" L2"," B2"," M'", ' U ', ' B2', " U'", " M'", ' U ', ' M2', ' B2', ' M2', ' B2', " U'", ' M ', ' U ', ' B2', " U'", ' M '," B2"," L2"), legacy = 'MidEdge4-D-E')

            

            self._add_myperm2('ME2[UB>BU;UF>FU]~v01', (' U2', " M'", ' U ', " M'", ' U ', ' F2', " U'", " M ", ' U ', " M ", ' F2'), legacy = 'MidEdgeFlip-A2')
            self._add_myperm2('ME2[UF>FU;UR>RU]~v01', (' M ', ' U ', " M'", ' U2', ' B2', ' M ', ' U ', ' M ', " U'", ' B2', ' U ', ' M2'), legacy = 'MidEdgeFlip-B2')
            self._add_myperm2('ME2[BR>RB;DF>FD]~v01', (' E ', ' F ', ' E ', ' F2', ' R2', " E'", ' F ', " E'", " F'", ' R2', ' F '), legacy = 'MidEdgeFlip-C2')
            self._add_myperm2('ME2[BR>RB;FL>LF]~v01', (' E ', ' R ', ' E ', " R'", ' F2', ' R ', " E'", ' R ', " E'", ' R2', ' F2'), legacy = 'MidEdgeFlip-D2')
            self._add_myperm2('ME4[DB>BD;DF>FD;DL>LD;DR>RD]~v01', (' M ', ' D ', " M'", ' D2', ' M ', ' D ', " M'", ' S2', ' D ', ' S ', ' D2', " S'", ' L2', ' D ', " S'", " D'", ' L2', ' D ', " S'"), legacy = 'MidEdgeFlip-E4')
            self._add_myperm2('ME4[DB>BD;DF>FD;UB>BU;UF>FU]~v01', (' F2', " M'", " F'", " M'", " F'", ' D2', ' F ', ' M ', " F'", ' M2', " B'", ' M ', ' B ', ' D2', " B'", " M'", " B'", " M'", ' B2'), legacy = 'MidEdgeFlip-F4')
            self._add_myperm2('ME4[DB>BD;DF>FD;UL>LU;UR>RU]~v01', (" M'", " U'", " M'", ' U2', ' M ', " U'", ' M2', ' U ', ' B2', " U'", " M'", ' U ', ' B2', ' M ', ' D2', " M'", ' D2', " U'"), legacy = 'MidEdgeFlip-G4')


            self._add_myperm2('ME2[UB>BU;UF>FU]~v02', self.invert_moves(self.myperms2['ME2[UB>BU;UF>FU]~v01']), legacy = 'MidEdgeFlip-A2I')
            self._add_myperm2('ME2[UF>FU;UR>RU]~v02', self.invert_moves(self.myperms2['ME2[UF>FU;UR>RU]~v01']), legacy = 'MidEdgeFlip-B2I')
            self._add_myperm2('ME2[BR>RB;DF>FD]~v02', self.invert_moves(self.myperms2['ME2[BR>RB;DF>FD]~v01']), legacy = 'MidEdgeFlip-C2I')
            self._add_myperm2('ME2[BR>RB;FL>LF]~v02', self.invert_moves(self.myperms2['ME2[BR>RB;FL>LF]~v01']), legacy = 'MidEdgeFlip-D2I')
            self._add_myperm2('ME4[DB>BD;DF>FD;DL>LD;DR>RD]~v02', self.invert_moves(self.myperms2['ME4[DB>BD;DF>FD;DL>LD;DR>RD]~v01']), legacy = 'MidEdgeFlip-E4I')
            self._add_myperm2('ME4[DB>BD;DF>FD;UB>BU;UF>FU]~v02', self.invert_moves(self.myperms2['ME4[DB>BD;DF>FD;UB>BU;UF>FU]~v01']), legacy = 'MidEdgeFlip-F4I')
            self._add_myperm2('ME4[DB>BD;DF>FD;UL>LU;UR>RU]~v02', self.invert_moves(self.myperms2['ME4[DB>BD;DF>FD;UL>LU;UR>RU]~v01']), legacy = 'MidEdgeFlip-G4I')
            








            

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
            






            
        

            self._add_myperm2('W2-3[DB@R>UF@R>UB@R]~v01', (" U2", ' B2', ' U ', "2B'", " U'", ' B2', ' U ', '2B ', " U "), legacy = 'Wing3-Parallel-I00')
            self._add_myperm2('W2-3[DB@R>UB@R>UF@R]~v01', self.invert_moves(self.myperms2['W2-3[DB@R>UF@R>UB@R]~v01']), legacy = 'Wing3-Parallel-I01')
            self._add_myperm2('W2-3[DB@R>UF@R>UB@R]~v02', (" U2", ' B2', " U'", "2F'", ' U ', ' B2', " U'", '2F ', " U'"), legacy = 'Wing3-Parallel-I02')
            self._add_myperm2('W2-3[DB@R>UB@R>UF@R]~v02', self.invert_moves(self.myperms2['W2-3[DB@R>UF@R>UB@R]~v02']), legacy = 'Wing3-Parallel-I03')

            

            

            self._add_myperm2('W2-3[DB@R>UB@L>UF@L]~v01', (' B2', ' U ', "2B'", " U'", ' B2', ' U ', '2B ', " U'"), legacy = 'Wing3-Parallel-J00')
            self._add_myperm2('W2-3[DB@R>UF@L>UB@L]~v01', self.invert_moves(self.myperms2['W2-3[DB@R>UB@L>UF@L]~v01']), legacy = 'Wing3-Parallel-J01')
            self._add_myperm2('W2-3[DB@R>UB@L>UF@L]~v02', (' B2', " U'", "2F'", ' U ', ' B2', " U'", '2F ', ' U '), legacy = 'Wing3-Parallel-J02')
            self._add_myperm2('W2-3[DB@R>UF@L>UB@L]~v02', self.invert_moves(self.myperms2['W2-3[DB@R>UB@L>UF@L]~v02']), legacy = 'Wing3-Parallel-J03')
            self._add_myperm2('W2-3[DB@R>UB@L>UF@L]~v03', (" B'", '2D ', " F'", "2D'", ' B2', '2D ', ' F ', "2D'", " B'"), legacy = 'Wing3-Parallel-J04')
            self._add_myperm2('W2-3[DB@R>UF@L>UB@L]~v03', self.invert_moves(self.myperms2['W2-3[DB@R>UB@L>UF@L]~v03']), legacy = 'Wing3-Parallel-J05')
            self._add_myperm2('W2-3[DB@R>UB@L>UF@L]~v04', (' B ', '2U ', ' F ', "2U'", ' B2', '2U ', " F'", "2U'", ' B '), legacy = 'Wing3-Parallel-J06')
            self._add_myperm2('W2-3[DB@R>UF@L>UB@L]~v04', self.invert_moves(self.myperms2['W2-3[DB@R>UB@L>UF@L]~v04']), legacy = 'Wing3-Parallel-J07')
 
            


            self._add_myperm2('W2-3[DL@B>UL@F>UR@B]~v01', ('2B2', ' R2', '2B ', ' R2', "2F'", ' L2', '2B2', ' L2', ' U2', '2F ', ' U2', '2B '), legacy = 'Wing3-Parallel-K00')
            self._add_myperm2('W2-3[DL@B>UL@F>UR@B]~v02', ('2F ', ' U2', "2F'", ' L2', "2F'", ' L2', '2F ', ' L2', '2B ', ' L2', "2B'", ' U2'), legacy = 'Wing3-Parallel-K01')


            self._add_myperm2('W2-3[RF@U>UF@R>UB@L]~v01', (' U2', " B'", "2U'", ' B ', ' U2', " B'", '2U ', ' B '), legacy = 'Wing3-Parallel2Plus1-B00')
            self._add_myperm2('W2-3[RF@U>UB@L>UF@R]~v01', self.invert_moves(self.myperms2['W2-3[RF@U>UF@R>UB@L]~v01'])            , legacy = 'Wing3-Parallel2Plus1-B01')
            self._add_myperm2('W2-3[FL@D>UF@R>UB@L]~v01', (' U2', " B ", "2D'", " B'", ' U2', " B ", "2D ", " B'"), legacy = 'Wing3-Parallel2Plus1-B02')
            self._add_myperm2('W2-3[FL@D>UB@L>UF@R]~v01', self.invert_moves(self.myperms2['W2-3[FL@D>UF@R>UB@L]~v01'])  , legacy = 'Wing3-Parallel2Plus1-B03')
            self._add_myperm2('W2-3[RF@D>UF@R>UB@L]~v01', (' U2', ' B ', '2D2', " B'", ' U2', ' B ', '2D2', " B'"), legacy = 'Wing3-Parallel2Plus1-B04')
            self._add_myperm2('W2-3[RF@D>UB@L>UF@R]~v01', self.invert_moves(self.myperms2['W2-3[RF@D>UF@R>UB@L]~v01']), legacy = 'Wing3-Parallel2Plus1-B05')
            self._add_myperm2('W2-3[FL@U>UF@R>UB@L]~v01', (' U2', " B'", '2U2', " B ", ' U2', " B'", '2U2', " B "), legacy = 'Wing3-Parallel2Plus1-B06')
            self._add_myperm2('W2-3[FL@U>UB@L>UF@R]~v01', self.invert_moves(self.myperms2['W2-3[FL@U>UF@R>UB@L]~v01']), legacy = 'Wing3-Parallel2Plus1-B07')

            self._add_myperm2('W2-3[RF@U>UF@R>UB@L]~v02', (" U ", " L'", '2U2', ' L ', ' U2', " L'", '2U2', ' L ', " U "), legacy = 'Wing3-Parallel2Plus1-B00B')
            self._add_myperm2('W2-3[RF@U>UB@L>UF@R]~v02', (" U'", " L'", '2U2', ' L ', ' U2', " L'", '2U2', ' L ', " U'"), legacy = 'Wing3-Parallel2Plus1-B01B')
            self._add_myperm2('W2-3[FL@D>UF@R>UB@L]~v02', (" U'", ' R ', '2D2', " R'", ' U2', ' R ', '2D2', " R'", " U'"), legacy = 'Wing3-Parallel2Plus1-B02B')
            self._add_myperm2('W2-3[FL@D>UB@L>UF@R]~v02', (" U ", ' R ', '2D2', " R'", ' U2', ' R ', '2D2', " R'", " U "), legacy = 'Wing3-Parallel2Plus1-B03B')
            self._add_myperm2('W2-3[RF@D>UF@R>UB@L]~v02', (" U ", ' L ', "2D'", " L'", ' U2', ' L ', '2D ', " L'", " U "), legacy = 'Wing3-Parallel2Plus1-B04B')
            self._add_myperm2('W2-3[RF@D>UB@L>UF@R]~v02', (" U'", ' L ', "2D'", " L'", ' U2', ' L ', '2D ', " L'", " U'"), legacy = 'Wing3-Parallel2Plus1-B05B')
            self._add_myperm2('W2-3[FL@U>UF@R>UB@L]~v02', (" U'", " R'", "2U'", ' R ', ' U2', " R'", '2U ', ' R ', " U'"), legacy = 'Wing3-Parallel2Plus1-B06B')
            self._add_myperm2('W2-3[FL@U>UB@L>UF@R]~v02', (" U ", " R'", "2U'", ' R ', ' U2', " R'", '2U ', ' R ', " U "), legacy = 'Wing3-Parallel2Plus1-B07B')



            self._add_myperm2('W2-3[UB@L>UF@L>UR@F]', (" B'", ' R ', ' B ', "2L'", " B'", " R'", ' B ', '2L '), legacy = 'Wing3-U00')
            self._add_myperm2('W2-3[UB@L>UR@F>UF@L]', self.invert_moves(self.myperms2['W2-3[UB@L>UF@L>UR@F]']), legacy = 'Wing3-U01')
            self._add_myperm2('W2-3[UB@L>UF@L>UL@B]', (' B ', " L'", " B'", "2L'", ' B ', ' L ', " B'", '2L '), legacy = 'Wing3-U02')
            self._add_myperm2('W2-3[UB@L>UL@B>UF@L]', self.invert_moves(self.myperms2['W2-3[UB@L>UF@L>UL@B]']), legacy = 'Wing3-U03')


            self._add_myperm2('W2-3[DR@F>UF@L>UB@L]', (' F ', ' R ', " F'", '2L ', ' F ', " R'", " F'", "2L'"), legacy = 'Wing3-V00')
            self._add_myperm2('W2-3[DR@F>UB@L>UF@L]', self.invert_moves(self.myperms2['W2-3[DR@F>UF@L>UB@L]']), legacy = 'Wing3-V01')
            self._add_myperm2('W2-3[DL@B>UF@L>UB@L]', (" F'", " L'", ' F ', '2L ', " F'", ' L ', ' F ', "2L'"), legacy = 'Wing3-V02')
            self._add_myperm2('W2-3[DL@B>UB@L>UF@L]', self.invert_moves(self.myperms2['W2-3[DL@B>UF@L>UB@L]']), legacy = 'Wing3-V03')
            
            self._add_myperm2('W2-3[UB@L>UF@R>UL@F]~v01', (" L'", ' U2', " F'", "2U'", ' F ', ' U2', " F'", '2U ', ' F ', ' L '), legacy = 'Wing3-U04')
            self._add_myperm2('W2-3[UB@L>UL@F>UF@R]~v01', (' L ', ' U2', ' B ', "2D'", " B'", ' U2', ' B ', '2D ', " B'", " L'"), legacy = 'Wing3-U05')
            self._add_myperm2('W2-3[UB@L>UL@B>UF@R]~v01', (' L ', ' U2', " B'", '2U2', ' B ', ' U2', " B'", '2U2', ' B ', " L'"), legacy = 'Wing3-U06')
            self._add_myperm2('W2-3[UB@L>UF@R>UL@B]~v01', (" L'", ' U2', ' F ', '2D2', " F'", ' U2', ' F ', '2D2', " F'", ' L '), legacy = 'Wing3-U07')
            
            self._add_myperm2('W2-3[UB@L>UF@R>UL@F]~v02', (' L ', ' B ', "2D'", " B'", ' U2', ' B ', '2D ', " B'", ' U2', " L'"), legacy = 'Wing3-U04B')
            self._add_myperm2('W2-3[UB@L>UL@F>UF@R]~v02', (" L'", " F'", "2U'", ' F ', ' U2', " F'", '2U ', ' F ', ' U2', ' L '), legacy = 'Wing3-U05B')
            self._add_myperm2('W2-3[UB@L>UL@B>UF@R]~v02', (" L'", ' F ', '2D2', " F'", ' U2', ' F ', '2D2', " F'", ' U2', ' L '), legacy = 'Wing3-U06B')
            self._add_myperm2('W2-3[UB@L>UF@R>UL@B]~v02', (' L ', " B'", '2U2', ' B ', ' U2', " B'", '2U2', ' B ', ' U2', " L'"), legacy = 'Wing3-U07B')

            self._add_myperm2('W2-3[DL@B>UB@L>UF@R]~v01', (' L ', ' U2', " F'", "2U'", ' F ', ' U2', " F'", '2U ', ' F ', " L'"), legacy = 'Wing3-V04')
            self._add_myperm2('W2-3[DL@B>UF@R>UB@L]~v01', (" L'", ' U2', ' B ', "2D'", " B'", ' U2', ' B ', '2D ', " B'", ' L '), legacy = 'Wing3-V05')
            self._add_myperm2('W2-3[DL@F>UF@R>UB@L]~v01', (" L'", ' U2', " B'", '2U2', ' B ', ' U2', " B'", '2U2', ' B ', ' L '), legacy = 'Wing3-V06')
            self._add_myperm2('W2-3[DL@F>UB@L>UF@R]~v01', (' L ', ' U2', ' F ', '2D2', " F'", ' U2', ' F ', '2D2', " F'", " L'"), legacy = 'Wing3-V07')

            self._add_myperm2('W2-3[DL@B>UB@L>UF@R]~v02', (" L'", ' B ', "2D'", " B'", ' U2', ' B ', '2D ', " B'", ' U2', ' L '), legacy = 'Wing3-V04B')
            self._add_myperm2('W2-3[DL@B>UF@R>UB@L]~v02', (' L ', " F'", "2U'", ' F ', ' U2', " F'", '2U ', ' F ', ' U2', " L'"), legacy = 'Wing3-V05B')
            self._add_myperm2('W2-3[DL@F>UF@R>UB@L]~v02', (' L ', ' F ', '2D2', " F'", ' U2', ' F ', '2D2', " F'", ' U2', " L'"), legacy = 'Wing3-V06B')
            self._add_myperm2('W2-3[DL@F>UB@L>UF@R]~v02', (" L'", " B'", '2U2', ' B ', ' U2', " B'", '2U2', ' B ', ' U2', ' L '), legacy = 'Wing3-V07B')


            self._add_myperm2('W2-3[DF@L>FL@U>UB@L]~v01', (' B ', ' L2', " B'", '2L2', ' B ', ' L2', " B'", '2L2'), legacy = 'Wing3-Parallel2Plus1-I00')
            self._add_myperm2('W2-3[DF@L>UB@L>FL@U]~v01', ('2L2', ' B ', ' L2', " B'", '2L2', ' B ', ' L2', " B'"), legacy = 'Wing3-Parallel2Plus1-I01')
            self._add_myperm2('W2-3[DF@L>RF@D>UB@L]~v01', (" B'", ' R2', ' B ', '2L2', " B'", ' R2', ' B ', '2L2'), legacy = 'Wing3-Parallel2Plus1-I02')
            self._add_myperm2('W2-3[DF@L>UB@L>RF@D]~v01', ('2L2', " B'", ' R2', ' B ', '2L2', " B'", ' R2', ' B '), legacy = 'Wing3-Parallel2Plus1-I03')
            self._add_myperm2('W2-3[DF@L>FL@D>UB@L]~v01', (" U'", " L'", ' U ', '2L2', " U'", ' L ', ' U ', '2L2'), legacy = 'Wing3-Parallel2Plus1-I04')
            self._add_myperm2('W2-3[DF@L>UB@L>FL@D]~v01', ('2L2', " U'", " L'", ' U ', '2L2', " U'", ' L ', ' U '), legacy = 'Wing3-Parallel2Plus1-I05')
            self._add_myperm2('W2-3[DF@L>RF@U>UB@L]~v01', (' U ', ' R ', " U'", '2L2', ' U ', " R'", " U'", '2L2'), legacy = 'Wing3-Parallel2Plus1-I06')
            self._add_myperm2('W2-3[DF@L>UB@L>RF@U]~v01', ('2L2', ' U ', ' R ', " U'", '2L2', ' U ', " R'", " U'"), legacy = 'Wing3-Parallel2Plus1-I07')

            self._add_myperm2('W2-3[DF@L>FL@D>UB@L]~v02', ('2L2', " D'", ' L ', ' D ', '2L2', " D'", " L'", ' D '), legacy = 'Wing3-Parallel2Plus1-I04B')
            self._add_myperm2('W2-3[DF@L>UB@L>FL@D]~v02', (" D'", ' L ', ' D ', '2L2', " D'", " L'", ' D ', '2L2'), legacy = 'Wing3-Parallel2Plus1-I05B')
            self._add_myperm2('W2-3[DF@L>RF@U>UB@L]~v02', ('2L2', ' D ', " R'", " D'", '2L2', ' D ', ' R ', " D'"), legacy = 'Wing3-Parallel2Plus1-I06B')
            self._add_myperm2('W2-3[DF@L>UB@L>RF@U]~v02', (' D ', " R'", " D'", '2L2', ' D ', ' R ', " D'", '2L2'), legacy = 'Wing3-Parallel2Plus1-I07B')

            self._add_myperm2('W2-3[DF@L>FL@U>UB@L]~v02', (" B'", '2U2', " B'", ' D2', ' B ', '2U2', " B'", ' D2', ' B2'), legacy = 'Wing3-Parallel2Plus1-I00C')
            self._add_myperm2('W2-3[DF@L>UB@L>FL@U]~v02', (' B2', ' D2', ' B ', "2U2", " B'", ' D2', ' B ', '2U2', ' B '), legacy = 'Wing3-Parallel2Plus1-I01C')
            self._add_myperm2('W2-3[DF@L>RF@D>UB@L]~v02', (' B ', '2D2', ' B ', ' D2', " B'", '2D2', ' B ', ' D2', ' B2'), legacy = 'Wing3-Parallel2Plus1-I02C')
            self._add_myperm2('W2-3[DF@L>UB@L>RF@D]~v02', (' B2', ' D2', " B'", "2D2", ' B ', ' D2', " B'", '2D2', " B'"), legacy = 'Wing3-Parallel2Plus1-I03C')
            self._add_myperm2('W2-3[DF@L>FL@D>UB@L]~v03', (' B ', "2D'", ' B ', ' D2', " B'", '2D ', ' B ', ' D2', ' B2'), legacy = 'Wing3-Parallel2Plus1-I04C')
            self._add_myperm2('W2-3[DF@L>UB@L>FL@D]~v03', (' B2', ' D2', " B'", "2D'", ' B ', ' D2', " B'", '2D ', " B'"), legacy = 'Wing3-Parallel2Plus1-I05C')
            self._add_myperm2('W2-3[DF@L>RF@U>UB@L]~v03', (" B'", "2U'", " B'", ' D2', ' B ', '2U ', " B'", ' D2', ' B2'), legacy = 'Wing3-Parallel2Plus1-I06C')
            self._add_myperm2('W2-3[DF@L>UB@L>RF@U]~v03', (' B2', ' D2', ' B ', "2U'", " B'", ' D2', ' B ', '2U ', ' B '), legacy = 'Wing3-Parallel2Plus1-I07C')



            self._add_myperm2('W2-3[DR@B>UB@L>FL@U]', (" F'", '2L2', " F'", ' R ', ' F ', '2L2', " F'", " R'", ' F2'), legacy = 'Wing3-O00')
            self._add_myperm2('W2-3[DR@F>UB@R>FL@D]', (" F'", '2R2', " F'", ' R ', ' F ', '2R2', " F'", " R'", ' F2'), legacy = 'Wing3-O01')
            self._add_myperm2('W2-3[DR@B>FL@D>UB@L]', (' D ', "2L'", " B'", ' L2', ' B ', '2L ', " B'", ' L2', ' B ', " D'"), legacy = 'Wing3-O02')
            self._add_myperm2('W2-3[DR@F>FL@U>UB@R]', (' D ', '2R ', " B'", ' L2', ' B ', "2R'", " B'", ' L2', ' B ', " D'"), legacy = 'Wing3-O03')

            self._add_myperm2('W2-3[RF@U>UR@B>UF@L]~v01', (" R'", " D'", '2L ', ' D ', " R'", " D'", "2L'", ' D ', ' R2'), legacy = 'Wing3-Y00')
            self._add_myperm2('W2-3[RF@D>UR@F>UF@R]~v01', (" R'", " D'", "2R'", ' D ', " R'", " D'", '2R ', ' D ', ' R2'), legacy = 'Wing3-Y01')
            self._add_myperm2('W2-3[RF@U>UR@B>UF@L]~v02', (" F'", ' L ', '2B2', " L'", " F'", ' L ', '2B2', " L'", ' F2'), legacy = 'Wing3-Y00B')
            self._add_myperm2('W2-3[RF@D>UR@F>UF@R]~v02', (" F'", ' L ', '2F2', " L'", " F'", ' L ', '2F2', " L'", ' F2'), legacy = 'Wing3-Y01B')
            self._add_myperm2('W2-3[RF@U>UR@B>UF@L]~v03', (' R2', " B'", '2L2', ' B ', " R'", " B'", '2L2', ' B ', " R'"), legacy = 'Wing3-Y00C')
            self._add_myperm2('W2-3[RF@D>UR@F>UF@R]~v03', (' R2', " B'", '2R2', ' B ', " R'", " B'", '2R2', ' B ', " R'"), legacy = 'Wing3-Y01C')
            self._add_myperm2('W2-3[RF@U>UF@R>UR@F]', (' R ', ' U ', " F'", '2U2', ' F ', " U'", " F'", '2U2', ' F ', " R'"), legacy = 'Wing3-Y02')
            self._add_myperm2('W2-3[RF@D>UF@L>UR@B]', (' R ', ' U ', " F'", '2D2', ' F ', " U'", " F'", '2D2', ' F ', " R'"), legacy = 'Wing3-Y03')


            self._add_myperm2('W2-3[RF@U>UF@L>UL@B]~v01', (' U ', ' L ', '2U ', " L'", " U'", ' L ', "2U'", " L'"), legacy = 'Wing3-N00')
            self._add_myperm2('W2-3[RF@U>UL@B>UF@L]~v01', (" L ", "2U ", " L'", " U ", " L ", "2U'", " L'", " U'"), legacy = 'Wing3-N01')
            self._add_myperm2('W2-3[RF@U>UF@L>UL@B]~v02', (' R ', "2B'", " R'", ' F ', ' R ', '2B ', " R'", " F'"), legacy = 'Wing3-N00B')
            self._add_myperm2('W2-3[RF@U>UL@B>UF@L]~v02', (' F ', ' R ', "2B'", " R'", " F'", ' R ', '2B ', " R'"), legacy = 'Wing3-N01B')
            self._add_myperm2('W2-3[RF@D>UL@B>UF@L]', (" L'", '2D2', ' L ', ' U ', " L'", '2D2', ' L ', " U'"), legacy = 'Wing3-N02')
            self._add_myperm2('W2-3[RF@D>UF@L>UL@B]', (' U ', " L'", '2D2', ' L ', " U'", " L'", '2D2', ' L '), legacy = 'Wing3-N03')
            self._add_myperm2('W2-3[RF@U>UL@F>UF@R]', (" L'", '2U2', ' L ', ' U ', " L'", '2U2', ' L ', " U'"), legacy = 'Wing3-N04')
            self._add_myperm2('W2-3[RF@U>UF@R>UL@F]', (' U ', " L'", '2U2', ' L ', " U'", " L'", '2U2', ' L '), legacy = 'Wing3-N05')
            self._add_myperm2('W2-3[RF@U>UF@R>UL@B]~v01', (" L'", " B'", ' U2', " B'", "2U'", ' B ', ' U2', " B'", '2U ', ' B2', ' L '), legacy = 'Wing3-N06')
            self._add_myperm2('W2-3[RF@U>UL@B>UF@R]~v01', (" L'", ' B2', "2U'", ' B ', ' U2', " B'", '2U ', ' B ', ' U2', ' B ', ' L '), legacy = 'Wing3-N07')
            self._add_myperm2('W2-3[RF@U>UF@R>UL@B]~v02', (" R'", ' D2', '2B ', ' D ', ' F2', " D'", "2B'", ' D ', ' F2', ' D ', ' R '), legacy = 'Wing3-N06B')
            self._add_myperm2('W2-3[RF@U>UL@B>UF@R]~v02', (" R'", " D'", ' F2', " D'", '2B ', ' D ', ' F2', " D'", "2B'", ' D2', ' R '), legacy = 'Wing3-N07B')




            self._add_myperm2('W2-3[DL@F>RF@U>UF@L]~v01', (" R'", "2F'", ' R ', ' F ', " R'", '2F ', ' R ', " F'"), legacy = 'Wing3-Q00')
            self._add_myperm2('W2-3[DL@F>UF@L>RF@U]~v01', (' F ', " R'", "2F'", ' R ', " F'", " R'", '2F ', ' R '), legacy = 'Wing3-Q01')
            self._add_myperm2('W2-3[DR@B>FL@D>UF@L]~v01', (' L ', "2B'", " L'", " F'", ' L ', '2B ', " L'", ' F '), legacy = 'Wing3-Q02')
            self._add_myperm2('W2-3[DR@B>UF@L>FL@D]~v01', (" F'", ' L ', "2B'", " L'", ' F ', ' L ', '2B ', " L'"), legacy = 'Wing3-Q03')
            self._add_myperm2('W2-3[DL@F>RF@U>UF@L]~v02', (" F'", " U'", '2F2', ' U ', ' F ', " U'", '2F2', ' U '), legacy = 'Wing3-Q00B')
            self._add_myperm2('W2-3[DL@F>UF@L>RF@U]~v02', (" U'", '2F2', ' U ', " F'", " U'", '2F2', ' U ', ' F '), legacy = 'Wing3-Q01B')
            self._add_myperm2('W2-3[DR@B>FL@D>UF@L]~v02', (' F ', ' U ', '2B2', " U'", " F'", ' U ', '2B2', " U'"), legacy = 'Wing3-Q02B')
            self._add_myperm2('W2-3[DR@B>UF@L>FL@D]~v02', (' U ', '2B2', " U'", ' F ', ' U ', '2B2', " U'", " F'"), legacy = 'Wing3-Q03B')
            self._add_myperm2('W2-3[DL@F>UF@L>RF@D]', (' L ', '2D2', ' L ', ' U ', " L'", '2D2', ' L ', " U'", ' L2'), legacy = 'Wing3-Q04')
            self._add_myperm2('W2-3[DR@B>UF@L>FL@U]', (" R'", '2U2', " R'", " U'", ' R ', '2U2', " R'", ' U ', ' R2'), legacy = 'Wing3-Q05')
            self._add_myperm2('W2-3[DL@F>RF@U>UF@R]', (" R'", "2F'", " D'", ' F2', ' D ', '2F ', " D'", ' F2', ' D ', ' R '), legacy = 'Wing3-Q06')
            self._add_myperm2('W2-3[DR@B>FL@D>UF@R]', (' L ', "2B'", ' D ', ' F2', " D'", '2B ', ' D ', ' F2', " D'", " L'"), legacy = 'Wing3-Q07')

            self._add_myperm2('W2-3[RF@U>UF@L>UB@L]', ("2U'", " B'", '2U ', ' F ', "2U'", ' B ', '2U ', " F'"), legacy = 'Wing3-Parallel2Plus1-A00')
            self._add_myperm2('W2-3[FL@D>UF@L>UB@L]', ("2D'", ' B ', '2D ', " F'", "2D'", " B'", '2D ', ' F '), legacy = 'Wing3-Parallel2Plus1-A01')
            self._add_myperm2('W2-3[LB@U>UF@L>UB@L]', ("2U ", " B'", '2U ', ' F ', "2U'", ' B ', '2U ', " F'", "2U2"), legacy = 'Wing3-Parallel2Plus1-A02')
            self._add_myperm2('W2-3[BR@D>UF@L>UB@L]', ("2D ", ' B ', '2D ', " F'", "2D'", " B'", '2D ', ' F ', "2D2"), legacy = 'Wing3-Parallel2Plus1-A03')
            self._add_myperm2('W2-3[RF@U>UB@L>UF@L]', self.invert_moves(self.myperms2['W2-3[RF@U>UF@L>UB@L]']), legacy = 'Wing3-Parallel2Plus1-A04')
            self._add_myperm2('W2-3[FL@D>UB@L>UF@L]', self.invert_moves(self.myperms2['W2-3[FL@D>UF@L>UB@L]']), legacy = 'Wing3-Parallel2Plus1-A05')
            self._add_myperm2('W2-3[LB@U>UB@L>UF@L]', self.invert_moves(self.myperms2['W2-3[LB@U>UF@L>UB@L]']), legacy = 'Wing3-Parallel2Plus1-A06')
            self._add_myperm2('W2-3[BR@D>UB@L>UF@L]', self.invert_moves(self.myperms2['W2-3[BR@D>UF@L>UB@L]']), legacy = 'Wing3-Parallel2Plus1-A07')
            self._add_myperm2('W2-3[DB@L>FL@U>UF@R]', ('2U ', " B'", "2U'", " F'", '2U ', ' B ', "2U'", ' F '), legacy = 'Wing3-Parallel2Plus1-J00')
            self._add_myperm2('W2-3[DB@L>RF@D>UF@R]', ('2D ', ' B ', "2D'", ' F ', '2D ', " B'", "2D'", " F'"), legacy = 'Wing3-Parallel2Plus1-J01')
            self._add_myperm2('W2-3[BR@U>UF@R>DB@L]', ("2U'", " B'", "2U'", " F'", '2U ', ' B ', "2U'", ' F ', "2U2"), legacy = 'Wing3-Parallel2Plus1-J02')
            self._add_myperm2('W2-3[DB@L>LB@D>UF@R]', ("2D'", ' B ', "2D'", ' F ', '2D ', " B'", "2D'", " F'", "2D2"), legacy = 'Wing3-Parallel2Plus1-J03')
            self._add_myperm2('W2-3[DB@L>UF@R>FL@U]', self.invert_moves(self.myperms2['W2-3[DB@L>FL@U>UF@R]']), legacy = 'Wing3-Parallel2Plus1-J04')
            self._add_myperm2('W2-3[DB@L>UF@R>RF@D]', self.invert_moves(self.myperms2['W2-3[DB@L>RF@D>UF@R]']), legacy = 'Wing3-Parallel2Plus1-J05')
            self._add_myperm2('W2-3[BR@U>DB@L>UF@R]', self.invert_moves(self.myperms2['W2-3[BR@U>UF@R>DB@L]']), legacy = 'Wing3-Parallel2Plus1-J06')
            self._add_myperm2('W2-3[DB@L>UF@R>LB@D]', self.invert_moves(self.myperms2['W2-3[DB@L>LB@D>UF@R]']), legacy = 'Wing3-Parallel2Plus1-J07')


            self._add_myperm2('W2-3[RF@U>UF@L>UF@R]', ('2R2', "2U'", ' B ', '2U ', ' F ', "2U'", " B'", '2U ', " F'", '2R2'), legacy = 'Wing3-SameEdgePairPlus1-K00')
            self._add_myperm2('W2-3[FL@D>UF@L>UF@R]', ('2R2', "2D'", " B'", '2D ', " F'", "2D'", ' B ', '2D ', ' F ', '2R2'), legacy = 'Wing3-SameEdgePairPlus1-K01')
            self._add_myperm2('W2-3[RF@U>UF@R>UF@L]', ('2R2', ' F ', "2U'", ' B ', '2U ', " F'", "2U'", " B'", '2U ', '2R2'), legacy = 'Wing3-SameEdgePairPlus1-K02')
            self._add_myperm2('W2-3[FL@D>UF@R>UF@L]', ('2R2', " F'", "2D'", " B'", '2D ', ' F ', "2D'", ' B ', '2D ', '2R2'), legacy = 'Wing3-SameEdgePairPlus1-K03')
            self._add_myperm2('W2-3[FL@D>UB@R>UB@L]', ('2R2', "2D'", ' B ', '2D ', ' F ', "2D'", " B'", '2D ', " F'", '2R2'), legacy = 'Wing3-SameEdgePairPlus1-K04')
            self._add_myperm2('W2-3[RF@U>UB@R>UB@L]', ('2R2', "2U'", " B'", '2U ', " F'", "2U'", ' B ', '2U ', ' F ', '2R2'), legacy = 'Wing3-SameEdgePairPlus1-K05')
            self._add_myperm2('W2-3[FL@D>UB@L>UB@R]', ('2R2', ' F ', "2D'", ' B ', '2D ', " F'", "2D'", " B'", '2D ', '2R2'), legacy = 'Wing3-SameEdgePairPlus1-K06')
            self._add_myperm2('W2-3[RF@U>UB@L>UB@R]', ('2R2', " F'", "2U'", " B'", '2U ', ' F ', "2U'", ' B ', '2U ', '2R2'), legacy = 'Wing3-SameEdgePairPlus1-K07')




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


            self._add_myperm2('W2-2s[UB@L<>UF@L]~v01', ("2R2"," U2","2L'"," U2","2L "," F2","2R'"," F2"," U2","2R'"," U2","2R'"), legacy = 'WingSwapParallel-A0')
            self._add_myperm2('W2-2s[UB@L<>UF@L]~v02', ("2R2"," U2","2R "," B2","2L'"," D2","2R "," D2"," B2","2L "," U2","2R "), legacy = 'WingSwapParallel-A1')
            self._add_myperm2('W2-2s[UB@L<>UF@L]~v03', ("2L2"," D2","2L "," D2","2R'"," D2","2R "," D2"," B2","2R "," B2","2L "), legacy = 'WingSwapParallel-A2')
            self._add_myperm2('W2-2s[UB@L<>UF@L]~v04', ("2L2"," D2","2R "," F2","2R'"," F2","2L "," D2"," B2","2R "," B2","2L "), legacy = 'WingSwapParallel-A3')



            
            #SwapD ('2L2', ' B2', ' U2', '2L ', ' U2', '2L2', ' B2', '2L ', ' U2', '2L2', ' B ', "2D'", " B'", ' U2', ' B ', '2D ', ' B ', '2L ', ' B2')
            #SwapE ('2L2', ' B ', '2D2', " B'", ' U2', ' B ', '2D2', ' B ', '2R ', ' B2', '2R2', ' U2', '2L ', ' F2', '2L2', ' F2', ' U2', '2R ', ' U2')

            self._add_myperm2('W2-2s[UF@L<>UF@R]~v01', ("2L2"," B2","2R'"," F2","2L "," F2"," U2","2L "," U2","2L'"," U2","2R "," U2"," B2","2L2"), legacy = 'WingSwapParallel-K0')
            self._add_myperm2('W2-2s[UF@L<>UF@R]~v02', self.invert_moves(self.myperms2['W2-2s[UF@L<>UF@R]~v01']), legacy = 'WingSwapParallel-K1')
            

    

            swapc = ('2D2', ' B ', '2R ', " B'", ' R2', ' B ', "2R'", ' B ', '2D ', ' B2', '2D2', ' R2', '2U ', ' F2', '2D2', ' F2', ' R2', "2U'", ' R2')
            swapd = ('2U2', ' B ', "2L'", " B'", ' R2', ' B ', '2L ', ' B ', "2U'", ' B2', '2U2', ' R2', "2D'", ' F2', '2U2', ' F2', ' R2', '2D ', ' R2')
            swapex = ('2U2', " B'", '2R2', ' B ', ' R2', " B'", '2R2', " B'", "2U'", ' B2', '2U2', ' R2', "2D'", ' F2', '2U2', ' F2', ' R2', '2D ', ' R2')
            swapey = ('2D2', " B'", '2L2', ' B ', ' R2', " B'", '2L2', " B'", "2D ", ' B2', '2D2', ' R2', "2U ", ' F2', '2D2', ' F2', ' R2', "2U'", ' R2')
            swapfx = ('2U2', " F'", "2R'", ' F ', ' R2', " F'", '2R ', " F'", "2U'", ' F2', '2U2', ' R2', '2U ', ' F2', '2U2', ' F2', ' R2', '2U ', ' R2')
            swapfy = ('2D2', " F'", '2L ', ' F ', ' R2', " F'", "2L'", " F'", '2D ', ' F2', '2D2', ' R2', "2D'", ' F2', '2D2', ' F2', ' R2', "2D'", ' R2')
            swapg = ('2U2', ' F ', '2L2', " F'", ' R2', ' F ', '2L2', ' F ', "2U'", ' F2', '2U2', ' R2', '2U ', ' F2', '2U2', ' F2', ' R2', '2U ', ' R2')
            swaph = ('2D2', ' F ', '2R2', " F'", ' R2', ' F ', '2R2', ' F ', '2D ', ' F2', '2D2', ' R2', "2D'", ' F2', '2D2', ' F2', ' R2', "2D'", ' R2')

            self._add_myperm2('W2-2s[DF@L<>UB@L]~v03', ('2L ', ' F2', "2L'", "2R'", ' F2', '2R ', ' F2', '2R2', ' F2', ' U2', "2R'", ' U2', ' F2', '2R2', ' F2'), legacy = 'WingSwapParallel-IX0')
            self._add_myperm2('W2-2s[DF@L<>UB@L]~v04', ("2R'", ' F2', '2R ', '2L ', ' F2', "2L'", ' F2', '2R2', ' F2', ' U2', "2R'", ' U2', ' F2', '2R2', ' F2'), legacy = 'WingSwapParallel-IX1')
            self._add_myperm2('W2-2s[DF@L<>UB@L]~v05', ('2L ', ' F2', "2L'", "2R'", ' F2', '2R ', ' U2', '2R2', ' U2', ' F2', '2R ', ' F2', ' U2', '2R2', ' U2'), legacy = 'WingSwapParallel-IX2')
            self._add_myperm2('W2-2s[DF@L<>UB@L]~v06', ("2R'", ' F2', '2R ', '2L ', ' F2', "2L'", ' U2', '2R2', ' U2', ' F2', '2R ', ' F2', ' U2', '2R2', ' U2'), legacy = 'WingSwapParallel-IX3')
            self._add_myperm2('W2-2s[DF@L<>UB@L]~v07', ('2L ', ' B2', "2L'", "2R'", ' B2', '2R ', ' B2', '2L2', ' B2', ' D2', "2L'", ' D2', ' B2', '2L2', ' B2'), legacy = 'WingSwapParallel-IX4')
            self._add_myperm2('W2-2s[DF@L<>UB@L]~v08', ("2R'", ' B2', "2R ", "2L ", ' B2', "2L'", ' B2', '2L2', ' B2', ' D2', "2L'", ' D2', ' B2', '2L2', ' B2'), legacy = 'WingSwapParallel-IX5')
            self._add_myperm2('W2-2s[DF@L<>UB@L]~v09', ('2L ', ' B2', "2L'", "2R'", ' B2', '2R ', ' D2', '2L2', ' D2', ' B2', "2L ", ' B2', ' D2', '2L2', ' D2'), legacy = 'WingSwapParallel-IX6')
            self._add_myperm2('W2-2s[DF@L<>UB@L]~v10', ("2R'", ' B2', "2R ", "2L ", ' B2', "2L'", ' D2', '2L2', ' D2', ' B2', "2L ", ' B2', ' D2', '2L2', ' D2'), legacy = 'WingSwapParallel-IX7')

            self._add_myperm2('W2-2s[DF@L<>UB@L]~v11', self.invert_moves((" F2", '2L2', ' B2', ' D2', '2R ', ' D2', ' B2', '2L2', ' F2', '2R ', ' U2', "2R'", "2L'", ' U2', '2L ')), legacy = 'WingSwapParallel-IY0')
            self._add_myperm2('W2-2s[DF@L<>UB@L]~v12', self.invert_moves((" F2", '2L2', ' B2', ' D2', '2R ', ' D2', ' B2', '2L2', ' F2', "2L'", ' U2', '2L ', '2R ', ' U2', "2R'")), legacy = 'WingSwapParallel-IY1')
            self._add_myperm2('W2-2s[DF@L<>UB@L]~v13', self.invert_moves((" F2", '2L2', ' B2', ' D2', '2R ', ' D2', ' B2', '2L2', ' F2', "2R'", ' F2', "2R ", "2L ", ' F2', "2L'")), legacy = 'WingSwapParallel-IY2')
            self._add_myperm2('W2-2s[DF@L<>UB@L]~v14', self.invert_moves((" F2", '2L2', ' B2', ' D2', '2R ', ' D2', ' B2', '2L2', ' F2', '2L ', ' F2', "2L'", "2R'", ' F2', '2R ')), legacy = 'WingSwapParallel-IY3')
            self._add_myperm2('W2-2s[DF@L<>UB@L]~v15', self.invert_moves((" F2", '2R2', ' B2', ' D2', '2L ', ' D2', ' B2', '2R2', ' F2', '2R ', ' U2', "2R'", "2L'", ' U2', '2L ')), legacy = 'WingSwapParallel-IY4')
            self._add_myperm2('W2-2s[DF@L<>UB@L]~v16', self.invert_moves((" F2", '2R2', ' B2', ' D2', '2L ', ' D2', ' B2', '2R2', ' F2', "2L'", ' U2', "2L ", "2R ", ' U2', "2R'")), legacy = 'WingSwapParallel-IY5')
            self._add_myperm2('W2-2s[DF@L<>UB@L]~v17', self.invert_moves((" F2", '2R2', ' B2', ' D2', '2L ', ' D2', ' B2', '2R2', ' F2', "2R'", ' F2', '2R ', '2L ', ' F2', "2L'")), legacy = 'WingSwapParallel-IY6')
            self._add_myperm2('W2-2s[DF@L<>UB@L]~v18', self.invert_moves((" F2", '2R2', ' B2', ' D2', '2L ', ' D2', ' B2', '2R2', ' F2', "2L ", ' F2', "2L'", "2R'", ' F2', "2R ")), legacy = 'WingSwapParallel-IY7')




            self._add_myperm2('W2-2s[DF@L<>UB@L]~v01', ('2L ', ' B2', '2R2', ' U2', '2R ', ' U2', "2R'", "2L'", ' B2', ' D2', "2R'", ' D2', ' B2', '2R2', ' B2'), legacy = 'WingSwapParallel-I0')
            self._add_myperm2('W2-2s[DF@L<>UB@L]~v02', ('2R ', ' D2', '2L2', ' F2', '2L ', ' F2', '2L ', '2R ', ' D2', ' B2', '2R ', ' B2', ' D2', '2R2', ' D2'), legacy = 'WingSwapParallel-I1')

            

            self._add_myperm2('W2-2s[DF@L<>UB@R]~v01', ('2L2', ' D2', ' B2', "2L'", ' B2', '2R ', ' D2', "2R'", ' D2', '2L2', ' B2', '2L ', ' B2', "2L'", ' D2'), legacy = 'WingSwapParallel-J0')
            self._add_myperm2('W2-2s[DF@L<>UB@R]~v02', ('2R2', ' B2', ' D2', "2R'", ' D2', '2R2', "2L'", ' F2', "2L'", ' F2', '2L2', ' D2', '2R ', ' D2', "2R'", ' B2'), legacy = 'WingSwapParallel-J1')


      
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v01', ("2R2", " F2", " U2", "2R ", " U2", "2R2", " F2", "2R ", " U2", "2R2", " U2", " F2", "2R ", " F2"), legacy = 'WingSwapParallel-B0')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v02', ("2L2", " F2", " U2", "2L ", " U2", "2L2", " F2", "2L'", " U2", "2L2", " U2", " F2", "2L'", " F2"), legacy = 'WingSwapParallel-B1')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v03', ('2L2', ' F2', ' D2', '2L ', ' D2', '2L2', ' F2', '2R ', ' U2', '2L2', ' U2', ' F2', "2R'", ' F2'), legacy = 'WingSwapParallel-B2')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v04', ("2R2", " F2", " U2", "2L'", " U2", "2L2", " F2", "2L'", " U2", "2R2", " U2", " F2", "2R ", " F2"), legacy = 'WingSwapParallel-B3')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v05', ('2R2', ' B2', ' D2', "2L'", ' D2', '2L2', ' B2', "2R'", ' U2', '2R2', ' U2', ' B2', "2L'", ' B2'), legacy = 'WingSwapParallel-B4')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v06', ('2L2', ' B2', ' D2', "2R'", ' D2', '2R2', ' B2', "2L ", ' U2', '2L2', ' U2', ' B2', "2R ", ' B2'), legacy = 'WingSwapParallel-B5')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v07', ('2R2', ' F2', ' U2', "2R'", ' U2', '2R2', ' F2', "2L'", '2R2', ' U2', '2L2', ' U2', ' B2', '2R ', ' B2'), legacy = 'WingSwapParallel-B6')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v08', ('2L2', ' B2', ' D2', '2R2', "2L'", ' D2', '2L2', ' B2', "2R'", ' U2', '2L2', ' U2', ' B2', '2R ', ' B2'), legacy = 'WingSwapParallel-B7')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v09', ('2L2', ' F2', ' U2', '2L2', '2R ', ' U2', '2R2', ' F2', '2R ', ' U2', '2L2', ' U2', ' F2', "2L'", ' F2'), legacy = 'WingSwapParallel-B8')


            self._add_myperm2('W2-2s[UB@L<>UF@R]~v10', ('2R2', ' F2', ' U2', '2R ', ' U2', ' F2', '2R2', ' F2', "2R'", ' F2', '2R ', '2L ', ' F2', "2L'", " F2"), legacy = 'WingSwapParallel-BX00')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v11', ('2R2', ' F2', ' U2', '2R ', ' U2', ' F2', '2R2', ' F2', "2L ", ' F2', "2L'", "2R'", ' F2', "2R ", " F2"), legacy = 'WingSwapParallel-BX01')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v12', ('2R2', ' B2', ' U2', '2R ', ' U2', ' B2', '2R2', ' B2', '2L ', ' U2', "2L'", "2R'", ' U2', '2R ', ' B2'), legacy = 'WingSwapParallel-BX02')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v13', ('2R2', ' B2', ' U2', '2R ', ' U2', ' B2', '2R2', ' B2', "2R'", ' U2', "2R ", "2L ", ' U2', "2L'", ' B2'), legacy = 'WingSwapParallel-BX03')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v14', ('2L2', ' F2', ' U2', '2L ', ' U2', ' F2', '2L2', ' F2', "2R'", ' F2', '2R ', '2L ', ' F2', "2L'", ' F2'), legacy = 'WingSwapParallel-BX04')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v15', ('2L2', ' F2', ' U2', '2L ', ' U2', ' F2', '2L2', ' F2', "2L ", ' F2', "2L'", "2R'", ' F2', "2R ", ' F2'), legacy = 'WingSwapParallel-BX05')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v16', ('2L2', ' B2', ' U2', '2L ', ' U2', ' B2', '2L2', ' B2', '2L ', ' U2', "2L'", "2R'", ' U2', '2R ', ' B2'), legacy = 'WingSwapParallel-BX06')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v17', ('2L2', ' B2', ' U2', '2L ', ' U2', ' B2', '2L2', ' B2', "2R'", ' U2', "2R ", "2L ", ' U2', "2L'", ' B2'), legacy = 'WingSwapParallel-BX07')

            self._add_myperm2('W2-2s[UB@L<>UF@R]~v18', ('2L2', ' B2', ' D2', '2R ', ' D2', ' B2', '2L2', ' F2', '2R ', ' U2', "2R'", "2L'", ' U2', '2L ', ' F2'), legacy = 'WingSwapParallel-BY00')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v19', ('2L2', ' B2', ' D2', '2R ', ' D2', ' B2', '2L2', ' F2', "2L'", ' U2', '2L ', '2R ', ' U2', "2R'", ' F2'), legacy = 'WingSwapParallel-BY01')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v20', ('2L2', ' B2', ' D2', '2R ', ' D2', ' B2', '2L2', ' F2', "2R'", ' F2', "2R ", "2L ", ' F2', "2L'", ' F2'), legacy = 'WingSwapParallel-BY02')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v21', ('2L2', ' B2', ' D2', '2R ', ' D2', ' B2', '2L2', ' F2', '2L ', ' F2', "2L'", "2R'", ' F2', '2R ', ' F2'), legacy = 'WingSwapParallel-BY03')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v22', ('2R2', ' B2', ' D2', '2L ', ' D2', ' B2', '2R2', ' F2', '2R ', ' U2', "2R'", "2L'", ' U2', '2L ', ' F2'), legacy = 'WingSwapParallel-BY04')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v23', ('2R2', ' B2', ' D2', '2L ', ' D2', ' B2', '2R2', ' F2', "2L'", ' U2', "2L ", "2R ", ' U2', "2R'", ' F2'), legacy = 'WingSwapParallel-BY05')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v24', ('2R2', ' B2', ' D2', '2L ', ' D2', ' B2', '2R2', ' F2', "2R'", ' F2', '2R ', '2L ', ' F2', "2L'", ' F2'), legacy = 'WingSwapParallel-BY06')
            self._add_myperm2('W2-2s[UB@L<>UF@R]~v25', ('2R2', ' B2', ' D2', '2L ', ' D2', ' B2', '2R2', ' F2', "2L ", ' F2', "2L'", "2R'", ' F2', "2R ", ' F2'), legacy = 'WingSwapParallel-BY07')

            #('2L2', ' F2', ' D2', "2R'", ' D2', ' B2', '2L2', ' B2', '2R ', ' D2', "2L'", "2R'", ' D2', '2R ', ' F2')

            self._add_myperm2('W2-2s[RF@U<>UF@R]', swapc, legacy = 'WingSwapSkew-C')
            self._add_myperm2('W2-2s[RF@D<>UF@L]', swapd, legacy = 'WingSwapSkew-D')
            self._add_myperm2('W2-2s[RF@D<>UF@R]', swapex, legacy = 'WingSwapSkew-Ex')
            self._add_myperm2('W2-2s[RF@U<>UF@L]', swapey, legacy = 'WingSwapSkew-Ey')
            self._add_myperm2('W2-2s[FL@U<>UB@R]', swapfx, legacy = 'WingSwapSkew-Fx')
            self._add_myperm2('W2-2s[FL@D<>UB@L]', swapfy, legacy = 'WingSwapSkew-Fy')
            self._add_myperm2('W2-2s[FL@U<>UB@L]', swapg, legacy = 'WingSwapSkew-G')
            self._add_myperm2('W2-2s[FL@D<>UB@R]', swaph, legacy = 'WingSwapSkew-H')
            
            self._add_myperm2('W2-4[UL@B>UR@B>UL@F>UR@F]~v01', ('2F2', ' R2', "2F'", ' U2', "2F'", ' U2', ' R2', "2F'", ' R2', '2F ', ' R2', "2F'", ' R2', '2F2', ' R2'), legacy = 'L2NA')
            self._add_myperm2('W2-4[UL@B>UR@B>UL@F>UR@F]~v02', ('2B2', ' D2', '2F ', ' U2', '2B ', ' L2', "2F'", ' L2', ' U2', "2B'", ' R2', '2F ', ' R2', ' D2', '2B2'), legacy = 'L2NA1')
            self._add_myperm2('W2-4[UL@B>UR@F>UL@F>UR@B]', ('2B2', ' D2', ' R2', "2F'", ' R2', '2B ', ' U2', ' L2', '2F ', ' L2', "2B'", ' U2', "2F'", ' D2', '2B2'), legacy = 'L2NA2')
            self._add_myperm2('W2-4[UL@B>UR@B>UR@F>UL@F]', ('2B2', ' U2', "2B'", ' U2', ' D2', '2F ', ' D2', ' U2', "2B'", ' U2', '2B2'), legacy = 'L2OA')
            self._add_myperm2('W2-4[UL@B>UL@F>UR@B>UR@F]', ("2F'", ' U2', "2F'", ' U2', ' R2', "2F'", ' R2', "2F'", ' L2', '2B2', ' L2', '2F ', ' U2', '2F2', ' U2'), legacy = 'L2ZA')

            self._add_myperm2('W2-4[UF@L>UR@F>UF@R>UR@B]~v02', ('2F ', " D'", '2L ', ' D ', ' R2', " D'", "2L'", ' D ', ' R2', '2F ', ' D ', '2R2', " D'", ' R2', ' D ', '2R2', ' D ', '2F ', ' D2', '2F2', ' R2', '2B ', ' U2', '2F2', ' U2', ' R2', "2B'", ' R2'), legacy = 'L2E-NB0')
            self._add_myperm2('W2-4[UF@L>UF@R>UR@F>UR@B]', (" R "," B ","2R'"," U2","2R'"," U2"," B2","2R'"," B2","2R'"," F2","2L2"," F2","2R "," U2","2R2"," U2"," B'"," R'"), legacy = 'L2E-OB0')
            self._add_myperm2('W2-4[UF@L>UF@R>UR@B>UR@F]', (" R "," B ","2R2"," U2","2R "," U2"," D2","2L'"," U2"," D2","2R "," U2","2R2"," B'"," R'"), legacy = 'L2E-ZB0')
            self._add_myperm2('W2-4[UF@L>UR@F>UR@B>UF@R]', self.invert_moves(self.myperms2['W2-4[UF@L>UF@R>UR@B>UR@F]']), legacy = 'L2E-ZB1')
            self._add_myperm2('W2-4[UF@L>UR@B>UR@F>UF@R]', self.invert_moves(self.myperms2['W2-4[UF@L>UF@R>UR@F>UR@B]']), legacy = 'L2E-OB1')


            self._add_myperm2('W2-4[DF@L>UL@B>DF@R>UL@F]', ('2B ', " U'", '2R ', ' U ', ' L2', " U'", "2R'", ' U ', ' L2', '2B ', ' U ', "2L'", " U'", ' R2', ' U ', '2L ', ' U ', "2B'", ' U2', '2B2', ' R2', '2B ', ' U2', '2B2', ' U2', ' R2', '2B ', ' R2'), legacy = 'L2E-NC0')
            self._add_myperm2('W2-4[DF@L>UL@F>UL@B>DF@R]', (" U'", '2L ', ' F2', '2L ', ' F2', ' U2', '2L ', ' U2', '2L ', ' D2', '2R2', ' D2', "2L'", ' F2', '2L2', ' F2', ' U '), legacy = 'L2E-ZC0')
            self._add_myperm2('W2-4[DF@L>UL@B>UL@F>DF@R]', (" U'", '2L2', ' F2', "2L'", ' F2', ' B2', '2R ', ' F2', ' B2', "2L'", ' F2', '2L2', ' U '), legacy = 'L2E-OC0')
            self._add_myperm2('W2-4[DF@L>DF@R>UL@F>UL@B]', self.invert_moves(self.myperms2['W2-4[DF@L>UL@B>UL@F>DF@R]']), legacy = 'L2E-OC1')
            self._add_myperm2('W2-4[DF@L>DF@R>UL@B>UL@F]', self.invert_moves(self.myperms2['W2-4[DF@L>UL@F>UL@B>DF@R]']), legacy = 'L2E-ZC1')


            self._add_myperm2('W2-4[DL@B>UR@F>DL@F>UR@B]', ('2F2', ' D2', '2F ', ' D2', "2F'", ' D2', '2F ', ' D2', ' R2', '2F ', ' R2', '2F ', ' D2', '2F2', ' D2'), legacy = 'L2ND')
            self._add_myperm2('W2-4[DL@B>DL@F>UR@F>UR@B]', ('2B2', ' U2', "2B'", ' U2', '2B ', ' U2', "2B'", ' U2', "2B'", ' U2', '2B ', ' L2', "2B'", ' L2', '2B ', ' U2'), legacy = 'L2OD')
            self._add_myperm2('W2-4[DL@B>UR@F>UR@B>DL@F]', ('2B2', ' D2', '2B ', ' R2', "2B'", ' R2', "2F'", ' U2', "2B'", ' U2', '2F ', ' D2', '2B2'), legacy = 'L2ZD')

            
            self._add_myperm2('W2-4s[UL@B<>UR@F;UL@F<>UR@B]~v01', ('2B2', ' L2', ' U2', '2B2', ' U2', ' L2', '2B2'), legacy = 'L2XA')
            self._add_myperm2('W2-4s[UL@B<>UR@F;UL@F<>UR@B]~v02', ('2B2', ' R2', ' D2', '2F2', ' D2', ' R2', '2B2'), legacy = 'L2XA1')
            self._add_myperm2('W2-4s[UL@B<>UL@F;UR@B<>UR@F]~v01', ('2B ', ' L2', '2F2', ' D2', '2F ', ' D2', '2F ', ' L2', '2B2', ' U2', '2B ', ' U2'), legacy = 'L2FA')
            self._add_myperm2('W2-4s[UL@B<>UL@F;UR@B<>UR@F]~v02', ('2B ', ' L2', '2F2', ' D2', '2F ', ' D2', "2B'", ' L2', '2F2', ' U2', "2F'", ' U2'), legacy = 'L2FA1')
            self._add_myperm2('W2-4s[UL@B<>UR@B;UL@F<>UR@F]', (" R2",'2F ', ' R2', '2B ', ' R2', ' L2', '2F ', "2B'", ' L2', ' R2', '2B ', ' R2', '2B '," R2"), legacy = 'L2HA')

            self._add_myperm2('W2-4s[DL@B<>UR@B;DL@F<>UR@F]', (' L2', '2B2', ' L2', ' U2', '2B2', ' U2', ' L2', '2B2', ' L2'), legacy = 'L2HD')
            self._add_myperm2('W2-4s[DL@B<>DL@F;UR@B<>UR@F]~v01', (" L2","2B "," L2","2F2"," D2","2F "," D2","2F "," L2","2B2"," U2","2B "," U2"," L2"), legacy = 'L2FD')
            self._add_myperm2('W2-4s[DL@B<>DL@F;UR@B<>UR@F]~v02', (' U2', '2F ', ' U2', '2B2', ' R2', '2B ', ' R2', "2F'", ' U2', '2B2', ' L2', "2B'", ' L2', ' U2'), legacy = 'L2FD1')
            self._add_myperm2('W2-4s[DL@B<>UR@F;DL@F<>UR@B]', ('2F ', ' U2', '2B ', ' U2', ' D2', '2F ', "2B'", ' D2', ' U2', '2B ', ' U2', '2B '), legacy = 'L2XD')


            self._add_myperm2('W2-4s[UF@L<>UR@B;UF@R<>UR@F]', ("2R'", ' F2', " D'", '2F2', ' D ', ' F2', " D'", '2F2', ' D ', ' F2', ' D ', "2B'", " D'", ' F2', ' D ', '2B ', " D'", '2R '), legacy = 'L2E-HB0')
            self._add_myperm2('W2-4s[UF@L<>UR@F;UF@R<>UR@B]~v02', ("2R'", ' F2', ' D ', "2B'", " D'", ' F2', ' D ', '2B ', " D'", ' F2', " D'", '2F2', ' D ', ' F2', " D'", '2F2', ' D ', '2R '), legacy = 'L2E-XB0')
            self._add_myperm2('W2-4s[UF@L<>UF@R;UR@B<>UR@F]', (" R "," B ","2L "," F2","2R2"," D2","2R "," D2","2R "," F2","2L2"," U2","2L "," U2"," B'"," R'"), legacy = 'L2E-FB0')

            self._add_myperm2('W2-4s[DF@L<>UL@F;DF@R<>UL@B]', ('2R ', ' F2', ' D ', '2F2', " D'", ' F2', ' D ', '2F2', " D'", ' F2', " D'", '2B ', ' D ', ' F2', " D'", "2B'", ' D ', "2R'"), legacy = 'L2E-XC0')
            self._add_myperm2('W2-4s[DF@L<>UL@B;DF@R<>UL@F]', ('2R ', ' F2', " D'", '2B ', ' D ', ' F2', " D'", "2B'", ' D ', ' F2', ' D ', '2F2', " D'", ' F2', ' D ', '2F2', " D'", "2R'"), legacy = 'L2E-HC0')
            self._add_myperm2('W2-4s[DF@L<>DF@R;UL@B<>UL@F]', (" U'", "2R'", ' D2', '2L2', ' B2', "2L'", ' B2', "2L'", ' D2', '2R2', ' F2', "2R'", ' F2', ' U '), legacy = 'L2E-FC0')

            self._add_myperm2('W2-6p[3x2][BR@D>LB@U>FL@D;BR@U>LB@D>FL@U]', ('2U ', "2D'", ' B2', "2U'", '2D ', ' B2'), legacy = 'WingParallel6-A')
            self._add_myperm2('W2-6p[3x2][BR@D>LB@D>FL@U;BR@U>LB@U>FL@D]', (" F'", "2U'", '2D ', ' F ', ' L2', " F'", '2U ', "2D'", ' F ', ' L2'), legacy = 'WingParallel6-B')
            self._add_myperm2('W2-6p[3x2][BR@D>LB@D>FL@D;BR@U>LB@U>FL@U]', (" L'", '2F ', "2B'", " L'", ' B2', ' L ', "2F'", '2B ', " L'", ' B2', ' L2'), legacy = 'WingParallel6-C')
            self._add_myperm2('W2-6p[3x2][BR@D>LB@U>FL@U;BR@U>LB@D>FL@D]', (' B2', " R'", '2U ', "2D'", ' R ', ' B2', " R'", "2U'", '2D ', ' R '), legacy = 'WingParallel6-D')
            self._add_myperm2('W2-6[BR@D>LB@U>FL@D>BR@U>LB@D>FL@U]', ('2U2', ' R2', "2D'", ' L2', "2U'", ' F2', '2D ', ' F2', ' L2', '2U ', ' B2', "2D'", '2U2', ' B2', ' R2', '2U2'), legacy = 'WingParallel6-E')
            self._add_myperm2('W2-6[BR@D>LB@D>FL@U>BR@U>LB@U>FL@D]', ('2U2', ' L2', '2U ', ' B2', '2U ', ' B2', ' L2', '2U ', ' L2', "2U'", ' L2', '2U ', ' B2', '2U2', ' B2', ' L2', '2U2', ' L2'), legacy = 'WingParallel6-F')
            self._add_myperm2('W2-6[BR@D>LB@D>FL@D>BR@U>LB@U>FL@U]', ('2D ', ' L2', "2D'", '2U ', ' L2', ' F2', ' R2', '2D ', ' R2', ' F2', '2D2', ' B2', '2D ', ' L2', "2U'", ' F2', '2D ', ' F2', ' L2', '2U ', ' B2', '2D '), legacy = 'WingParallel6-G')
            self._add_myperm2('W2-6[BR@D>LB@U>FL@U>BR@U>LB@D>FL@D]', ('2U ', "2D'", ' B2', '2D ', "2U'", ' B2', '2U2', ' R2', "2D'", ' L2', '2U ', ' L2', ' F2', '2U ', ' F2', "2U'", ' F2', '2D ', ' F2', ' R2', '2U2'), legacy = 'WingParallel6-H')
            
            self._add_myperm2('W2-4s[FL@D<>UF@R;FL@U<>UF@L]', (" F'", '2U ', "2D'", ' F ', ' L2', " F'", "2U'", '2D ', ' F ', ' L2', '2D2', ' F2', ' L2', '2D2', ' L2', ' F2', '2D2'), legacy = 'Edge6PAX')
            self._add_myperm2('W2-4s[FL@D<>UF@L;FL@U<>UF@R]', (' F ', '2D2', '2U2', " F'", ' L2', ' F ', '2D2', '2U2', " F'", ' L2', '2U2', ' F2', ' L2', '2U2', ' L2', ' F2', '2U2'), legacy = 'Edge6PBX')
            self._add_myperm2('W2-4s[FL@D<>UB@L;FL@U<>UB@R]', (' U2', ' F ', '2R ', "2L'", " F'", ' U2', ' F ', "2R'", '2L ', " F'", '2L2', ' B2', ' U2', '2L2', ' U2', ' B2', '2L2'), legacy = 'Edge6PCX')
            self._add_myperm2('W2-4s[FL@D<>UB@R;FL@U<>UB@L]', (' U2', " F'", '2L2', '2R2', ' F ', ' U2', " F'", '2L2', '2R2', ' F ', '2R2', ' B2', ' U2', '2R2', ' U2', ' B2', '2R2'), legacy = 'Edge6PDX')
        
            self._add_myperm2('W2-4[FL@D>UF@R>FL@U>UF@L]', (" F'", '2U ', "2D'", ' F ', ' L2', " F'", "2U'", '2D ', ' F ', ' L2', '2D2', ' F2', "2D'", ' L2', "2D'", ' L2', ' F2', "2D'", ' F2', '2D ', ' F2', "2D'", ' F2', '2D2', ' F2'), legacy = 'Edge6PAN')
            self._add_myperm2('W2-4[FL@D>UF@L>FL@U>UF@R]', (' F ', '2D2', '2U2', " F'", ' L2', ' F ', '2D2', '2U2', " F'", ' L2', '2U2', ' F2', '2U ', ' L2', '2U ', ' L2', ' F2', '2U ', ' F2', "2U'", ' F2', '2U ', ' F2', '2U2', ' F2'), legacy = 'Edge6PBN')
            self._add_myperm2('W2-4[FL@D>UB@R>FL@U>UB@L]', (' U2', ' F ', '2R ', "2L'", " F'", ' U2', ' F ', "2R'", '2L ', " F'", '2L2', ' B2', '2L ', ' U2', '2L ', ' U2', ' B2', '2L ', ' B2', "2L'", ' B2', '2L ', ' B2', '2L2', ' B2'), legacy = 'Edge6PCN')
            self._add_myperm2('W2-4[FL@D>UB@L>FL@U>UB@R]', (' U2', " F'", '2L2', '2R2', ' F ', ' U2', " F'", '2L2', '2R2', ' F ', '2R2', ' B2', "2R'", ' U2', "2R'", ' U2', ' B2', "2R'", ' B2', '2R ', ' B2', "2R'", ' B2', '2R2', ' B2'), legacy = 'Edge6PDN')


            self._add_myperm2('W2-3[FL@D>RF@U>FL@U]~v01', ("2U'", ' R ', '2B ', " R'", ' F2', ' R ', "2B'", " R'", ' F2', '2U '), legacy = 'EdgePK-A00')
            self._add_myperm2('W2-3[FL@D>FL@U>RF@U]~v01', ("2U'", ' F2', ' R ', '2B ', " R'", ' F2', ' R ', "2B'", " R'", '2U '), legacy = 'EdgePK-A01')
            self._add_myperm2('W2-3[FL@D>RF@U>FL@U]~v02', ("2U'", " R'", '2F ', ' R ', ' F2', " R'", "2F'", ' R ', ' F2', '2U '), legacy = 'EdgePK-A02')
            self._add_myperm2('W2-3[FL@D>FL@U>RF@U]~v02', ("2U'", ' F2', " R'", '2F ', ' R ', ' F2', " R'", "2F'", ' R ', '2U '), legacy = 'EdgePK-A03')
            self._add_myperm2('W2-3[FL@D>RF@U>FL@U]~v03', ('2U2', ' L2', " B'", "2R'", ' B ', ' L2', " B'", '2R ', ' B ', '2U2'), legacy = 'EdgePK-A04')
            self._add_myperm2('W2-3[FL@D>FL@U>RF@U]~v03', ('2U2', " B'", "2R'", ' B ', ' L2', " B'", '2R ', ' B ', ' L2', '2U2'), legacy = 'EdgePK-A05')
            self._add_myperm2('W2-3[FL@D>RF@U>FL@U]~v04', ('2U2', ' L2', ' B ', "2L'", " B'", ' L2', ' B ', '2L ', " B'", '2U2'), legacy = 'EdgePK-A06')
            self._add_myperm2('W2-3[FL@D>FL@U>RF@U]~v04', ('2U2', ' B ', "2L'", " B'", ' L2', ' B ', '2L ', " B'", ' L2', '2U2'), legacy = 'EdgePK-A07')
            self._add_myperm2('W2-3[FL@U>RF@U>RF@D]', ('2U2', ' R2', ' B ', '2L ', " B'", ' R2', ' B ', "2L'", " B'", '2U2'), legacy = 'EdgePK-A08')
            self._add_myperm2('W2-3[FL@U>RF@D>RF@U]', ('2U2', ' B ', '2L ', " B'", ' R2', ' B ', "2L'", " B'", ' R2', '2U2'), legacy = 'EdgePK-A09')

            self._add_myperm2('W2-3[BR@U>FL@U>FL@D]~v01', ('2U2', " L'", '2F ', ' U2', "2F'", ' L ', '2U2', " L'", '2F ', ' U2', "2F'", ' L '), legacy = 'EdgePK-D00')
            self._add_myperm2('W2-3[BR@U>FL@D>FL@U]~v01', (" L'", '2F ', ' U2', "2F'", ' L ', '2U2', " L'", '2F ', ' U2', "2F'", ' L ', '2U2'), legacy = 'EdgePK-D01')
            self._add_myperm2('W2-3[BR@U>FL@U>FL@D]~v02', ('2U2', ' L2', '2D ', ' L2', "2D'", ' L2', '2U2', ' L2', '2D ', ' L2', "2D'", ' L2'), legacy = 'EdgePK-D02')
            self._add_myperm2('W2-3[BR@U>FL@D>FL@U]~v02', (' L2', '2D ', ' L2', "2D'", ' L2', '2U2', ' L2', '2D ', ' L2', "2D'", ' L2', '2U2'), legacy = 'EdgePK-D03')



            self.myperms2['WingParallel8-OneOpposite'] = ('2U ', ' F2', ' B2', "2U'", '2D ', ' B2', ' F2', '2U ')
            self.myperms2['WingParallel8-OneParallel'] = ('2D ', ' F2', ' B2', "2D'", ' F2', ' B2', '2D2', ' L2', ' R2', '2U ', ' L2', ' R2', '2D ')
            self.myperms2['WingParallel8-OneTwo'] = ("2U ","2D'"," F2"," B2","2U'","2D "," B2","2U "," F2","2U "," F2","2U "," F2","2U "," F2","2U "," F2")
            self.myperms2['WingParallel8-TwoTwo'] = ("2U ","2D'"," F2"," B2","2U'","2D "," B2"," F2")

            self.myperms2['SideLA-'] = ('2L ', ' U2', "2R'", ' F2', '2R ', ' F2', ' U2', ' B2', "2R'", ' D2', '2R2', ' D2', "2R'", ' B2')
            self.myperms2['SideLB-'] = self.invert_moves(self.myperms2['SideLA-'])
            self.myperms2['SideLC-'] = ('2L ', ' F2', '2L ', ' F2', '2L ', ' F2', '2L2', ' B2', "2L'", ' F2', '2L ', ' B2')
            self.myperms2['SideLD-'] = self.invert_moves(self.myperms2['SideLC-'])
            self.myperms2['SideLE-'] = ("2R'", ' B2', ' U2', '2R ', ' U2', "2R'", ' U2', '2R2', '2L ', ' U2', "2L'", ' B2', '2R2')
            self.myperms2['SideLF-'] = self.invert_moves(self.myperms2['SideLE-'])

            self.myperms2['SideLG-'] = (' F2', ' U2', ' F ', "2U'", " F'", ' U2', ' F ', '2U2', ' F ', ' U2', " F'", "2U'", ' F ', ' U2')
            self.myperms2['SideLH-'] = ('2R2', ' B2', "2R'", ' B2', "2R'", ' U2', '2L2', ' F2', "2L'", ' F2', "2L'", ' U2')
            self.myperms2['SideLI-'] = ('2R ', ' B2', "2L'", ' B2', '2L2', ' F2', "2L'", ' U2', '2L ', ' U2', '2L ', ' F2', '2L2', ' U2', "2R'", ' U2')

            self.myperms2['SideRA-'] = (" F2",'2L2', ' F2', '2L ', ' B2', "2L'", ' B2', ' F2', ' U2', "2L'", ' U2', '2R ', ' B2', "2R'", ' B2', '2L2'," F2")
            self.myperms2['SideRB-'] = self.invert_moves(self.myperms2['SideRA-'])
            self.myperms2['SideRC-'] = ('2L ', ' B2', '2L ', ' B2', ' D2', '2L ', '2R ', ' D2', "2R'", ' D2', "2L'", ' D2', "2L'")
            self.myperms2['SideRD-'] = self.invert_moves(self.myperms2['SideRC-'])
            self.myperms2['SideRE-'] = ('2L2', ' U2', "2L'", ' U2', "2L'", "2R'", ' D2', ' B2', '2R ', ' B2', ' D2', '2R ')
            self.myperms2['SideRF-'] = self.invert_moves(self.myperms2['SideRE-'])

            self.myperms2['SideRG-'] = ('2L2', ' D2', '2L ', ' B2', "2L'", ' B2', "2L'", ' D2', '2L2', ' B2', '2R ', ' B2', "2R'", ' U2', '2L ', ' U2')
            self.myperms2['SideRH-'] = ("2R'", ' B2', '2R2', ' F2', "2R'", ' D2', '2R ', ' D2', '2R ', ' F2', '2R2', ' D2', "2L'", ' D2', '2L ', ' B2')
            self.myperms2['SideRI-'] = ("2L'", ' U2', '2L ', ' U2', '2L ', ' F2', '2L2', ' U2', "2R'", ' U2', '2R ', ' B2', "2L'", ' B2', '2L2', ' F2')


            self.myperms2['SideSA-'] = (" U2",'2L ', ' U2', "2R'", ' F2', '2R ', ' F2', ' U2', ' B2', "2R'", ' D2', '2R2', ' D2', "2R'", ' B2'," U2")
            self.myperms2['SideSB-'] = (" U2",'2L ', ' F2', '2L ', ' F2', '2L ', ' F2', '2L2', ' B2', "2L'", ' F2', '2L ', ' B2'," U2")
            self.myperms2['SideSC-'] = ("2R'", ' U2', "2R'", ' U2', "2R'", ' F2', ' D2', ' B2', '2L ', ' B2', ' D2', ' F2', "2R'")
            self.myperms2['SideSD-'] = self.invert_moves(self.myperms2['SideSC-'])

            self.myperms2['SideSE-'] = (" F'", '2U ', ' F ', ' U2', " F'", "2U'", ' F ', ' U2', ' F2', " U'", '2F ', ' U ', ' F2', " U'", "2F'", ' U ')
            self.myperms2['SideSF-'] = ('2L ', ' F2', '2L ', ' F2', '2L2', ' U2', '2R ', ' B2', '2R ', ' B2', '2R2', ' U2')
            self.myperms2['SideSG-'] = ('2L ', ' F2', '2R2', ' U2', "2R'", ' F2', "2R'", ' F2', '2R ', ' U2', '2R2', ' D2', '2R ', ' D2', "2L'", ' F2')

            self.myperms2['SideTA-'] = ('2L2', ' F2', '2L ', ' B2', "2L'", ' B2', ' F2', ' U2', "2L'", ' U2', '2R ', ' B2', "2R'", ' B2', '2L2')
            self.myperms2['SideTB-'] = ('2R2', ' D2', "2R'", ' B2', '2R2', ' B2', ' D2', ' B2', '2R ', ' B2', ' D2', '2R ', ' D2')
            self.myperms2['SideTC-'] = self.invert_moves(self.myperms2['SideTB-'])

            self.myperms2['SideTD-'] = ('2R ', ' B2', "2R'", ' B2', "2R'", ' U2', '2L2', ' F2', "2L'", ' F2', "2L'", ' U2',"2R ")
            self.myperms2['SideTE-'] = ('2R2', ' D2', '2L ', ' D2', "2L'", ' F2', '2L2', ' F2', '2L ', ' D2', "2L'", ' D2', '2R2')
            
            self.myperms2['SideKKA-'] = ("2R "," U2","2R "," U2","2R "," U2","2R "," U2","2R ")
            self.myperms2['SideKKB-'] = ("2L'", ' D2', "2L'", ' D2', '2L ', ' B2', '2L2', ' D2', "2L'", ' D2', "2L'", ' B2')
            self.myperms2['SideKKC-'] = ('2R ', ' U2', '2R ', ' U2', '2R2', ' B2', "2R'", ' U2', '2R ', ' U2', '2R ', ' B2')

            self.myperms2['SideKKD-'] = (' U2', '2R ', ' U2', ' F2', ' B2', ' D2', '2L ', ' D2', ' B2', ' F2')
            self.myperms2['SideKKE-'] = (" D'", "2F'", ' D ', ' B2', " D'", '2F ', ' D ', " U'", "2F'", ' U ', ' B2', " U'", '2F ', ' U ')
            self.myperms2['SideKKF-'] = (' F2', "2L'", ' B2', '2L ', ' F2', "2L'", ' B2', '2L ', '2R ', ' F2', "2R'", ' B2', '2R ', ' F2', "2R'", ' B2')
            
            self.myperms2['SideJJA-'] = ('2R2', ' U2', '2L ', ' U2', "2L'", ' B2', '2R ', ' B2', ' U2', '2R2', ' D2', "2R'", ' U2', '2R ', ' D2')
            self.myperms2['SideJJB-'] = ("2R'", ' U2', ' D2', "2R'", ' U2', "2R ", ' D2', '2R2', ' U2', "2R'", ' U2', "2R'")
            self.myperms2['SideJJC-'] = self.invert_moves(self.myperms2['SideJJB-'])

            self.myperms2['SideJJD-'] = ('2L ', ' D2', '2L ', ' D2', '2L ', ' B2', '2R2', ' U2', '2R ', ' U2', '2R ', ' B2', '2L ')
            self.myperms2['SideJJE-'] = ('2L2', ' B2', "2R'", ' B2', '2L ', '2R2', ' B2', "2L'", ' B2', '2L ', ' B2', '2R2', ' B2', ' U2', '2L ', ' U2', '2R ')

            self.myperms2['SideIIA-'] = (" U2","2R "," U2","2R "," U2","2R "," U2","2R "," U2","2R "," U2")
            self.myperms2['SideIIB-'] = (' B2', '2L ', ' B2', "2L'", ' D2', ' B2', "2R'", ' B2', ' D2', "2R'", ' D2', "2R'", ' D2', "2R'", ' D2', "2R'", ' D2')
            
            self.myperms2['SideIIC-'] = ("2R'", ' B2', ' U2', ' D2', ' F2', "2L'", ' F2', ' D2', ' U2', ' B2')
            self.myperms2['SideIID-'] = (" B'", "2D'", " B'", ' D2', ' B ', '2D ', " B'", ' D2', ' B2', ' D ', "2B'", ' D ', ' F2', " D'", '2B ', ' D ', ' F2', ' D2')

            self.myperms2['SideSSA-'] = ('2R ', ' U2', '2R2', ' B2', "2R'", ' B2', '2R2', ' F2', ' D2', '2L ', ' D2', ' F2', ' U2')
            self.myperms2['SideSSB-'] = (' F2', '2R ', ' F2', '2L ', ' D2', "2L'", ' D2', "2R'", ' F2', '2L ', ' F2', "2L'", ' U2', '2R ', ' U2')

            self.myperms2['SideSSC-'] = (' F2', '2L2', ' F2', ' U2', ' F2', '2L ', "2R'", ' F2', '2R ', '2L ', ' U2')
            self.myperms2['SideSSD-'] = ('2R2', ' U2', ' B2', '2R ', "2L'", ' B2', "2R'", '2L ', ' B2', '2R2', ' B2', ' U2')
            

        self.myperms2['E-Perm'] = (" R "," B'"," R'"," F "," R "," B "," R'"," F'"," R "," B "," R'"," F "," R "," B'"," R'"," F'")
        self.myperms2['X-Perm-A'] = (" U'", ' L2', ' F2', ' B2', ' R2', " D'", ' R2', ' B2', ' F2', ' L2')
        self.myperms2['X-Perm-B'] = (" L "," F2"," R2"," D2"," R "," D2"," R "," F2"," L2"," U2"," L "," U2")
        self.myperms2['X-Perm-C'] = (" F ", ' R2', ' F ', ' U2', " F'", ' R2', ' D2', ' B ', ' L2', " B'", ' D2', " F'")

        self.myperms2['X-Perm-D'] = (' R2', ' B2', " D'", ' B2', ' R2', ' L2', ' F2', " U'", ' F2', ' L2')
        self.myperms2['X-Perm-E'] = (" F'",) + (" R "," U "," R'"," U'") * 3 + (" F ",)
        self.myperms2['X-Perm-F'] = (" F'", ' R2', ' F ', ' U2', " F'", ' R2', ' D2', ' B ', ' L2', " B'", ' D2', " F ")        

        self.myperms2['X-Perm-G'] = (' R2', ' D ', ' U ', ' R2', ' F2', ' L2', ' B2', ' D ', ' U ', ' B2', ' L2', ' F2')

        

        self._add_myperm2('C2[UBR>BRU;ULB>BUL]', (" R'"," U2"," R'"," B "," D2"," B'"," R "," U2"," R'"," B "," D2"," B'"," R2"), legacy = 'CornerTwist-A')
        self._add_myperm2('C2[ULB>BUL;URF>RFU]', (" U2"," R'"," B "," D2"," B'"," R "," U2"," R'"," B "," D2"," B'"," R "), legacy = 'CornerTwist-B')
        self._add_myperm2('C2[DFR>FRD;ULB>BUL]', (" R "," U2"," R'"," B "," D2"," B'"," R "," U2"," R'"," B "," D2"," B'"), legacy = 'CornerTwist-C')
        self._add_myperm2('C3[UBR>BRU;UFL>FLU;URF>RFU]', (' B ', " L'", " B'", ' R ', ' B ', ' L ', " B'", ' U2', ' R ', ' D ', " R'", ' U2', ' R ', " D'", ' R2'), legacy = 'CornerTwist-D')
        self._add_myperm2('C3[DRB>RBD;UBR>BRU;UFL>FLU]', (" R'", ' F ', ' R ', ' B ', " R'", " F'", ' R ', ' B2', " D'", ' B ', ' U2', " B'", ' D ', ' B ', ' U2'), legacy = 'CornerTwist-E')
        self._add_myperm2('C3[DFR>FRD;UBR>BRU;UFL>FLU]', (' R2', ' B ', " L'", ' B2', ' U ', " F'", " U'", ' B ', ' U ', ' F ', " U'", ' R2', ' B ', ' L ', " B'"), legacy = 'CornerTwist-F')
        self._add_myperm2('C3[DFR>RDF;UBR>RUB;UFL>LUF]', self.invert_moves(self.myperms2['C3[DFR>FRD;UBR>BRU;UFL>FLU]']), legacy = 'CornerTwist-F01')
        


        self._add_myperm2('C3[UBR>UFL>URF]', (" R "," B'"," R "," F2"," R'"," B "," R "," F2"," R2"), legacy = 'CornerPermutation-A00')
        self._add_myperm2('C3[UBR>LUF>URF]', (" R ",' U2', ' R ', ' D ', " R'", ' U2', ' R ', " D'", " R2"), legacy = 'CornerPermutation-A01')
        self._add_myperm2('C3[UBR>FLU>URF]~v01', (' F ', ' R ', ' B ', " R'", " F'", ' R ', " B'", " R'"), legacy = 'CornerPermutation-A02')
        self._add_myperm2('C3[UBR>FLU>FUR]', (" L'", ' B ', ' U2', " B'", ' L ', ' B ', " L'", ' U2', ' L ', " B'"), legacy = 'CornerPermutation-A03')
        self._add_myperm2('C3[UBR>UFL>FUR]', (" B'", ' R2', " B'", ' L2', ' B ', ' R2', " B'", ' L2', ' B2'), legacy = 'CornerPermutation-A04')
        self._add_myperm2('C3[UBR>LUF>FUR]', (' F2', " D'", ' F ', ' U2', " F'", ' D ', ' F ', ' U2', ' F '), legacy = 'CornerPermutation-A05')
        self._add_myperm2('C3[UBR>LUF>RFU]~v01', (' F ', " U'", " B'", ' U ', " F'", " U'", ' B ', ' U '), legacy = 'CornerPermutation-A06')
        self._add_myperm2('C3[UBR>FLU>RFU]~v01', (' R ', ' B ', " L'", " B'", " R'", ' B ', ' L ', " B'"), legacy = 'CornerPermutation-A07')
        self._add_myperm2('C3[UBR>UFL>RFU]', (' L2', ' B2', " L'", ' F2', ' L ', ' B2', " L'", ' F2', " L'"), legacy = 'CornerPermutation-A08')

        self._add_myperm2('C3[UBR>FLU>URF]~v02', (" L'", ' B ', ' L ', " F'", " L'", " B'", ' L ', ' F '), legacy = 'CornerPermutation-A09')
        self._add_myperm2('C3[UBR>LUF>RFU]~v02', (' U ', ' L ', " U'", " R'", ' U ', " L'", " U'", ' R '), legacy = 'CornerPermutation-A10')
        self._add_myperm2('C3[UBR>FLU>RFU]~v02', (" F'", " L'", ' F ', " R'", " F'", ' L ', ' F ', ' R '), legacy = 'CornerPermutation-A11')

        self._add_myperm2('C3[DRB>LUF>BRU]', (" B'"," R "," F2"," R'"," B "," R "," F2"," R'"), legacy = 'CornerPermutation-B00')
        self._add_myperm2('C3[DRB>FLU>BRU]~v01', (' U2', ' R ', ' D ', " R'", ' U2', ' R ', " D'", " R'"), legacy = 'CornerPermutation-B01')
        self._add_myperm2('C3[DRB>UFL>BRU]~v01', (" R'", ' F ', ' R ', ' B ', " R'", " F'", ' R ', " B'"), legacy = 'CornerPermutation-B02')
        self._add_myperm2('C3[DRB>UFL>UBR]', (' U ', ' R2', " U'", ' B2', ' U ', ' B2', ' U ', ' R2', " U'", ' B2', " U'", ' B2'), legacy = 'CornerPermutation-B03')
        self._add_myperm2('C3[DRB>LUF>UBR]', (" D'", ' R2', " D'", ' L ', ' D ', ' R2', " D'", " L'", ' D2'), legacy = 'CornerPermutation-B04')

        self._add_myperm2('C3[DRB>BRU>LUF]', self.invert_moves(self.myperms2['C3[DRB>LUF>BRU]']), legacy = 'CornerPermutation-B05')
        self._add_myperm2('C3[DRB>BRU>FLU]~v01', self.invert_moves(self.myperms2['C3[DRB>FLU>BRU]~v01']), legacy = 'CornerPermutation-B06')
        self._add_myperm2('C3[DRB>BRU>UFL]~v01', self.invert_moves(self.myperms2['C3[DRB>UFL>BRU]~v01']), legacy = 'CornerPermutation-B07')
        self._add_myperm2('C3[DRB>UBR>UFL]', self.invert_moves(self.myperms2['C3[DRB>UFL>UBR]']), legacy = 'CornerPermutation-B08')
        self._add_myperm2('C3[DRB>UBR>LUF]', self.invert_moves(self.myperms2['C3[DRB>LUF>UBR]']), legacy = 'CornerPermutation-B09')

        self._add_myperm2('C3[DRB>FLU>BRU]~v02', (" F'", ' D2', ' F ', ' U2', " F'", ' D2', ' F ', ' U2'), legacy = 'CornerPermutation-B10')
        self._add_myperm2('C3[DRB>UFL>BRU]~v02', (" B'", " D'", ' F2', ' D ', ' B ', " D'", ' F2', ' D '), legacy = 'CornerPermutation-B11')
        self._add_myperm2('C3[DRB>BRU>FLU]~v02', self.invert_moves(self.myperms2['C3[DRB>FLU>BRU]~v02']), legacy = 'CornerPermutation-B12')
        self._add_myperm2('C3[DRB>BRU>UFL]~v02', self.invert_moves(self.myperms2['C3[DRB>UFL>BRU]~v02']), legacy = 'CornerPermutation-B13')

        

        self._add_myperm2('C3[DFR>UBR>UFL]~v01', (" R "," U2"," R'"," U2"," R'"," F2"," R "," U2"," R "," U2"," R'"," F2"), legacy = 'CornerPermutation-C00')
        self._add_myperm2('C3[DFR>RUB>FLU]~v01', (" D'", ' F2', ' D ', " B'", " D'", ' F ', ' D ', ' B ', " D'", ' F ', ' D '), legacy = 'CornerPermutation-C01')
        self._add_myperm2('C3[DFR>BRU>LUF]', (" U'", " F'", ' U ', ' B ', " U'", " F'", ' U ', " B'", " U'", ' F2', ' U '), legacy = 'CornerPermutation-C02')
        self._add_myperm2('C3[DFR>BRU>FLU]', (" B'", ' D ', ' B ', ' U2', " B'", " D'", ' B ', ' U2'), legacy = 'CornerPermutation-C03')
        self._add_myperm2('C3[DFR>FLU>BRU]', (' U2', " B'", ' D ', ' B ', ' U2', " B'", " D'", ' B '), legacy = 'CornerPermutation-C04')
        self._add_myperm2('C3[DFR>UFL>UBR]~v01', self.invert_moves(self.myperms2['C3[DFR>UBR>UFL]~v01']), legacy = 'CornerPermutation-C05')
        self._add_myperm2('C3[DFR>FLU>RUB]~v01', self.invert_moves(self.myperms2['C3[DFR>RUB>FLU]~v01']), legacy = 'CornerPermutation-C06')
        self._add_myperm2('C3[DFR>LUF>BRU]', self.invert_moves(self.myperms2['C3[DFR>BRU>LUF]']), legacy = 'CornerPermutation-C07')
        self._add_myperm2('C3[DFR>RUB>FLU]~v02', (' U ', ' F2', ' U ', ' B ', " U'", ' F ', ' U ', " B'", " U'", ' F ', " U'"), legacy = 'CornerPermutation-C08')
        self._add_myperm2('C3[DFR>FLU>RUB]~v02', self.invert_moves(self.myperms2['C3[DFR>RUB>FLU]~v02']), legacy = 'CornerPermutation-C09')
        self._add_myperm2('C3[DFR>UBR>UFL]~v02', (' F2', " U'", ' R2', ' U ', ' R2', ' D ', ' R2', " D'", ' R2', " D'", ' F2', ' D '), legacy = 'CornerPermutation-C10')
        self._add_myperm2('C3[DFR>UFL>UBR]~v02', self.invert_moves(self.myperms2['C3[DFR>UBR>UFL]~v02']), legacy = 'CornerPermutation-C11')

        self._add_myperm2('EAll2[FL>LF;RF>FR]', (' R ', ' F ', ' U ', " F'", ' U2', ' F2', ' U ', ' D ', ' R ', " U'", " R'", " D'", ' F2', ' U ', " R'"), legacy = 'EdgeFlip2-A') 
        self._add_myperm2('EAll2[RF>FR;UF>FU]', (' D ', ' R ', " U'", " R'", " D'", ' F2', ' U ', ' F ', ' U ', " F'", ' U2', ' F2', ' U '), legacy = 'EdgeFlip2-B')
        self._add_myperm2('EAll2[RF>FR;UB>BU]', (' U ', ' R ', " U'", ' R2', ' U2', ' R ', ' L ', ' F ', " R'", " F'", " L'", ' U2', ' R '), legacy = 'EdgeFlip2-C')
        self._add_myperm2('EAll2[LB>BL;RF>FR]', (' L ', " F'", " U'", ' F ', ' U2', ' F2', " U'", " D'", " L'", ' U ', ' L ', ' D ', ' F2', " U'", " L'"), legacy = 'EdgeFlip2-D')
        self._add_myperm2('EAll4[DF>FD;FL>LF;RF>FR;UF>FU]', (" R'", ' F2', ' R ', " F'", ' U ', ' F ', ' L ', ' D ', ' F2', " D'", ' F ', " L'", " F'", " U'"), legacy = 'EdgeFlip4-E')
        self._add_myperm2('EAll4[BR>RB;FL>LF;LB>BL;RF>FR]', (' F ', ' L2', ' R ', ' F2', ' R ', " F'", ' U ', ' F ', ' L ', ' D ', ' F2', " D'", ' F ', " L'", " F'", " U'", ' L2', ' R2', " F'"), legacy = 'EdgeFlip4-F')
        self._add_myperm2('EAll4[DB>BD;FL>LF;RF>FR;UB>BU]', (' R2', " L'", ' B2', " L'", ' B ', " D'", " B'", " R'", " U'", ' B2', ' U ', " B'", ' R ', ' B ', ' D ', ' R2', ' L2'), legacy = 'EdgeFlip4-G')
        
        #(" F'", ' U2', " F'", ' R ', ' D2', " R'", ' F ', ' U2', " F'", ' R ', ' D2', " R'", ' F2')
        #(' F2', ' R ', ' D2', " R'", ' F ', ' U2', " F'", ' R ', ' D2', " R'", ' F ', ' U2', ' F ')

        self.myperms2['EdgeBlock3Cycle-PA'] = (' D2', ' B2', ' R ', ' B2', ' D2', ' F2', ' L ', ' F2')
        self.myperms2['EdgeBlock3Cycle-PB'] = (' F ', " U'", " R'", ' L ', ' F2', " L'", ' R ', " U'", " F'")
        self.myperms2['EdgeBlock3Cycle-PC'] = (" U "," F'"," U'"," B "," F'"," L "," F "," L'"," B'"," F ")
        self.myperms2['EdgeBlock3Cycle-PD'] = (" F "," B'"," R'"," F'"," R "," F'"," B "," U'"," F "," U ")
        self.myperms2['EdgeBlock3Cycle-PE'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-PA'])
        self.myperms2['EdgeBlock3Cycle-PF'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-PB'])
        self.myperms2['EdgeBlock3Cycle-PG'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-PC'])
        self.myperms2['EdgeBlock3Cycle-PH'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-PD'])

        self.myperms2['EdgeBlock3Cycle-UA'] = self.conjugate((" F ",),self.myperms2['EdgeBlock3Cycle-PA'])
        self.myperms2['EdgeBlock3Cycle-UB'] = (' F2', " U'", ' L ', " R'", ' F2', " L'", ' R ', " U'", ' F2')
        self.myperms2['EdgeBlock3Cycle-UC'] = self.conjugate((" F ",),self.myperms2['EdgeBlock3Cycle-PC'])
        self.myperms2['EdgeBlock3Cycle-UD'] = (" B'", " R'", ' F ', ' R ', ' B ', " F'", " U'", " F'", ' U ', ' F ')
        self.myperms2['EdgeBlock3Cycle-UE'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-UA'])
        self.myperms2['EdgeBlock3Cycle-UF'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-UB'])
        self.myperms2['EdgeBlock3Cycle-UG'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-UC'])
        self.myperms2['EdgeBlock3Cycle-UH'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-UD'])

        
        self.myperms2['EdgeBlock3Cycle-VA'] = self.conjugate((" F'",),self.myperms2['EdgeBlock3Cycle-PA'])
        self.myperms2['EdgeBlock3Cycle-VB'] = (" U'", " R'", ' L ', ' F2', " L'", ' R ', " U'")
        self.myperms2['EdgeBlock3Cycle-VC'] = (' F ', ' U ', ' F ', " U'", " F'", ' B ', ' L ', " F'", " L'", " B'")
        self.myperms2['EdgeBlock3Cycle-VD'] = self.conjugate((" F'",),self.myperms2['EdgeBlock3Cycle-PD'])
        self.myperms2['EdgeBlock3Cycle-VE'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-VA'])
        self.myperms2['EdgeBlock3Cycle-VF'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-VB'])
        self.myperms2['EdgeBlock3Cycle-VG'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-VC'])
        self.myperms2['EdgeBlock3Cycle-VH'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-VD'])
        
        self.myperms2['EdgeBlock3Cycle-RA'] = (" L'", ' D2', ' B2', ' U2', ' R ', ' B2', ' D2', ' F2', ' L2')
        self.myperms2['EdgeBlock3Cycle-RB'] = self.conjugate((" L2",),self.myperms2['EdgeBlock3Cycle-PB'])
        self.myperms2['EdgeBlock3Cycle-RC'] = self.conjugate((" L2",),self.myperms2['EdgeBlock3Cycle-PC'])
        self.myperms2['EdgeBlock3Cycle-RD'] = (' D2', " F'", ' B ', " R'", ' F ', ' R ', " B'", ' F ', " D'", " F'", " D'")
        self.myperms2['EdgeBlock3Cycle-RE'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-RA'])
        self.myperms2['EdgeBlock3Cycle-RF'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-RB'])
        self.myperms2['EdgeBlock3Cycle-RG'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-RC'])
        self.myperms2['EdgeBlock3Cycle-RH'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-RD'])

        self.myperms2['EdgeBlock3Cycle-NA'] = (' D ', ' L2', ' R2', " U'", " R'", ' U ', ' L2', ' R2', " D'", ' R ')
        self.myperms2['EdgeBlock3Cycle-NB'] = self.conjugate((" U "," R ",),self.myperms2['EdgeBlock3Cycle-PB'])
        self.myperms2['EdgeBlock3Cycle-NC'] = (" F'", ' U ', ' B2', ' F2', " D'", ' F ', ' D ', ' B2', ' F2', " U'")
        self.myperms2['EdgeBlock3Cycle-ND'] = (" F'", " R'", " F'", " R'", " F'", ' R ', ' F ', ' R ', ' F ', ' R ')

        self.myperms2['EdgeBlock3Cycle-NE'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-NA'])
        self.myperms2['EdgeBlock3Cycle-NF'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-NB'])
        self.myperms2['EdgeBlock3Cycle-NG'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-NC'])
        self.myperms2['EdgeBlock3Cycle-NH'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-ND'])

        self.myperms2['EdgeBlock3Cycle-QA'] = (' R ', " U'", " R'", " U'", " R'", " U'", ' R ', ' U ', ' R ', ' U ')
        self.myperms2['EdgeBlock3Cycle-QB'] = self.conjugate((" U'"," R ",),self.myperms2['EdgeBlock3Cycle-PB'])
        self.myperms2['EdgeBlock3Cycle-QC'] = (' D ', ' B ', " D'", ' U ', ' R2', " U'", ' D ', ' B ', " D'")
        self.myperms2['EdgeBlock3Cycle-QD'] = (' B ', ' R ', ' B ', ' R ', " B'", " R'", " B'", " R'", " B'", ' R ')
        self.myperms2['EdgeBlock3Cycle-QE'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-QA'])
        self.myperms2['EdgeBlock3Cycle-QF'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-QB'])
        self.myperms2['EdgeBlock3Cycle-QG'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-QC'])
        self.myperms2['EdgeBlock3Cycle-QH'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-QD'])

        self.myperms2['EdgeBlock3Cycle-YA'] = (" U'", " R'", " U'", " R'", " U'", " R'", ' U ', ' R ', ' U ', ' R ', ' U2')
        self.myperms2['EdgeBlock3Cycle-YB'] = self.conjugate((" R'"," U "," R ",),self.myperms2['EdgeBlock3Cycle-PC'])
        self.myperms2['EdgeBlock3Cycle-YC'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-YA'])
        self.myperms2['EdgeBlock3Cycle-YD'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-YB'])

        self.myperms2['EdgeBlock3Cycle-OA'] = (" L'", ' B ', " L'", ' R ', ' D2', ' L ', " R'", ' B ', ' L ')
        self.myperms2['EdgeBlock3Cycle-OB'] = self.conjugate((" R'"," F'",),self.myperms2['EdgeBlock3Cycle-PC'])
        self.myperms2['EdgeBlock3Cycle-OC'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-OA'])
        self.myperms2['EdgeBlock3Cycle-OD'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-OB'])

        self.myperms2['EdgeBlock3Cycle-IA'] = (' L ', " B'", ' F ', " U'", ' R2', ' U ', " F'", ' B ', " L'", ' U2')
        self.myperms2['EdgeBlock3Cycle-IB'] = (' U2', ' F ', " B'", ' R2', ' B ', " F'")
        self.myperms2['EdgeBlock3Cycle-IC'] = (' R2', " D'", ' B ', " F'", ' R ', ' U2', " R'", ' F ', " B'", ' D ')
        self.myperms2['EdgeBlock3Cycle-ID'] = (" D'"," B'"," R'"," F'"," R "," F'"," B "," U'"," F "," U "," F "," D ")
        self.myperms2['EdgeBlock3Cycle-IE'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-IA'])
        self.myperms2['EdgeBlock3Cycle-IF'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-IB'])
        self.myperms2['EdgeBlock3Cycle-IG'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-IC'])
        self.myperms2['EdgeBlock3Cycle-IH'] = self.invert_moves(self.myperms2['EdgeBlock3Cycle-ID'])

        self.myperms2['EdgeCornerSwap-X-A'] = (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F2', ' R ')
        self.myperms2['EdgeCornerSwap-X-B'] = (" L "," R'",' F ', " R'", " F'", ' R2', " U'", " R'", " F'", " U'", ' F ', ' R ', ' U ', " L'")
        self.myperms2['EdgeCornerSwap-X-C'] = self.conjugate((" F2"," R "),self.myperms2['EdgeCornerSwap-X-A'])
        self.myperms2['EdgeCornerSwap-X-D'] = self.conjugate((" F "," R "),self.myperms2['EdgeCornerSwap-X-A'])
        self.myperms2['EdgeCornerSwap-X-E'] = self.conjugate((" R ",),self.myperms2['EdgeCornerSwap-X-A'])
        self.myperms2['EdgeCornerSwap-X-F'] = self.conjugate((" F'"," R "),self.myperms2['EdgeCornerSwap-X-A'])
        self.myperms2['EdgeCornerSwap-X-G'] = self.conjugate((" R2",),self.myperms2['EdgeCornerSwap-X-A'])
        self.myperms2['EdgeCornerSwap-X-H'] = (' F2', " U'", " F'", ' U ', ' F ', ' R ', " U'", " R'", " F'", ' L ', " F'", " L'")

        
        self.myperms2['EdgeCornerSwap-Y-A'] = (" R "," U "," R'"," U'"," R'"," F "," R2"," U'"," R'"," U'"," R "," U "," R'"," F'")
        self.myperms2['EdgeCornerSwap-Y-B'] = (' F ', " R'", " F'", ' R2', " U'", " R'", " F'", " U'", ' F ', ' R ', ' U ', " R'")
        self.myperms2['EdgeCornerSwap-Y-C'] = self.conjugate((" F2"," R "),self.myperms2['EdgeCornerSwap-Y-A'])
        self.myperms2['EdgeCornerSwap-Y-D'] = self.conjugate((" F "," R "),self.myperms2['EdgeCornerSwap-Y-A'])
        self.myperms2['EdgeCornerSwap-Y-E'] = self.conjugate((" R ",),self.myperms2['EdgeCornerSwap-Y-A'])
        self.myperms2['EdgeCornerSwap-Y-F'] = self.conjugate((" F'"," R "),self.myperms2['EdgeCornerSwap-Y-A'])
        self.myperms2['EdgeCornerSwap-Y-G'] = self.conjugate((" R2",),self.myperms2['EdgeCornerSwap-Y-A'])
        self.myperms2['EdgeCornerSwap-Y-H'] = self.conjugate((" R'"," U'"," F "," U "),self.myperms2['EdgeCornerSwap-Y-A'])

        self.myperms2['EdgeCornerSwap-Z-A'] = (' D2', " R'", ' B ', ' R ', ' D2', ' U2', " B'", ' U2', ' B ', ' U2', ' L2', ' F ', ' L ', " F'", ' L ')
        self.myperms2['EdgeCornerSwap-Z-B'] = self.conjugate((" F'"," U "," L'"),self.myperms2['EdgeCornerSwap-Z-A'])
        self.myperms2['EdgeCornerSwap-Z-C'] = self.conjugate((" U2"," L "),self.myperms2['EdgeCornerSwap-Z-A'])
        self.myperms2['EdgeCornerSwap-Z-D'] = self.conjugate((" U "," L "),self.myperms2['EdgeCornerSwap-Z-A'])
        self.myperms2['EdgeCornerSwap-Z-E'] = self.conjugate((" L ",),self.myperms2['EdgeCornerSwap-Z-A'])
        self.myperms2['EdgeCornerSwap-Z-F'] = self.conjugate((" U'"," L "),self.myperms2['EdgeCornerSwap-Z-A'])
        self.myperms2['EdgeCornerSwap-Z-G'] = self.conjugate((" L2",),self.myperms2['EdgeCornerSwap-Z-A'])
        self.myperms2['EdgeCornerSwap-Z-H'] = self.conjugate((" F "," U "," L'"),self.myperms2['EdgeCornerSwap-Z-A'])   
        

        self.myperms2['CornerEdgeBlockSwap-K00-'] = (' D2', ' L ', ' B ', " L'", ' D2', ' R ', " F'", ' R ', ' F ', ' R2', ' F2', ' D2', ' F ', ' U2', " F'", ' D2', ' F ', ' U2', ' F ')
        self.myperms2['CornerEdgeBlockSwap-K01-'] = (' R2', " D'", ' R ', ' U2', " R'", ' D ', ' R ', ' U2', ' R ', " U'", " L'", ' U ', ' R2', " U'", ' L ', ' U ', ' R2', ' U ', ' R ', " U'", ' R ', ' U ', ' R ', " U'", ' R2')
        self.myperms2['CornerEdgeBlockSwap-K02-'] = (" B'", ' R2', " B'", ' D2', ' L2', ' F2', ' L2', ' D2', ' B ', ' R2', ' D2', " L'", " F'", ' L ', ' D2', " R'", ' B ', ' R ')
        self.myperms2['CornerEdgeBlockSwap-K03-'] = (' B2', ' D ', " B'", ' U2', ' B ', " D'", " B'", ' U2', ' L ', " B'", " U'", ' L ', ' F ', ' U ', ' L2', " U'", ' L ', ' F ', ' L ', " F'", ' L2', ' U ')
        self.myperms2['CornerEdgeBlockSwap-K04-'] = (' U2', ' F ', ' L2', " B'", ' U2', ' B ', ' U2', ' F2', ' L2', " F'", ' L2', ' F ', ' L2', ' U2', ' F2', ' U2', ' B ', ' L2', ' U2', ' R2', ' F ', ' D2', ' R2')
        self.myperms2['CornerEdgeBlockSwap-K05-'] = (' D ', ' R ', " D'", ' U ', ' F2', " U'", ' D ', " R'", ' D2', ' B2', ' D2', " B'", ' D2', ' B ', ' D2', " B'", ' D2', ' R2', " B'", ' R2', " B'", ' D2', ' B2', " D'")
        self.myperms2['CornerEdgeBlockSwap-K06-'] = (' U2', " B'", ' L2', ' F ', ' U2', " F'", ' U2', ' B2', ' L2', ' B ', ' L2', " B'", ' L2', ' U2', ' B2', ' D2', " B'", ' D2', ' R2', ' U2', " F'", ' U2', ' R2')
        self.myperms2['CornerEdgeBlockSwap-K07-'] = (' D ', " R'", ' D ', " U'", ' B2', " D'", ' U ', ' R ', ' D2', ' B2', ' D2', " B'", ' D2', ' B ', ' D2', " B'", ' D2', ' R2', " B'", ' R2', " B'", ' D2', ' B2', " D'")
        self.myperms2['CornerEdgeBlockSwap-K08-'] = (' F2', ' D2', ' B ', ' U2', " B'", ' F ', ' R2', ' L2', ' F ', ' L2', ' D2', ' F ', ' D2', " F'", ' D2', ' F ', ' D2', ' F2', ' D2', ' L2')
        self.myperms2['CornerEdgeBlockSwap-K09-'] = self.conjugate((" L "," R'"),self.myperms2['EdgeCornerSwap-Z-H'])


        
        self.myperms2['CornerEdgeBlockSwap-J00-'] = (' L2', ' U2', ' F ', ' U ', " F'", ' U ', ' L2', " D'", ' B ', ' D ')
        self.myperms2['CornerEdgeBlockSwap-J01-'] = (' D2', " R'", " U'", ' R ', ' D2', " R'", ' U ', ' F2', " L'", " U'", ' L ', ' F2', " R'", ' D ', " R'", " D'", " R'")
        self.myperms2['CornerEdgeBlockSwap-J02-'] = (' U ', " B'", ' U ', ' R2', " D'", ' F ', ' D ', ' R2', ' F ', ' U2', ' B ', ' U2', " F'", ' U2')
        self.myperms2['CornerEdgeBlockSwap-J03-'] = (' U2', ' L ', ' D ', " L'", ' U2', ' L ', " D'", ' L2', ' F2', ' R ', ' U ', " R'", ' F2', ' L ', " D'", ' L ', ' D ', " L'")
        self.myperms2['CornerEdgeBlockSwap-J04-'] = (' L2', ' U2', " F'", ' L2', ' D2', ' R2', ' B2', ' D2', ' L2', " U'", " F'", ' U ', ' L2', " D'", ' B ', ' D ')
        self.myperms2['CornerEdgeBlockSwap-J05-'] = (' R2', " D'", " F'", ' D ', ' F ', ' D ', " R'", ' D2', ' F ', ' D ', ' F ', " D'", " F'", ' D ', ' B2', ' L2', ' U ', ' L2', ' B2', ' R2', ' D ', ' R ')
        self.myperms2['CornerEdgeBlockSwap-J06-'] = (" D'", " B'", ' D ', ' L2', " U'", ' F ', ' U ', ' L2', ' D2', ' B ', ' D2', ' U2', ' L2', ' D2', ' F ', ' R2', ' D2', ' L2', ' B ')
        self.myperms2['CornerEdgeBlockSwap-J07-'] = (" L'", ' U2', ' L2', ' R2', " D'", ' L2', ' D ', ' L2', ' R2', " U'", ' L2', " U'", ' L ', ' R ', " D'", ' F ', ' D ', " F'", " D'", " F'", ' D2', ' R ', " D'", " F'", " D'", ' F ', ' D ', ' R2')
        self.myperms2['CornerEdgeBlockSwap-J08-'] = self.conjugate((" F "," B "),self.myperms2['EdgeCornerSwap-Z-G'])
        self.myperms2['CornerEdgeBlockSwap-J09-'] = self.conjugate((" L "," R "),self.myperms2['EdgeCornerSwap-Z-H'])        
        self.myperms2['CornerEdgeBlockSwap-J10-'] = self.conjugate((" F2",),self.myperms2['EdgeCornerSwap-Z-G'])
        self.myperms2['CornerEdgeBlockSwap-J11-'] = self.conjugate((" L2",),self.myperms2['EdgeCornerSwap-Z-H'])

        self.myperms2['CornerEdgeBlockSwap-JX'] = (' B2', " L'", ' B2', ' D2', " F'", " R'", ' F ', ' D2', " B'", ' L ', ' B ')
        self.myperms2['CornerEdgeBlockSwap-JY'] = (' F2', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F ', " L'", ' F ', ' R ', " F'", ' L ')
        self.myperms2['CornerEdgeBlockSwap-JZ'] = (' B2', ' U2', ' B ', ' U2', " B'", ' U2', ' B2', ' L2', " F'", " L'", ' F ', " L'", ' U2', ' R ', " B'", " R'", ' U2')     
        

        self.myperms2['CornerEdgeBlockSwap-Super00-'] = (" U'", " F'", ' U ', ' B ', " U'", ' F ', ' U ', " B'", ' L ', ' D ', " L'", ' U ', ' L ', " D'", ' L2', ' U ', ' L ', " U'", ' F ', ' R ', ' U ', " R'", " F'", " L'", " U'", ' L ')
        self.myperms2['CornerEdgeBlockSwap-B00-'] = (" L2"," F2"," U2"," L'"," U2"," L2"," F2"," L'"," U2"," L2"," U2"," F2"," L'"," F2")
        self.myperms2['CornerEdgeBlockSwap-F00-'] = (' L ', " F'", ' D2', ' B ', " R'", " B'", ' D2', ' F2', " L'", " F'")
        self.myperms2['CornerEdgeBlockSwap-B01-'] = (' R2', ' B2', ' L2', ' D2', ' F2', ' L2', ' B2', ' R2', ' F2', ' U ', ' F2', ' U2', ' R2', ' U ', ' F2', ' U2', ' F2', ' R2', ' U ', ' R2')
        self.myperms2['CornerEdgeBlockSwap-F01-'] = (" F "," R2"," F "," R "," U "," R'"," U'"," R'"," F "," R2"," U'"," R'"," U'"," R "," U "," R'"," F2"," R2"," F'")
        
        self.myperms2['CornerEdgeBlockSwap-Super05-'] = (' D2', ' B2', ' D2', ' F2', ' U2', ' F2', ' B2', " R'", ' B2', ' D2', ' R2', ' D2', " R'", ' B2', ' R2', ' D2', ' R ', ' D2', ' B2', ' R2')
        self.myperms2['CornerEdgeBlockSwap-Super06-'] = (' D2', ' F2', ' B2', ' U2', ' F2', " R'", ' B ', ' U2', " F'", ' L ', ' F ', ' U2', ' B2', ' R ', " B'")
        
        

    def _register_myperms2_center_general(self):
        """4x4以上で使うCenter系・Bar系の手順を登録する。"""
        # 命名メモ:
        # - X-Center / Plus-Center / Oblique-Center は center の配置 family。
        # - Adjacent3Center / Line3Center は 3面の center 配置 family。
        # - OuterCenterBar / MidCenterBar は center の bar を動かす family。
        if self.size >= 4:
            self._add_myperm2('CtrX6p[3x2][D@2L.2F>D@2R.2B>U@2R.2F;D@2R.2F>U@2L.2F>U@2R.2B]', ("2R2","2F2","2R2","2F2"), legacy = 'X-Center-XA')
            self._add_myperm2('CtrX6p[3x2][D@2L.2F>D@2R.2B>U@2R.2B;D@2R.2F>U@2R.2F>U@2L.2B]', (" U ","2R2","2F2","2R2","2F2"," U'"), legacy = 'X-Center-XB')
            self._add_myperm2('CtrX6p[3x2][D@2L.2F>D@2R.2B>U@2L.2B;D@2R.2F>U@2R.2B>U@2L.2F]', (" U2","2R2","2F2","2R2","2F2"," U2"), legacy = 'X-Center-XC')
            self._add_myperm2('CtrX6p[3x2][D@2L.2B>D@2L.2F>U@2L.2F;D@2R.2B>D@2R.2F>U@2R.2B]', ('2F2', " D'", '2R2', '2F2', '2R2', '2F2', ' D ', '2F2'), legacy = 'X-Center-WA')
            self._add_myperm2('CtrX6p[3x2][D@2L.2B>U@2L.2F>D@2L.2F;D@2R.2B>U@2R.2B>D@2R.2F]', ('2B2', " D'", '2L2', '2B2', '2L2', '2B2', ' D ', '2B2'), legacy = 'X-Center-WB')
            self._add_myperm2('CtrX6p[3x2][D@2L.2B>D@2L.2F>U@2L.2F;D@2R.2F>U@2R.2F>U@2R.2B]', ('2F2', ' U ', '2L2', '2F2', '2L2', '2F2', " U'", '2F2'), legacy = 'X-Center-WC')
            self._add_myperm2('CtrX6p[3x2][D@2L.2B>U@2L.2B>U@2L.2F;D@2R.2B>U@2R.2B>D@2R.2F]', ('2B2', ' U ', '2R2', '2B2', '2R2', '2B2', " U'", '2B2'), legacy = 'X-Center-WD')
            self._add_myperm2('CtrX4s[B@2R.2U<>F@2R.2U;L@2F.2D<>R@2F.2D]~v01', ("2F'", '2U ', '2F ', "2U'", '2F ', "2R'", "2F'", '2R '), legacy = 'X-Center-VA')
            self._add_myperm2('CtrX4s[B@2R.2U<>F@2R.2U;L@2B.2D<>R@2B.2D]', ("2F'", '2U ', "2F'", "2U'", '2F ', "2R'", '2F ', '2R '), legacy = 'X-Center-VB')
            self._add_myperm2('CtrX4s[B@2R.2U<>F@2R.2U;L@2F.2D<>R@2F.2D]~v02', ("2R'", '2F ', '2R ', "2F'", '2U ', "2F'", "2U'", '2F '), legacy = 'X-Center-VC')
            self._add_myperm2('CtrX6p[3x2][D@2L.2F>U@2R.2F>U@2R.2B;D@2R.2B>U@2L.2F>D@2R.2F]', ('2F2', ' U ', '2R2', " U'", '2R2', '2F2', '2R2', ' U ', '2R2', " U'"), legacy = 'X-Center-UA')
            self._add_myperm2('CtrX4s[D@2L.2F<>U@2L.2B;D@2R.2B<>U@2R.2F]', ('2R2', '2F2', '2R2', '2F2', ' U2', '2R2', '2F2', '2R2', '2F2', ' U2'), legacy = 'X-Center-UB')


            

            self._add_myperm2('CtrX10p[5x2]~v02', ("2U2","2R2","2U'","2R2","2U'","2R2","2U'","2R2","2U "), legacy = 'X-Center-8')
            
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>L@2F.2U>U@2R.2F;D@2R.2F>F@2R.2U>R@2F.2U]', ("2R ","2U ","2R'","2U'"), legacy = 'X-Center-6A')
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>R@2F.2D>U@2R.2F;D@2R.2F>F@2R.2U>L@2F.2D]', ("2R ","2U'","2R'","2U "), legacy = 'X-Center-6B')
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>D@2R.2F>F@2R.2U;B@2R.2U>F@2L.2U>U@2R.2F]', ("2R ","2U2","2R'","2U2"), legacy = 'X-Center-6C')
            self._add_myperm2('CtrX6p[3x2][B@2R.2D>F@2R.2U>L@2F.2D;B@2R.2U>R@2F.2D>F@2R.2D]', ("2R2","2U'","2R2","2U "), legacy = 'X-Center-6D')
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>L@2F.2D>D@2R.2F;F@2R.2U>R@2F.2D>U@2R.2F]', ("2U'","2R ","2U ","2R2","2F ","2R ","2F'"), legacy = 'X-Center-6E')
            self._add_myperm2('CtrX10p[5x2]~v01', ('2B ', '2D2', '2B ', '2D2', "2L'", "2B'", '2L ', '2U ', "2B'", "2U'"), legacy = 'X-Center-6F')
            self._add_myperm2('CtrX6p[3x2][B@2R.2D>R@2F.2U>D@2R.2F;F@2R.2D>L@2F.2U>U@2R.2F]', ("2R ","2U ","2R ","2U'","2R2"), legacy = 'X-Center-6G')
            self._add_myperm2('CtrX6p[3x2][B@2R.2D>L@2F.2D>D@2R.2F;F@2R.2D>R@2F.2D>U@2R.2F]', ("2R ","2U'","2R ","2U ","2R2"), legacy = 'X-Center-6H')
            self._add_myperm2('CtrX6p[3x2][B@2R.2D>D@2R.2F>R@2F.2U;F@2R.2D>U@2R.2F>L@2F.2U]', ("2R2","2U ","2R'","2U'","2R'"), legacy = 'X-Center-6I')
            self._add_myperm2('CtrX6p[3x2][B@2R.2D>D@2R.2F>L@2F.2D;F@2R.2D>U@2R.2F>R@2F.2D]', ("2R2","2U'","2R'","2U ","2R'"), legacy = 'X-Center-6J')
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>F@2R.2U>R@2F.2U;B@2R.2U>L@2F.2U>F@2L.2U]', ("2R2","2U ","2R2","2U ","2R2","2U2","2R2"), legacy = 'X-Center-6K')
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>L@2F.2D>F@2R.2U;B@2R.2U>F@2L.2U>R@2F.2D]', ("2R2","2U2","2R2","2U ","2R2","2U ","2R2"), legacy = 'X-Center-6L')
            self._add_myperm2('CtrX6p[3x2][B@2R.2D>R@2F.2U>L@2F.2D;F@2R.2D>L@2F.2U>R@2F.2D]', ("2U ","2R2","2U2","2R2","2U "), legacy = 'X-Center-6M')
            self._add_myperm2('CtrX6p[3x2][B@2R.2D>R@2F.2D>L@2B.2D;F@2R.2D>L@2F.2D>R@2B.2D]', ("2F ","2R ","2F2","2R'","2F "), legacy = 'X-Center-6N')
            self._add_myperm2('CtrX6p[3x2][B@2R.2D>L@2B.2D>R@2F.2D;F@2R.2D>R@2B.2D>L@2F.2D]', ("2F'","2R ","2F2","2R'","2F'"), legacy = 'X-Center-6O')

            

            self._add_myperm2('CtrX6p[3x2][B@2L.2U>B@2R.2D>D@2R.2F;F@2L.2U>F@2R.2D>U@2R.2F]', ("2R2","2U2","2R'","2U2","2R'"), legacy = 'X-Center-4A')
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>D@2R.2F>B@2R.2D;F@2L.2U>U@2R.2F>F@2R.2D]', ("2R ","2U2","2R ","2U2","2R2"), legacy = 'X-Center-4B')
            self._add_myperm2('CtrX6p[3x2][B@2R.2D>B@2R.2U>D@2R.2F;F@2R.2D>F@2R.2U>U@2R.2F]', ("2U ","2F ","2R ","2F'","2R'","2U'"), legacy = 'X-Center-4C')
            self._add_myperm2('CtrX6p[3x2][B@2R.2D>D@2R.2B>D@2L.2F;F@2R.2D>U@2R.2B>U@2L.2F]~v01', ('2F ', "2D'", '2F2', '2D ', "2F'", '2R2', '2F2', '2R2'), legacy = 'X-Center-4D')
            self._add_myperm2('CtrX8s~v01', ("2D'", "2F'", '2D2', '2F2', '2L ', "2F'", "2L'", "2D'"), legacy = 'X-Center-4D01')
            self._add_myperm2('CtrX6p[3x2][B@2R.2D>D@2R.2B>D@2L.2F;F@2R.2D>U@2R.2B>U@2L.2F]~v02', ("2F'", '2D2', "2B'", '2D2', '2F ', '2D ', '2B ', "2D'"), legacy = 'X-Center-4D02')
            self._add_myperm2('CtrX6p[3x2][B@2R.2D>D@2R.2B>D@2L.2F;F@2R.2D>U@2R.2B>U@2L.2F]~v03', ("2D'", "2F'", '2D ', "2F'", '2R2', '2F ', '2R2', '2F '), legacy = 'X-Center-4D03')
            self._add_myperm2('CtrX6p[3x2][B@2L.2D>B@2R.2D>D@2L.2F;F@2L.2D>F@2R.2D>U@2L.2F]', ('2F ', "2L'", "2F'", '2L ', "2F'", "2D'", '2F ', '2D '), legacy = 'X-Center-4D04')
            self._add_myperm2('CtrX8s~v02', ('2F ', "2D'", '2F2', '2D ', "2F'", '2L2', '2F2', '2L2'), legacy = 'X-Center-4D05')
            self._add_myperm2('CtrX6p[3x2][B@2R.2D>D@2L.2B>D@2R.2F;F@2R.2D>U@2L.2B>U@2R.2B]', ("2L'"," U ","2R'","2D2","2R "," U'","2D2","2L "), legacy = 'X-Center-4XA')
            self._add_myperm2('CtrX6p[3x2][B@2R.2D>D@2L.2B>D@2R.2B;F@2R.2D>U@2L.2B>U@2R.2F]', ("2L'"," D'","2R'","2D2","2R "," D ","2D2","2L "), legacy = 'X-Center-4XB')
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>U@2R.2B>U@2L.2F;D@2R.2B>D@2R.2F>F@2R.2U]', ("2B "," U'","2R ","2U ","2R'"," U ","2U'","2B'"), legacy = 'X-Center-4YA')
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>U@2R.2B>U@2R.2F;D@2L.2F>F@2R.2U>D@2R.2B]', ("2B "," D ","2R ","2U ","2R'"," D'","2U'","2B'"), legacy = 'X-Center-4YB')
            self._add_myperm2('CtrX6p[3x2][B@2L.2D>B@2R.2U>U@2L.2B;D@2R.2F>F@2L.2D>F@2R.2U]', (' D2', '2L ', '2U2', '2L ', '2U2', '2L2', ' D2'), legacy = 'X-Center-4YC')
            self._add_myperm2('CtrX6p[3x2][B@2L.2D>B@2R.2U>U@2R.2F;D@2L.2B>F@2L.2D>F@2R.2U]', (' U2', '2L ', '2U2', '2L ', '2U2', '2L2', ' U2'), legacy = 'X-Center-4YD')
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>D@2R.2F>B@2R.2D;F@2L.2U>U@2L.2B>F@2R.2D]', (" U2","2R ","2U2","2R'"," U2","2R2","2U2","2R2"), legacy = 'X-Center-4ZA')
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>D@2L.2B>B@2R.2D;F@2L.2U>U@2R.2F>F@2R.2D]', (" D2","2R ","2U2","2R'"," D2","2R2","2U2","2R2"), legacy = 'X-Center-4ZB')
            self._add_myperm2('CtrX8s~v03', (' U2', '2R ', '2U2', "2R'", ' U2', '2L2', '2U2', '2L2'), legacy = 'X-Center-4ZC')
            self._add_myperm2('CtrX8s~v04', (' D2', '2R ', '2U2', "2R'", ' D2', '2L2', '2U2', '2L2'), legacy = 'X-Center-4ZD')
            self._add_myperm2('CtrX6p[3x2][B@2R.2U>U@2R.2F>U@2L.2B;D@2L.2B>F@2R.2U>F@2L.2U]', ('2L ', '2U2', ' D ', '2R ', '2U2', "2R'", " D'", "2L'"), legacy = 'X-Center-4WA')
            self._add_myperm2('CtrX6p[3x2][B@2L.2U>U@2L.2B>B@2R.2U;D@2L.2B>F@2R.2U>D@2R.2F]', ('2L ', '2U2', " U'", '2R ', '2U2', "2R'", " U ", "2L'"), legacy = 'X-Center-4WB')



            self._add_myperm2('CtrX3[D@2R.2F>U@2L.2F>U@2L.2B]', (" U ", '2R2', " U'", '2F2', ' U ', '2R2', " U'", '2F2'), legacy = 'X-Center-Opp2X')
            self._add_myperm2('CtrX3[D@2R.2F>U@2R.2B>U@2L.2B]~v01', (" D'", '2L2', ' D ', '2R2', " D'", '2L2', ' D ', '2R2'), legacy = 'X-Center-Opp2X01')
            self._add_myperm2('CtrX3[D@2L.2F>U@2R.2B>D@2R.2F]', ('2R2', '2F ', ' L ', "2F'", '2R2', '2F ', " L'", "2F'"), legacy = 'X-Center-Opp2X02')
            self._add_myperm2('CtrX3[D@2R.2F>U@2R.2B>U@2L.2B]~v02', ('2R2', "2B'", ' R ', '2B ', '2R2', "2B'", " R'", '2B '), legacy = 'X-Center-Opp2X03')
            self._add_myperm2('CtrX3[D@2R.2F>U@2L.2B>U@2R.2B]~v01', ('2R2', " U'", '2F2', ' U ', '2R2', " U'", '2F2', ' U '), legacy = 'X-Center-Opp2Y')
            self._add_myperm2('CtrX3[D@2R.2F>U@2L.2B>U@2R.2B]~v02', ('2R2', " D'", '2L2', ' D ', '2R2', " D'", '2L2', ' D '), legacy = 'X-Center-Opp2Y01')
            self._add_myperm2('CtrX3[D@2L.2F>D@2R.2F>U@2R.2B]', ('2F ', ' L ', "2F'", '2R2', '2F ', " L'", "2F'", '2R2'), legacy = 'X-Center-Opp2Y02')
            self._add_myperm2('CtrX3[D@2R.2F>U@2L.2B>U@2R.2B]~v03', ("2B'", ' R ', '2B ', '2R2', "2B'", " R'", '2B ', '2R2'), legacy = 'X-Center-Opp2Y03')
            self._add_myperm2('CtrX3[D@2R.2F>U@2R.2B>U@2R.2F]', (" U'", '2R2', " U'", '2F2', ' U ', '2R2', " U'", '2F2', ' U2'), legacy = 'X-Center-Opp2Z')
            self._add_myperm2('CtrX3[D@2L.2F>D@2R.2F>U@2R.2F]', (" U'", '2R2', " U'", '2L2', ' U ', '2R2', " U'", '2L2', ' U2'), legacy = 'X-Center-Opp2Z01')
            self._add_myperm2('CtrX8s~v05', ('2R2', " D'", '2F ', '2B ', '2R2', "2F'", "2B'", '2R2', ' D ', '2R2'), legacy = 'X-Center-Opp2Z02')
            self._add_myperm2('CtrX8s~v06', ('2R2', " D'", '2R2', '2B ', '2F ', '2R2', "2B'", "2F'", ' D ', '2R2'), legacy = 'X-Center-Opp2Z03')
            self._add_myperm2('CtrX8s~v07', ('2F2', ' D ', '2R ', '2L ', '2F2', "2R'", "2L'", '2F2', " D'", '2F2'), legacy = 'X-Center-Opp2Z04')
            self._add_myperm2('CtrX8s~v08', ('2F2', ' D ', '2F2', '2L ', '2R ', '2F2', "2L'", "2R'", " D'", '2F2'), legacy = 'X-Center-Opp2Z05')
            self._add_myperm2('CtrX6p[2+4][D@2L.2F<>D@2R.2B;D@2R.2F>U@2L.2B>U@2L.2F>U@2R.2F]', ('2B2', " D'", '2F2', ' D ', '2B2', " D'", '2L2', '2F2', '2L2', ' D '), legacy = 'X-Center-Opp2Z06')
            


            self._add_myperm2('CtrX3[F@2R.2U>U@2R.2F>U@2L.2B]~v01', ("2F'", '2U ', '2F ', " U2", "2F'", "2U'", '2F ', " U2"), legacy = 'X-Center-InOut-Diagonal')
            self._add_myperm2('CtrX3[F@2L.2U>U@2R.2F>U@2L.2B]', ('2R ', '2U2', "2R'", " U2", '2R ', '2U2', "2R'", " U2"), legacy = 'X-Center-InOut-Vertical')
            self._add_myperm2('CtrX3[B@2R.2U>D@2R.2F>D@2L.2B]~v01', ("2F'", '2U ', '2F ', " D2", "2F'", "2U'", '2F ', " D2"), legacy = 'X-Center-InOut-Diagonal01')
            self._add_myperm2('CtrX3[B@2L.2U>D@2R.2F>D@2L.2B]', ('2R ', '2U2', "2R'", " D2", '2R ', '2U2', "2R'", " D2"), legacy = 'X-Center-InOut-Vertical01')
            self._add_myperm2('CtrX3[F@2L.2D>F@2R.2D>U@2R.2F]', ("2R "," U ","2L'"," U'","2R'"," U ","2L "," U'"), legacy = 'X-Center-InOut-Vertical02')
            self._add_myperm2('CtrX3[F@2L.2U>U@2L.2B>F@2R.2U]', ("2L'"," U ","2R "," U'","2L "," U ","2R'"," U'"), legacy = 'X-Center-InOut-Vertical03')
            self._add_myperm2('CtrX3[F@2L.2D>U@2R.2F>F@2R.2D]', (' U ', "2L'", " U'", '2R ', ' U ', '2L ', " U'", "2R'"), legacy = 'X-Center-InOut-Diagonal02')
            self._add_myperm2('CtrX3[F@2L.2U>F@2R.2U>U@2L.2B]', (' U ', "2R ", " U'", "2L'", ' U ', "2R'", " U'", "2L "), legacy = 'X-Center-InOut-Diagonal03')


            
            self._add_myperm2('CtrX3[F@2L.2U>U@2L.2B>U@2R.2F]', (' U2', '2R ', '2U2', "2R'", " U2", '2R ', '2U2', "2R'"), legacy = 'X-Center-InIn-Diagonal')
            self._add_myperm2('CtrX3[B@2L.2U>D@2L.2B>D@2R.2F]', (" D2", '2R ', '2U2', "2R'", " D2", '2R ', '2U2', "2R'"), legacy = 'X-Center-OutOut-Diagonal')
            self._add_myperm2('CtrX3[F@2R.2U>U@2L.2B>U@2R.2F]', (" U2", "2F'", '2U ', '2F ', " U2", "2F'", "2U'", '2F '), legacy = 'X-Center-InIn-Vertical')
            self._add_myperm2('CtrX3[B@2R.2U>D@2L.2B>D@2R.2F]', (" D2", "2F'", '2U ', '2F ', " D2", "2F'", "2U'", '2F '), legacy = 'X-Center-OutOut-Vertical')

            self._add_myperm2('CtrX3[F@2R.2U>U@2R.2F>U@2L.2B]~v02', (' U2', "2B'", "2U'", '2B ', ' U2', "2B'", '2U ', '2B '), legacy = 'X-Center-InOut-Diagonal04')
            self._add_myperm2('CtrX3[B@2R.2U>D@2R.2F>D@2L.2B]~v02', (' D2', "2B'", "2U'", '2B ', ' D2', "2B'", '2U ', '2B '), legacy = 'X-Center-InOut-Diagonal05')

            self._add_myperm2('CtrX3[F@2R.2U>U@2R.2F>R@2F.2D]', (" F ","2D "," R2","2D'","2R'","2D "," R2","2D'","2R "," F'"), legacy = 'X-Center-Adjacent3Center-AAA')
            self._add_myperm2('CtrX3[F@2L.2D>U@2L.2B>R@2B.2U]', (" F ","2U'"," R2","2U ","2L ","2U'"," R2","2U ","2L'"," F'"), legacy = 'X-Center-Adjacent3Center-CCC')
            self._add_myperm2('CtrX3[F@2L.2U>U@2R.2B>R@2B.2D]', (" F'","2D'","2L'"," U2","2L ","2D ","2L'"," U2","2L "," F "), legacy = 'X-Center-Adjacent3Center-BBB')
            self._add_myperm2('CtrX3[F@2R.2D>U@2L.2F>R@2F.2U]', (" F'","2U ","2R "," U2","2R'","2U'","2R "," U2","2R'"," F "), legacy = 'X-Center-Adjacent3Center-DDD')

            self._add_myperm2('CtrX3[F@2L.2D>U@2R.2F>R@2F.2D]', (" F'","2D "," R2","2D'","2R'","2D "," R2","2D'","2R "," F "), legacy = 'X-Center-Adjacent3Center-AAC')
            self._add_myperm2('CtrX3[F@2R.2U>U@2L.2B>R@2B.2U]', (" F'","2U'"," R2","2U ","2L ","2U'"," R2","2U ","2L'"," F "), legacy = 'X-Center-Adjacent3Center-CCA')
            self._add_myperm2('CtrX3[F@2R.2D>U@2R.2B>R@2B.2D]', (" F ","2D'","2L'"," U2","2L ","2D ","2L'"," U2","2L "," F'"), legacy = 'X-Center-Adjacent3Center-BBD')
            self._add_myperm2('CtrX3[F@2L.2U>U@2L.2F>R@2F.2U]', (" F ","2U ","2R "," U2","2R'","2U'","2R "," U2","2R'"," F'"), legacy = 'X-Center-Adjacent3Center-DDB')
            
            self._add_myperm2('CtrX3[F@2R.2D>U@2R.2F>R@2F.2D]', ("2D "," R2","2D'","2R'","2D "," R2","2D'","2R "), legacy = 'X-Center-Adjacent3Center-AAD')
            self._add_myperm2('CtrX3[F@2L.2U>U@2L.2B>R@2B.2U]', ("2U'"," R2","2U ","2L ","2U'"," R2","2U ","2L'"), legacy = 'X-Center-Adjacent3Center-CCB')
            self._add_myperm2('CtrX3[F@2L.2D>U@2R.2B>R@2B.2D]', ("2D'","2L'"," U2","2L ","2D ","2L'"," U2","2L "), legacy = 'X-Center-Adjacent3Center-BBC')
            self._add_myperm2('CtrX3[F@2R.2U>U@2L.2F>R@2F.2U]', ("2U ","2R "," U2","2R'","2U'","2R "," U2","2R'"), legacy = 'X-Center-Adjacent3Center-DDA')

            self._add_myperm2('CtrX3[F@2L.2U>U@2R.2F>R@2F.2D]', ('2U ', "2L'", ' U2', '2L ', "2U'", "2L'", ' U2', '2L '), legacy = 'X-Center-Adjacent3Center-AAB')
            self._add_myperm2('CtrX3[F@2R.2D>U@2L.2B>R@2B.2U]', ("2D'", '2R ', ' U2', "2R'", '2D ', '2R ', ' U2', "2R'"), legacy = 'X-Center-Adjacent3Center-CCD')
            self._add_myperm2('CtrX3[F@2L.2D>U@2L.2F>R@2F.2U]', ('2D ', ' R2', "2D'", '2L ', '2D ', ' R2', "2D'", "2L'"), legacy = 'X-Center-Adjacent3Center-DDC')
            self._add_myperm2('CtrX3[F@2R.2U>U@2R.2B>R@2B.2D]', ("2U'", ' R2', '2U ', "2R'", "2U'", ' R2', '2U ', '2R '), legacy = 'X-Center-Adjacent3Center-BBA')

            self._add_myperm2('CtrX3[F@2L.2D>U@2L.2F>R@2F.2D]', ("2L'"," D'","2F'",' D ','2L '," D'",'2F ',' D '), legacy = 'X-Center-Adjacent3Center-ADC')
            self._add_myperm2('CtrX3[F@2R.2U>U@2R.2B>R@2B.2U]', ("2R "," D'","2B ",' D ',"2R'"," D'","2B'",' D '), legacy = 'X-Center-Adjacent3Center-CBA')
            self._add_myperm2('CtrX3[F@2R.2D>U@2R.2F>R@2B.2D]', ("2R "," D ","2F'"," D'","2R'"," D ","2F "," D'"), legacy = 'X-Center-Adjacent3Center-BAD')
            self._add_myperm2('CtrX3[F@2L.2U>U@2L.2B>R@2F.2U]', ("2L'"," D ","2B "," D'","2L "," D ","2B'"," D'"), legacy = 'X-Center-Adjacent3Center-DCB')

            self._add_myperm2('CtrX3[F@2L.2D>U@2R.2B>R@2F.2D]', (" U2","2L'"," D'","2F'",' D ','2L '," D'",'2F ',' D '," U2"), legacy = 'X-Center-Adjacent3Center-ABC')
            self._add_myperm2('CtrX3[F@2R.2U>U@2L.2F>R@2B.2U]', (" U2","2R "," D'","2B ",' D ',"2R'"," D'","2B'",' D '," U2"), legacy = 'X-Center-Adjacent3Center-CDA')
            self._add_myperm2('CtrX3[F@2R.2D>U@2L.2B>R@2B.2D]', (" U2","2R "," D ","2F'"," D'","2R'"," D ","2F "," D'"," U2"), legacy = 'X-Center-Adjacent3Center-BCD')
            self._add_myperm2('CtrX3[F@2L.2U>U@2R.2F>R@2F.2U]', (" U2","2L'"," D ","2B "," D'","2L "," D ","2B'"," D'"," U2"), legacy = 'X-Center-Adjacent3Center-DAB')

            self._add_myperm2('CtrX3[B@2L.2U>F@2R.2D>U@2R.2F]', ("2R "," U ","2L "," U'","2R'"," U ","2L'"," U'"), legacy = 'X-Center-Line3Center-AAD')
            self._add_myperm2('CtrX3[B@2R.2D>F@2L.2U>U@2L.2B]', ("2L'"," U ","2R'"," U'","2L "," U ","2R "," U'"), legacy = 'X-Center-Line3Center-CCB')
            self._add_myperm2('CtrX3[B@2R.2U>F@2L.2D>U@2R.2F]', ("2R ", ' F ', '2U2', " F'", "2R'", ' F ', '2U2', " F'"), legacy = 'X-Center-Line3Center-ADA')
            self._add_myperm2('CtrX3[B@2L.2D>F@2R.2U>U@2L.2B]', ("2L'", ' F ', '2D2', " F'", '2L ', ' F ', '2D2', " F'"), legacy = 'X-Center-Line3Center-CBC')

            self._add_myperm2('CtrX3[B@2L.2U>F@2L.2D>U@2R.2F]', ("2L2"," F'","2R'"," F ","2L2"," F'","2R "," F "), legacy = 'X-Center-Line3Center-ADD')
            self._add_myperm2('CtrX3[B@2R.2D>F@2R.2U>U@2L.2B]', ("2R2"," F'","2L "," F ","2R2"," F'","2L'"," F "), legacy = 'X-Center-Line3Center-CBB')
            self._add_myperm2('CtrX3[B@2R.2U>F@2R.2D>U@2R.2F]', ('2U2', '2L ', '2F2', "2L'", ' F2', '2L ', '2F2', "2L'", ' F2', '2U2'), legacy = 'X-Center-Line3Center-AAA')
            self._add_myperm2('CtrX3[B@2L.2D>F@2L.2U>U@2L.2B]', ('2D2', "2R'", '2B2', '2R ', ' F2', "2R'", '2B2', '2R ', ' F2', '2D2'), legacy = 'X-Center-Line3Center-CCC')
            self._add_myperm2('CtrX3[B@2R.2U>F@2R.2D>U@2L.2B]', ('2D2', '2R ', '2B2', "2R'", ' B2', '2R ', '2B2', "2R'", ' B2', '2D2'), legacy = 'X-Center-Line3Center-CAA')
            self._add_myperm2('CtrX3[B@2L.2D>F@2L.2U>U@2R.2F]', ('2U2', "2L'", '2F2', '2L ', ' B2', "2L'", '2F2', '2L ', ' B2', '2U2'), legacy = 'X-Center-Line3Center-ACC')

            self._add_myperm2('CtrX3[B@2R.2D>F@2L.2D>U@2R.2F]', ('2D2', "2L'", ' U ', '2L ', '2D2', "2L'", " U'", '2L '), legacy = 'X-Center-Line3Center-ADB')
            self._add_myperm2('CtrX3[B@2L.2U>F@2R.2U>U@2L.2B]', ('2U2', '2R ', ' U ', "2R'", '2U2', '2R ', " U'", "2R'"), legacy = 'X-Center-Line3Center-CBD')
            self._add_myperm2('CtrX3[B@2L.2U>F@2R.2U>U@2R.2F]', ('2L ', ' U ', "2L'", '2U2', '2L ', " U'", "2L'", '2U2'), legacy = 'X-Center-Line3Center-ABD')
            self._add_myperm2('CtrX3[B@2R.2D>F@2L.2D>U@2L.2B]', ("2R'", ' U ', '2R ', '2D2', "2R'", " U'", '2R ', '2D2'), legacy = 'X-Center-Line3Center-CDB')
            
            self._add_myperm2('CtrX3[B@2L.2D>F@2R.2D>U@2R.2F]', ('2L ', ' U2', "2L'", '2D2', '2L ', ' U2', "2L'", '2D2'), legacy = 'X-Center-Line3Center-AAC')
            self._add_myperm2('CtrX3[B@2R.2U>F@2L.2U>U@2L.2B]', ("2R'", ' U2', '2R ', '2U2', "2R'", ' U2', '2R ', '2U2'), legacy = 'X-Center-Line3Center-CCA')
            self._add_myperm2('CtrX3[B@2L.2U>U@2R.2B>F@2R.2U]', ('2U2', '2L ', ' U2', "2L'", '2U2', '2L ', ' U2', "2L'"), legacy = 'X-Center-Line3Center-ACA')
            self._add_myperm2('CtrX3[B@2R.2D>U@2L.2F>F@2L.2D]', ('2D2', "2R'", ' U2', '2R ', '2D2', "2R'", ' U2', '2R '), legacy = 'X-Center-Line3Center-CAC')

            self._add_myperm2('CtrX3[B@2R.2D>F@2L.2U>U@2R.2F]', ('2F2', '2L ', ' D ', '2R ', " D'", "2L'", ' D ', "2R'", " D'", '2F2'), legacy = 'X-Center-Line3Center-ACB')
            self._add_myperm2('CtrX3[B@2L.2U>F@2R.2D>U@2L.2B]', ('2B2', "2R'", ' D ', "2L'", " D'", '2R ', ' D ', '2L ', " D'", '2B2'), legacy = 'X-Center-Line3Center-CAD')
            self._add_myperm2('CtrX3[B@2L.2D>F@2R.2U>U@2R.2F]', ('2F2', '2L ', ' F ', '2D2', " F'", "2L'", ' F ', '2D2', '2F2', " F'"), legacy = 'X-Center-Line3Center-ABC')
            self._add_myperm2('CtrX3[B@2R.2U>F@2L.2D>U@2L.2B]', ('2B2', "2R'", ' F ', '2U2', " F'", '2R ', ' F ', '2U2', '2B2', " F'"), legacy = 'X-Center-Line3Center-CDA')
            self._add_myperm2('CtrX3[B@2R.2D>F@2R.2U>U@2R.2F]', ('2F2', '2R2', " F'", "2L'", ' F ', '2R2', " F'", '2L ', '2F2', ' F '), legacy = 'X-Center-Line3Center-ABB')
            self._add_myperm2('CtrX3[B@2L.2U>F@2L.2D>U@2L.2B]', ('2B2', '2L2', " F'", '2R ', ' F ', '2L2', " F'", "2R'", '2B2', ' F '), legacy = 'X-Center-Line3Center-CDD')
            self._add_myperm2('CtrX3[B@2R.2D>F@2R.2D>U@2R.2F]', ('2F2', " F'", '2R2', " F'", "2L'", ' F ', '2R2', " F'", '2L ', '2F2', ' F2'), legacy = 'X-Center-Line3Center-AAB')
            self._add_myperm2('CtrX3[B@2L.2U>F@2L.2U>U@2L.2B]', ('2B2', " F'", '2L2', " F'", '2R ', ' F ', '2L2', " F'", "2R'", '2B2', ' F2'), legacy = 'X-Center-Line3Center-CCD')
            self._add_myperm2('CtrX3[B@2R.2U>F@2R.2U>U@2R.2F]', (' B2', '2F2', '2L ', ' B ', '2R2', " B'", "2L'", ' B ', '2R2', ' B ', '2F2'), legacy = 'X-Center-Line3Center-ABA')
            self._add_myperm2('CtrX3[B@2L.2D>F@2L.2D>U@2L.2B]', (' B2', '2B2', "2R'", ' B ', '2L2', " B'", '2R ', ' B ', '2L2', ' B ', '2B2'), legacy = 'X-Center-Line3Center-CDC')



            if self.size % 2 == 1:
                self._add_myperm2('CtrPlus4s[D@2L.S<>U@2L.S;D@2R.S<>U@2R.S]', (" S2","2R2"," S2","2R2"), legacy = 'Plus-Center-XA')
                self._add_myperm2('CtrPlus4s[D@2L.S<>U@M.2F;D@2R.S<>U@M.2B]', (" U "," S2","2R2"," S2","2R2"," U'"), legacy = 'Plus-Center-XB')
                self._add_myperm2('CtrPlus4s[B@2R.E<>F@2R.E;L@S.2D<>R@S.2D]', (" E ","2R2"," E'","2R2"), legacy = 'Plus-Center-Y')
                self._add_myperm2('CtrPlus4s[B@2L.E<>F@2L.E;D@M.2B<>U@M.2B]', ('2B ', " E'", '2B2', ' E ', '2B '), legacy = 'Plus-Center-Z')
                self._add_myperm2('CtrPlus8s~v02', ("2L2"," E2","2L'"," U ","2R'"," E2","2R "," U'","2L'"), legacy = 'Plus-Center-WA')
                self._add_myperm2('CtrPlus8s~v03', ("2R2"," E2","2R "," U ","2L "," E2","2L'"," U'","2R "), legacy = 'Plus-Center-WB')
                self._add_myperm2('CtrPlus4s[B@2R.E<>F@2R.E;D@2L.S<>U@2L.S]', ('2R ', ' S2', '2R2', ' S2', '2R '), legacy = 'Plus-Center-U')
                self._add_myperm2('CtrPlus4s[D@M.2B<>U@M.2B;L@S.2U<>R@S.2U]', (" E'", "2B'", ' E ', '2B2', " E'", "2B'", ' E '), legacy = 'Plus-Center-V')
                self._add_myperm2('CtrPlus4s[D@2L.S<>U@2L.S;D@M.2B<>U@M.2B]', (' M ', '2U2', " M'", '2U ', ' S ', '2U2', " S'", "2U'"), legacy = 'Plus-Center-TA')
                self._add_myperm2('CtrPlus4s[D@2L.S<>U@2L.S;D@M.2B<>U@M.2F]', ('2L2', " U'", ' S2', ' M2', ' U ', '2L2', " U'", ' M2', ' S2', ' U '), legacy = 'Plus-Center-TB')
                self._add_myperm2('CtrPlus4s[D@2L.S<>U@M.2F;D@M.2B<>U@2R.S]', (' S2', ' M2', ' D ', '2F2', " D'", ' S2', ' M2', ' D ', '2F2', " D'"), legacy = 'Plus-Center-TC')

                
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>R@S.2D>U@2R.S;D@2R.S>F@2R.E>L@S.2D]', ("2R "," E ","2R'"," E'"), legacy = 'Plus-Center-6A')
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>U@2R.S>R@S.2D;D@2R.S>L@S.2D>F@2R.E]', (" E ","2R "," E'","2R'"), legacy = 'Plus-Center-6B')
                self._add_myperm2('CtrPlus6p[3x2][B@2L.E>D@2R.S>F@2R.E;B@2R.E>F@2L.E>U@2R.S]', ("2R "," E2","2R'"," E2"), legacy = 'Plus-Center-6C')
                self._add_myperm2('CtrPlus6p[3x2][B@2L.E>F@2R.E>U@2R.S;B@2R.E>D@2R.S>F@2L.E]', (" E2","2R'"," E2","2R "), legacy = 'Plus-Center-6D')
                self._add_myperm2('CtrPlus10p[5x2]~v01', ('2U ', " M'", '2B ', ' M ', "2B'", "2U'", " E'", "2F'", " E'", '2F ', ' E2'), legacy = 'Plus-Center-6E')
                self._add_myperm2('CtrPlus10p[5x2]~v02', ('2U ', ' S ', '2R ', " S'", " E'", '2F ', ' E ', "2F'", "2R'", "2U'"), legacy = 'Plus-Center-6F')
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>L@S.2D>D@2R.S;F@2R.E>R@S.2D>U@2R.S]', ("2R "," E ","2R "," E'","2R2"), legacy = 'Plus-Center-6G')
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>D@2R.S>R@S.2U;F@2R.E>U@2R.S>L@S.2U]', ("2R2"," E'","2R'"," E ","2R'"), legacy = 'Plus-Center-6H')
                self._add_myperm2('CtrPlus6p[3x2][B@M.2D>R@2F.E>U@M.2B;D@M.2B>F@M.2D>L@2F.E]', (" M ","2U "," M ","2U'"," M2"), legacy = 'Plus-Center-6I')
                self._add_myperm2('CtrPlus6p[3x2][B@M.2D>U@M.2B>L@2F.E;D@M.2B>R@2F.E>F@M.2D]', (" M2","2U'"," M'","2U "," M'"), legacy = 'Plus-Center-6J')



                self._add_myperm2('CtrPlus6p[3x2][B@2L.E>B@2R.E>U@2R.S;D@2R.S>F@2L.E>F@2R.E]', ("2R2"," E2","2R "," E2","2R "), legacy = 'Plus-Center-4A')
                self._add_myperm2('CtrPlus6p[3x2][B@2L.E>D@2R.S>B@2R.E;F@2L.E>U@2R.S>F@2R.E]', ("2R "," E2","2R "," E2","2R2"), legacy = 'Plus-Center-4B')
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>B@M.2U>D@2R.S;F@2R.E>F@M.2U>U@2R.S]', ("2U "," S ","2R "," S'","2R'","2U'"), legacy = 'Plus-Center-4C')
                self._add_myperm2('CtrPlus8s~v01', (" M'", "2B'", ' M2', "2U'", " M'", '2U ', '2B '), legacy = 'Plus-Center-4D')
                self._add_myperm2('CtrPlus6p[3x2][B@M.2D>B@M.2U>D@M.2F;F@M.2D>F@M.2U>U@M.2F]', ('2U ', " M'", '2U ', " M'", "2U'", ' M2', "2U'"), legacy = 'Plus-Center-4E')
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>D@2L.S>D@2R.S;F@2R.E>U@2L.S>U@M.2B]', ("2L'"," U ","2R'"," E2","2R "," U'"," E2","2L "), legacy = 'Plus-Center-4XA')
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>D@2L.S>D@M.2B;F@2R.E>U@2L.S>U@2R.S]', ("2L'"," D'","2R'"," E2","2R "," D "," E2","2L "), legacy = 'Plus-Center-4XB')
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>U@M.2B>U@M.2F;D@2R.S>F@2R.E>D@M.2B]', ("2B "," U'","2R "," E'","2R'"," U "," E ","2B'"), legacy = 'Plus-Center-4YA')
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>U@M.2B>U@2R.S;D@M.2B>D@M.2F>F@2R.E]', ("2B "," D ","2R "," E'","2R'"," D'"," E ","2B'"), legacy = 'Plus-Center-4YB')
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>U@2L.S>U@M.2F;D@2L.S>D@2R.S>F@2R.E]', ('2L ', " U'", '2R ', ' E2', "2R'", " U ", ' E2', "2L'"), legacy = 'Plus-Center-4YC')
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>U@2L.S>U@2R.S;D@2L.S>D@M.2F>F@2R.E]', ('2L ', ' D ', '2R ', ' E2', "2R'", " D'", ' E2', "2L'"), legacy = 'Plus-Center-4YD')
                self._add_myperm2('CtrPlus6p[3x2][B@2L.E>D@2R.S>B@2R.E;F@2L.E>U@2L.S>F@2R.E]~v01', (' U2', '2R ', ' E2', '2R ', ' E2', '2R2', ' U2'), legacy = 'Plus-Center-4ZA')
                self._add_myperm2('CtrPlus6p[3x2][B@2L.E>D@2L.S>B@2R.E;F@2L.E>U@2R.S>F@2R.E]~v01', (' D2', '2R ', ' E2', '2R ', ' E2', '2R2', ' D2'), legacy = 'Plus-Center-4ZB')
                self._add_myperm2('CtrPlus6p[3x2][B@2L.E>D@2R.S>B@2R.E;F@2L.E>U@2L.S>F@2R.E]~v02', (' U2', '2R ', ' E2', "2R'", ' U2', '2L2', ' E2', '2L2'), legacy = 'Plus-Center-4ZC')
                self._add_myperm2('CtrPlus6p[3x2][B@2L.E>D@2L.S>B@2R.E;F@2L.E>U@2R.S>F@2R.E]~v02', (' D2', '2R ', ' E2', "2R'", ' D2', '2L2', ' E2', '2L2'), legacy = 'Plus-Center-4ZD')
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>U@2R.S>U@2L.S;D@2L.S>F@2R.E>D@M.2F]', ('2L ', ' E2', ' D ', '2R ', ' E2', "2R'", " D'", "2L'"), legacy = 'Plus-Center-4WA')
                self._add_myperm2('CtrPlus6p[3x2][B@2R.E>U@M.2F>U@2L.S;D@2L.S>F@2R.E>D@2R.S]', ('2L ', ' E2', " U'", '2R ', ' E2', "2R'", " U ", "2L'"), legacy = 'Plus-Center-4WB')


                self._add_myperm2('CtrMidBar6p[3x2][F@D>U@F>F@L;F@R>F@U>U@B]', (" M'"," U "," L "," R'"," M "," F'"," M "," F "," L'"," R "," M'"," U'"), legacy = 'MidCenterBar(VV)')
                self._add_myperm2('CtrMidBar6p[3x2][F@D>F@L>U@F;F@R>U@B>F@U]', (' U ', ' M ', " R'", ' L ', " F'", " M'", ' F ', " M'", ' R ', " L'", " U'", ' M '), legacy = 'MidCenterBar(HV)')
                self._add_myperm2('CtrMidBar6p[3x2][F@D>F@L>U@L;F@R>U@R>F@U]', (' M ', " R'", ' L ', " F'", " M'", ' F ', " M'", ' R ', " L'", " U'", ' M ',' U '), legacy = 'MidCenterBar(HH)')

                self._add_myperm2('CtrMidBar6p[3x2][D@B>U@F>D@L;D@F>U@B>D@R]', (" M2"," U "," L2"," R2"," M2"," D'"," M2"," D "," L2"," R2"," M2"," U'"), legacy = 'MidCenterBar-Opp(VV)')
                self._add_myperm2('CtrMidBar6p[3x2][D@B>D@L>U@F;D@F>D@R>U@B]', (' U ', ' M2', " R2", ' L2', " D'", " M2", ' D ', " M2", ' R2', " L2", " U'", ' M2'), legacy = 'MidCenterBar-Opp(HV)')

                self._add_myperm2('CtrMidBar6p[3x2][B@D>F@U>U@B;B@U>F@D>U@F]', (" F "," U "," L "," R'"," M "," F'"," M "," F "," L'"," R "," M'"," U'"," M'"," F'"), legacy = 'MidCenterBar-Adjacent3Center-A')
                self._add_myperm2('CtrMidBar6p[3x2][B@D>F@R>U@B;B@U>F@L>U@F]', (" U "," L "," R'"," M "," F'"," M "," F "," L'"," R "," M'"," U'"," M'"), legacy = 'MidCenterBar-Adjacent3Center-B')
                self._add_myperm2('CtrMidBar6p[3x2][B@D>F@U>U@R;B@U>F@D>U@L]', (' F ',' L '," R'",' M '," F'",' M ',' F '," L'",' R '," M'"," U'"," M'",' U '," F'"), legacy = 'MidCenterBar-Adjacent3Center-C')
                self._add_myperm2('CtrMidBar6p[3x2][B@D>F@R>U@R;B@U>F@L>U@L]', (" L "," R'"," M "," F'"," M "," F "," L'"," R "," M'"," U'"," M'"," U ")                , legacy = 'MidCenterBar-Adjacent3Center-D')
                self._add_myperm2('CtrMidBar6p[3x2][B@L>F@D>U@F;B@R>F@U>U@B]', (" B'"," F "," U "," L "," R'"," M "," F'"," M "," F "," L'"," R "," M'"," U'"," M'"," F'"," B "), legacy = 'MidCenterBar-Adjacent3Center-E')
                self._add_myperm2('CtrMidBar6p[3x2][B@L>F@L>U@F;B@R>F@R>U@B]', (" B'"," U "," L "," R'"," M "," F'"," M "," F "," L'"," R "," M'"," U'"," M'"," B "), legacy = 'MidCenterBar-Adjacent3Center-F')
                self._add_myperm2('CtrMidBar6p[3x2][B@L>F@U>U@L;B@R>F@D>U@R]', (' B '," F'"," R'",' L ',' M ',' F ',' M '," F'",' R '," L'"," M'",' U '," M'"," U'",' F '," B'"), legacy = 'MidCenterBar-Adjacent3Center-G')
                self._add_myperm2('CtrMidBar6p[3x2][B@L>F@L>U@L;B@R>F@R>U@R]', (" B'"," L "," R'"," M "," F'"," M "," F "," L'"," R "," M'"," U'"," M'"," U "," B "), legacy = 'MidCenterBar-Adjacent3Center-H')

                self._add_myperm2('CtrMidBar6p[3x2][F@D>U@L>R@D;F@U>U@R>R@U]', (" E'"," U'"," M'"," U "," L "," R'"," M "," F'"," M "," F "," L'"," R "," M'"," E "), legacy = 'MidCenterBar-Adjacent3Center-OA')
                self._add_myperm2('CtrMidBar6p[3x2][F@D>R@D>U@L;F@U>R@U>U@R]', self.invert_moves(self.myperms2['CtrMidBar6p[3x2][F@D>U@L>R@D;F@U>U@R>R@U]']), legacy = 'MidCenterBar-Adjacent3Center-OB')
                self._add_myperm2('CtrMidBar6p[3x2][F@D>U@F>R@D;F@U>U@B>R@U]', (" E'"," M'"," U "," L "," R'"," M "," F'"," M "," F "," L'"," R "," M'"," U'"," E "), legacy = 'MidCenterBar-Adjacent3Center-OC')
                self._add_myperm2('CtrMidBar6p[3x2][F@D>R@D>U@F;F@U>R@U>U@B]', self.invert_moves(self.myperms2['CtrMidBar6p[3x2][F@D>U@F>R@D;F@U>U@B>R@U]']), legacy = 'MidCenterBar-Adjacent3Center-OD')



                self._add_myperm2('CtrPlus3[D@2R.S>U@2L.S>U@M.2B]', (" U ", '2R2', " U'", ' S2', ' U ', '2R2', " U'", ' S2'), legacy = 'Plus-Center-Opp2X')
                self._add_myperm2('CtrPlus3[D@2R.S>U@M.2B>D@M.2F]', (' U ', '2R2', " U'", ' M2', ' U ', '2R2', " U'", ' M2'), legacy = 'Plus-Center-Opp2X01')
                self._add_myperm2('CtrPlus4s[D@2L.S<>D@M.2F;D@2R.S<>U@M.2B]', ('2F2', ' D ', ' M2', '2F2', ' M2', '2F2', " D'", '2F2'), legacy = 'Plus-Center-Opp2X02')
                self._add_myperm2('CtrPlus3[D@2R.S>U@M.2B>U@2R.S]', ('2R2', " U'", ' S2', ' U ', '2R2', " U'", ' S2', ' U '), legacy = 'Plus-Center-Opp2Y')
                self._add_myperm2('CtrPlus3[D@2R.S>U@2R.S>D@M.2F]', ('2R2', " U'", ' M2', ' U ', '2R2', " U'", ' M2', ' U '), legacy = 'Plus-Center-Opp2Y01')
                self._add_myperm2('CtrPlus3[D@M.2F>U@2R.S>U@M.2B]', (" M2"," U ","2F2"," U'"," M2"," U ","2F2"," U'"), legacy = 'Plus-Center-Opp2Z')
                self._add_myperm2('CtrPlus3[D@2R.S>D@M.2F>U@M.2B]', (' M2', ' U ', '2R2', " U'", ' M2', ' U ', '2R2', " U'"), legacy = 'Plus-Center-Opp2Z01')
                

                self._add_myperm2('CtrPlus3[F@M.2U>U@2R.S>U@2L.S]', (" S'", '2U ', ' S ', " U2", " S'", "2U'", ' S ', " U2"), legacy = 'Plus-Center-Middle-Inside')
                self._add_myperm2('CtrPlus3[B@M.2U>D@2R.S>D@2L.S]', (" S'", '2U ', ' S ', " D2", " S'", "2U'", ' S ', " D2"), legacy = 'Plus-Center-Middle-Outside')
                self._add_myperm2('CtrPlus3[F@2L.E>U@2R.S>U@2L.S]~v01', ('2R ', ' E2', "2R'", " U2", '2R ', ' E2', "2R'", " U2"), legacy = 'Plus-Center-Middle-Vertical')
                self._add_myperm2('CtrPlus3[B@2L.E>D@2R.S>D@2L.S]', ('2R ', ' E2', "2R'", " D2", '2R ', ' E2', "2R'", " D2"), legacy = 'Plus-Center-Middle-Vertical01')
                self._add_myperm2('CtrPlus3[F@2R.E>U@2R.S>F@M.2D]', ("2R "," U "," M'"," U'","2R'"," U "," M "," U'"), legacy = 'Plus-Center-Middle-Vertical02')
                self._add_myperm2('CtrPlus3[F@2L.E>U@2L.S>F@M.2U]', ("2L'"," U "," M'"," U'","2L "," U "," M "," U'"), legacy = 'Plus-Center-Middle-Vertical03')
                self._add_myperm2('CtrPlus3[F@2L.E>F@2R.E>U@2R.S]', (" E'"," L2"," E ","2R'"," E'"," L2"," E ","2R "), legacy = 'Plus-Center-Middle-Vertical04')
                self._add_myperm2('CtrPlus3[F@2L.E>U@2R.S>U@2L.S]~v02', (" S'", ' L2', ' S ', "2L'", " S'", ' L2', ' S ', '2L '), legacy = 'Plus-Center-Middle-Vertical05')

                
                self._add_myperm2('CtrPlus3[F@2L.E>U@2L.S>U@2R.S]~v01', (' U2', '2R ', ' E2', "2R'", " U2", '2R ', ' E2', "2R'"), legacy = 'Plus-Center-Middle-Diagonal')
                self._add_myperm2('CtrPlus3[B@2L.E>D@2L.S>D@2R.S]', (" D2", '2R ', ' E2', "2R'", " D2", '2R ', ' E2', "2R'"), legacy = 'Plus-Center-Middle-Diagonal01')
                self._add_myperm2('CtrPlus3[F@2L.E>U@2R.S>F@2R.E]', ("2R'"," E'"," L2"," E ","2R "," E'"," L2"," E "), legacy = 'Plus-Center-Middle-Diagonal02')
                self._add_myperm2('CtrPlus3[F@2L.E>U@2L.S>U@2R.S]~v02', ("2L'", " S'", ' L2', ' S ', '2L ', " S'", ' L2', ' S '), legacy = 'Plus-Center-Middle-Diagonal03')

                self._add_myperm2('CtrPlus3[F@M.2U>U@2L.S>U@2R.S]', (" U2", " S'", '2U ', ' S ', " U2", " S'", "2U'", ' S '), legacy = 'Plus-Center-Middle-Inside01')
                self._add_myperm2('CtrPlus3[B@M.2U>D@2L.S>D@2R.S]', (" D2", " S'", '2U ', ' S ', " D2", " S'", "2U'", ' S '), legacy = 'Plus-Center-Middle-Outside01')

                self._add_myperm2('CtrPlus3[F@2L.E>U@M.2B>U@M.2F]', (' U2', '2F ', ' E ', "2F'", ' U2', '2F ', " E'", "2F'"), legacy = 'Plus-Center-Middle-Inside02')
                self._add_myperm2('CtrPlus3[B@2L.E>D@M.2B>D@M.2F]', (' D2', '2F ', ' E ', "2F'", ' D2', '2F ', " E'", "2F'"), legacy = 'Plus-Center-Middle-Outside02')

                self._add_myperm2('CtrPlus3[F@2L.E>F@M.2U>U@2L.S]', (" U ", " M'", " U'", "2L'", " U ", ' M ', " U'", '2L '), legacy = 'Plus-Center-Middle-Inside03')
                self._add_myperm2('CtrPlus3[F@2R.E>F@M.2D>U@2R.S]', (" U ", " M'", " U'", "2R ", " U ", ' M ', " U'", "2R'"), legacy = 'Plus-Center-Middle-Outside03')

                self._add_myperm2('CtrPlus3[F@M.2U>U@M.2B>U@2L.S]', (" F'", '2L ', ' F ', ' M ', " F'", "2L'", ' F ', " M'"), legacy = 'Plus-Center-Middle-Inside04')
                self._add_myperm2('CtrPlus3[F@M.2D>U@M.2F>U@2R.S]', (" F'", "2R'", ' F ', ' M ', " F'", "2R ", ' F ', " M'"), legacy = 'Plus-Center-Middle-Outside04')




                self._add_myperm2('CtrPlus3[B@M.2D>U@M.2F>U@M.2B]', (" U2","2D'"," M'","2D "," M "," U2"," M'","2D'"," M ","2D "), legacy = 'Plus-Center-InOut')
                self._add_myperm2('CtrPlus3[B@M.2U>U@M.2B>U@M.2F]', (" U2","2U "," M'","2U'"," M "," U2"," M'","2U "," M ","2U'"), legacy = 'Plus-Center-InOut01')
                self._add_myperm2('CtrPlus3[F@2R.E>F@M.2U>U@M.2B]', (" M'"," U ","2R "," U'"," M "," U ","2R'"," U'"), legacy = 'Plus-Center-InOut02')
                self._add_myperm2('CtrPlus3[F@2L.E>F@M.2D>U@M.2F]', (" M'"," U ","2L'"," U'"," M "," U ","2L "," U'")                , legacy = 'Plus-Center-InOut03')
                self._add_myperm2('CtrPlus3[B@M.2D>U@M.2B>U@M.2F]', ("2D'"," M'","2D "," M "," U2"," M'","2D'"," M ","2D "," U2"), legacy = 'Plus-Center-OutOut')
                self._add_myperm2('CtrPlus3[B@M.2U>U@M.2F>U@M.2B]', ("2U "," M'","2U'"," M "," U2"," M'","2U "," M ","2U'"," U2"), legacy = 'Plus-Center-InIn')

                self._add_myperm2('CtrPlus3[F@2R.E>U@M.2F>R@2F.E]', (" F ","2D "," R2","2D'"," M ","2D "," R2","2D'"," M'"," F'"), legacy = 'Plus-Center-Adjacent3Center-AAA')
                self._add_myperm2('CtrPlus3[F@2L.E>U@M.2B>R@2B.E]', (" F ","2U'"," R2","2U "," M ","2U'"," R2","2U "," M'"," F'"), legacy = 'Plus-Center-Adjacent3Center-CCC')
                self._add_myperm2('CtrPlus3[F@M.2U>U@2R.S>R@S.2D]', (" F'"," E'","2L'"," U2","2L "," E ","2L'"," U2","2L "," F "), legacy = 'Plus-Center-Adjacent3Center-BBB')
                self._add_myperm2('CtrPlus3[F@M.2D>U@2L.S>R@S.2U]', (" F'"," E'","2R "," U2","2R'"," E ","2R "," U2","2R'"," F "), legacy = 'Plus-Center-Adjacent3Center-DDD')

                self._add_myperm2('CtrPlus3[F@2L.E>U@M.2F>R@2F.E]', (" F'","2D "," R2","2D'"," M ","2D "," R2","2D'"," M'"," F "), legacy = 'Plus-Center-Adjacent3Center-AAC')
                self._add_myperm2('CtrPlus3[F@2R.E>U@M.2B>R@2B.E]', (" F'","2U'"," R2","2U "," M ","2U'"," R2","2U "," M'"," F "), legacy = 'Plus-Center-Adjacent3Center-CCA')
                self._add_myperm2('CtrPlus3[F@M.2D>U@2R.S>R@S.2D]', (" F "," E'","2L'"," U2","2L "," E ","2L'"," U2","2L "," F'"), legacy = 'Plus-Center-Adjacent3Center-BBD')
                self._add_myperm2('CtrPlus3[F@M.2U>U@2L.S>R@S.2U]', (" F "," E'","2R "," U2","2R'"," E ","2R "," U2","2R'"," F'"), legacy = 'Plus-Center-Adjacent3Center-DDB')
            
                self._add_myperm2('CtrPlus3[F@M.2D>U@M.2F>R@2F.E]', ("2D "," R2","2D'"," M ","2D "," R2","2D'"," M'"), legacy = 'Plus-Center-Adjacent3Center-AAD')
                self._add_myperm2('CtrPlus3[F@M.2U>U@M.2B>R@2B.E]', ("2U'"," R2","2U "," M ","2U'"," R2","2U "," M'"), legacy = 'Plus-Center-Adjacent3Center-CCB')
                self._add_myperm2('CtrPlus3[F@2L.E>U@2R.S>R@S.2D]', (" E'","2L'"," U2","2L "," E ","2L'"," U2","2L "), legacy = 'Plus-Center-Adjacent3Center-BBC')
                self._add_myperm2('CtrPlus3[F@2R.E>U@2L.S>R@S.2U]', (" E'","2R "," U2","2R'"," E ","2R "," U2","2R'"), legacy = 'Plus-Center-Adjacent3Center-DDA')

                self._add_myperm2('CtrPlus3[F@2L.E>U@2L.S>R@S.2U]', ('2L2', " F'", ' E ', ' F ', '2L ', " F'", " E'", ' F ', '2L '), legacy = 'Plus-Center-Adjacent3Center-DDC')
                self._add_myperm2('CtrPlus3[F@2R.E>U@2R.S>R@S.2D]', ('2R2', " F'", ' E ', ' F ', "2R'", " F'", " E'", ' F ', "2R'"), legacy = 'Plus-Center-Adjacent3Center-BBA')
                self._add_myperm2('CtrPlus3[F@M.2D>U@M.2B>R@2B.E]', ('2D ', ' F ', " M'", " F'", '2D ', ' F ', ' M ', " F'", '2D2'), legacy = 'Plus-Center-Adjacent3Center-CCD')
                self._add_myperm2('CtrPlus3[F@M.2U>U@M.2F>R@2F.E]', ("2U'", ' F ', " M'", " F'", "2U'", ' F ', ' M ', " F'", '2U2'), legacy = 'Plus-Center-Adjacent3Center-AAB')
                

                self._add_myperm2('CtrPlus3[F@2L.E>U@2L.S>R@2F.E]', ("2L'"," D'"," S'",' D ','2L '," D'",' S ',' D '), legacy = 'Plus-Center-Adjacent3Center-ADC')
                self._add_myperm2('CtrPlus3[F@2R.E>U@2R.S>R@2B.E]', ("2R "," D'"," S'",' D ',"2R'"," D'"," S ",' D '), legacy = 'Plus-Center-Adjacent3Center-CBA')
                self._add_myperm2('CtrPlus3[F@M.2D>U@M.2F>R@S.2D]', (" M'", ' D ', "2F'", " D'", ' M ', ' D ', "2F ", " D'"), legacy = 'Plus-Center-Adjacent3Center-BAD')
                self._add_myperm2('CtrPlus3[F@M.2U>U@M.2B>R@S.2U]', (" M'", ' D ', '2B ', " D'", ' M ', ' D ', "2B'", " D'"), legacy = 'Plus-Center-Adjacent3Center-DCB')

                self._add_myperm2('CtrPlus3[F@2L.E>U@2R.S>R@2F.E]', (" U2","2L'"," D'"," S'",' D ','2L '," D'",' S ',' D '," U2"), legacy = 'Plus-Center-Adjacent3Center-ABC')
                self._add_myperm2('CtrPlus3[F@2R.E>U@2L.S>R@2B.E]', (" U2","2R "," D'"," S'",' D ',"2R'"," D'"," S ",' D '," U2"), legacy = 'Plus-Center-Adjacent3Center-CDA')
                self._add_myperm2('CtrPlus3[F@M.2D>U@M.2B>R@S.2D]', (" U2"," M'", "2F'", ' M ', ' U ', " M'", " U'", '2F ', ' U ', ' M ', " U "), legacy = 'Plus-Center-Adjacent3Center-BCD')
                self._add_myperm2('CtrPlus3[F@M.2U>U@M.2F>R@S.2U]', (" U2"," M'", "2B ", ' M ', ' U ', " M'", " U'", "2B'", ' U ', ' M ', " U "), legacy = 'Plus-Center-Adjacent3Center-DAB')

                self._add_myperm2('CtrPlus3[B@M.2U>F@2R.E>U@2R.S]', ("2R "," U "," M "," U'","2R'"," U "," M'"," U'"), legacy = 'Plus-Center-Line3Center-BBA')
                self._add_myperm2('CtrPlus3[B@M.2D>F@2R.E>U@2R.S]', ('2R '," U'",' M ',' U ',"2R'"," U'"," M'",' U '), legacy = 'Plus-Center-Line3Center-BBC')
                self._add_myperm2('CtrPlus3[B@M.2D>F@2L.E>U@2R.S]', (" U2","2L'"," U "," M "," U'","2L "," U "," M'"," U "), legacy = 'Plus-Center-Line3Center-BDC')
                self._add_myperm2('CtrPlus3[B@M.2U>F@2L.E>U@2R.S]', (" U2","2L'"," U'",' M ',' U ',"2L "," U'"," M'"," U'"), legacy = 'Plus-Center-Line3Center-BDA')
                self._add_myperm2('CtrPlus3[B@2R.E>F@M.2D>U@2R.S]', ("2R ", ' F ', ' E2', " F'", "2R'", ' F ', ' E2', " F'"), legacy = 'Plus-Center-Line3Center-BAB')
                self._add_myperm2('CtrPlus3[B@2R.E>F@M.2U>U@2R.S]', ("2R ", " F'", ' E2', " F ", "2R'", " F'", ' E2', " F "), legacy = 'Plus-Center-Line3Center-BCB')

                self._add_myperm2('CtrPlus3[B@2R.E>F@2R.E>U@M.2F]', ("2R2"," F "," M "," F'","2R2"," F "," M'"," F'"), legacy = 'Plus-Center-Line3Center-ABB')
                self._add_myperm2('CtrPlus3[B@2R.E>F@2R.E>U@M.2B]', ("2R2"," F'"," M "," F ","2R2"," F'"," M'"," F "), legacy = 'Plus-Center-Line3Center-CBB')
                self._add_myperm2('CtrPlus3[B@2R.E>F@M.2D>U@M.2B]', (" F'","2R2"," F'"," M "," F ","2R2"," F'"," M'"," F2"), legacy = 'Plus-Center-Line3Center-CAB')
                self._add_myperm2('CtrPlus3[B@2R.E>F@M.2U>U@M.2F]', (" F ","2R2"," F "," M "," F'","2R2"," F "," M'"," F2"), legacy = 'Plus-Center-Line3Center-ACB')

                self._add_myperm2('CtrPlus3[B@2R.E>F@M.2D>U@M.2F]', (" M'"," U'","2R'"," U "," M "," U'","2R "," U "), legacy = 'Plus-Center-Line3Center-AAB')
                self._add_myperm2('CtrPlus3[B@2R.E>F@M.2U>U@M.2B]', (" M'"," U ","2R'"," U'"," M "," U ","2R "," U'"), legacy = 'Plus-Center-Line3Center-CCB')
                self._add_myperm2('CtrPlus3[B@M.2U>F@2R.E>U@M.2F]', (" M'", " F'", '2U2', ' F ', ' M ', " F'", '2U2', ' F '), legacy = 'Plus-Center-Line3Center-ABA')
                self._add_myperm2('CtrPlus3[B@M.2D>F@2R.E>U@M.2B]', (" M'", ' F ', '2D2', " F'", ' M ', ' F ', '2D2', " F'"), legacy = 'Plus-Center-Line3Center-CBC')

                self._add_myperm2('CtrPlus3[B@M.2U>F@M.2D>U@2R.S]', (" M2"," F'","2R'"," F "," M2"," F'","2R "," F "), legacy = 'Plus-Center-Line3Center-BAA')
                self._add_myperm2('CtrPlus3[B@M.2D>F@M.2U>U@2R.S]', (" M2"," F ","2R'"," F'"," M2"," F ","2R "," F'"), legacy = 'Plus-Center-Line3Center-BCC')

                self._add_myperm2('CtrPlus3[B@2L.E>F@2L.E>U@2R.S]', ("2L ", " S ", ' R2', " S'", '2L2', " S ", ' R2', " S'", "2L "), legacy = 'Plus-Center-Line3Center-BDD')
                self._add_myperm2('CtrPlus3[B@2R.E>F@2R.E>U@2R.S]', ('2L ', ' E2', "2L'", ' U2', '2L ', ' E2', '2L2', ' E2', '2L ', ' U2', "2L'", ' E2', '2L '), legacy = 'Plus-Center-Line3Center-BBB')
                self._add_myperm2('CtrPlus3[B@2R.E>F@2L.E>U@2R.S]', (' E2', "2L'", ' U2', '2L ', ' E2', "2L'", ' U2', '2L '), legacy = 'Plus-Center-Line3Center-BDB')
                self._add_myperm2('CtrPlus3[B@2L.E>F@2R.E>U@2R.S]', ('2L ', ' U2', "2L'", ' E2', '2L ', ' U2', "2L'", ' E2'), legacy = 'Plus-Center-Line3Center-BBD')
                self._add_myperm2('CtrPlus3[B@M.2U>F@M.2U>U@2R.S]', (" S ", "2U ", " S'", ' U2', " S ", '2U2', " S'", ' U2', " S ", "2U ", " S'"), legacy = 'Plus-Center-Line3Center-BCA')
                self._add_myperm2('CtrPlus3[B@M.2D>F@M.2D>U@2R.S]', (" S'", '2D ', ' S ', ' U2', " S'", '2D2', ' S ', ' U2', " S'", '2D ', ' S '), legacy = 'Plus-Center-Line3Center-BAC')

                self._add_myperm2('CtrPlus3[B@M.2U>F@M.2U>U@M.2B]', ("2U2"," B "," M "," B'","2U2"," B "," M'"," B'"), legacy = 'Plus-Center-Line3Center-CCA')
                self._add_myperm2('CtrPlus3[B@M.2D>F@M.2D>U@M.2F]', ("2D2"," B "," M "," B'","2D2"," B "," M'"," B'"), legacy = 'Plus-Center-Line3Center-AAC')
                self._add_myperm2('CtrPlus3[B@M.2U>F@M.2U>U@M.2F]', (' F ', " M'", " F'", '2U2', ' F ', ' M ', " F'", '2U2'), legacy = 'Plus-Center-Line3Center-ACA')
                self._add_myperm2('CtrPlus3[B@M.2D>F@M.2D>U@M.2B]', (' F ', " M'", " F'", '2D2', ' F ', ' M ', " F'", '2D2'), legacy = 'Plus-Center-Line3Center-CAC')
                self._add_myperm2('CtrPlus3[B@M.2D>F@M.2U>U@M.2B]', (" B2","2U2"," B "," M "," B'","2U2"," B "," M'"," B "), legacy = 'Plus-Center-Line3Center-CCC')
                self._add_myperm2('CtrPlus3[B@M.2U>F@M.2D>U@M.2F]', (" B2","2D2"," B "," M "," B'","2D2"," B "," M'"," B "), legacy = 'Plus-Center-Line3Center-AAA')

                self._add_myperm2('CtrPlus3[B@2L.E>F@2R.E>U@M.2F]', (' E2', '2R ', " U'", "2R'", ' E2', '2R ', ' U ', "2R'"), legacy = 'Plus-Center-Line3Center-ABD')
                self._add_myperm2('CtrPlus3[B@2L.E>F@2R.E>U@M.2B]', (' E2', '2R ', ' U ', "2R'", ' E2', '2R ', " U'", "2R'"), legacy = 'Plus-Center-Line3Center-CBD')
                
            if self.size >= 6:
                self._add_myperm2('CtrObl6p[3x2][D@2L.3F>D@2R.3B>U@2R.3F;D@2R.3F>U@2L.3F>U@2R.3B]', ("2R2","3F2","2R2","3F2"), legacy = 'Oblique-Center-Opp4-XA')
                self._add_myperm2('CtrObl6p[3x2][D@2L.3F>D@2R.3B>U@3R.2B;D@2R.3F>U@3R.2F>U@3L.2B]', (" U ","2R2","3F2","2R2","3F2"," U'"), legacy = 'Oblique-Center-Opp4-XB')
                self._add_myperm2('CtrObl6p[3x2][D@2L.3F>D@2R.3B>U@2L.3B;D@2R.3F>U@2R.3B>U@2L.3F]', (" U2","2R2","3F2","2R2","3F2"," U2"), legacy = 'Oblique-Center-Opp4-XC')
                self._add_myperm2('CtrObl8s~v03', ("2L2","3D2","2L'"," U ","2R'","3D2","2R "," U'","2L'"), legacy = 'Oblique-Center-Opp4-WA')
                self._add_myperm2('CtrObl8s~v04', ("2R2","3U2","2R "," U ","2L ","3U2","2L'"," U'","2R "), legacy = 'Oblique-Center-Opp4-WB')
                self._add_myperm2('CtrObl8s~v05', ("2L2","3U2","2L'"," U ","2R'","3U2","2R "," U'","2L'"), legacy = 'Oblique-Center-Opp4-WC')
                self._add_myperm2('CtrObl8s~v06', ("2R2","3D2","2R "," U ","2L ","3D2","2L'"," U'","2R "), legacy = 'Oblique-Center-Opp4-WD')
                self._add_myperm2('CtrObl6p[3x2][D@2L.3B>U@3L.2B>U@3R.2F;D@2R.3B>U@2L.3B>U@3L.2F]', ('2L2', ' U ', '2L2', '3B2', '2L2', " U'", '2L2', ' U ', '3B2', " U'"), legacy = 'Oblique-Center-Opp4-VA')
                self._add_myperm2('CtrObl6p[3x2][D@2L.3B>U@2R.3B>U@2L.3F;D@2R.3B>U@3R.2B>U@2L.3B]', (" U'", '2L2', ' U ', '2L2', '3B2', '2L2', " U'", '2L2', ' U ', '3B2'), legacy = 'Oblique-Center-Opp4-VB')
                self._add_myperm2('CtrObl8p[3+5]~v02', ('2B2', '3L2', '2B2', '3L2', ' U ', '3B2', '2R2', '3B2', '2R2', " U'"), legacy = 'Oblique-Center-Opp4-VC')
                self._add_myperm2('CtrObl6p[3x2][D@2L.3B>U@3L.2B>D@3R.2B;D@3L.2B>U@3R.2B>U@2L.3B]', ('2L ', "2B'", '3U ', '2B2', "3U'", "2B'", "2L'"), legacy = 'Oblique-Center-Opp4-UA')
                self._add_myperm2('CtrObl6p[3x2][D@2L.3F>U@3R.2B>D@3L.2B;D@3R.2B>U@3L.2B>U@2L.3F]', ('2L ', "2B'", "3D'", '2B2', "3D ", "2B'", "2L'"), legacy = 'Oblique-Center-Opp4-UB')
                self._add_myperm2('CtrObl6p[3x2][D@2L.3B>U@2L.3F>D@3R.2B;D@3L.2B>U@2L.3B>U@3L.2F]', (" U ",'2L ', "2B'", '3U ', '2B2', "3U'", "2B'", "2L'"," U'"), legacy = 'Oblique-Center-Opp4-UC')
                self._add_myperm2('CtrObl6p[3x2][D@2L.3F>U@2L.3B>D@3L.2B;D@3R.2B>U@2L.3F>U@3R.2F]', (" U ",'2L ', "2B'", "3D'", '2B2', "3D ", "2B'", "2L'"," U'"), legacy = 'Oblique-Center-Opp4-UD')
                self._add_myperm2('CtrObl6p[3x2][D@2L.3B>U@3R.2F>D@3R.2B;D@3L.2B>U@3L.2F>U@2R.3F]', (" U2",'2L ', "2B'", '3U ', '2B2', "3U'", "2B'", "2L'"," U2"), legacy = 'Oblique-Center-Opp4-UE')
                self._add_myperm2('CtrObl6p[3x2][D@2L.3F>U@3L.2F>D@3L.2B;D@3R.2B>U@3R.2F>U@2R.3B]', (" U2",'2L ', "2B'", "3D'", '2B2', "3D ", "2B'", "2L'"," U2"), legacy = 'Oblique-Center-Opp4-UF')
                self._add_myperm2('CtrObl8s~v07', ('3L2', '2B2', '3L2', '2B2', ' U2', '3R2', '2B2', '3R2', '2B2', ' U2'), legacy = 'Oblique-Center-Opp4-ZA')
                self._add_myperm2('CtrObl8s~v08', (" U ", '3L2', '2B2', '3L2', '2B2', ' U2', '3R2', '2B2', '3R2', '2B2', ' U '), legacy = 'Oblique-Center-Opp4-ZB')
                self._add_myperm2('CtrObl4s[D@2L.3F<>U@2L.3B;D@3R.2F<>U@3R.2B]', (" U'", '2L2', '2F2', ' U ', '3R2', " U'", '2F2', '2L2', ' U ', '3R2'), legacy = 'Oblique-Center-Opp4-YA')
                self._add_myperm2('CtrObl4s[D@2L.3F<>U@3L.2F;D@3R.2F<>U@2L.3B]', ('2L2', '2F2', ' U ', '3R2', " U'", '2F2', '2L2', ' U ', '3R2', " U'"), legacy = 'Oblique-Center-Opp4-YB')
                self._add_myperm2('CtrObl4s[D@2L.3F<>U@2R.3F;D@3R.2F<>U@3L.2F]', (" U ", '2L2', '2F2', ' U ', '3R2', " U'", '2F2', '2L2', ' U ', '3R2', " U2"), legacy = 'Oblique-Center-Opp4-YC')
                self._add_myperm2('CtrObl8p[3+5]~v01', (" D'", '2L2', ' D ', '3F2', '2L2', '3F2', '2L2', '3B2', " D'", '2L2', ' D ', '3B2'), legacy = 'Oblique-Center-Opp4-T')


                self._add_myperm2('CtrObl10p[5x2]~v03', ("3U2","2R2","3U'","2R2","3U'","2R2","3U'","2R2","3U ")      , legacy = 'Oblique-Center-8')
                
                self._add_myperm2('CtrObl6p[3x2][B@2R.3U>L@3F.2U>U@2R.3F;D@2R.3F>F@2R.3U>R@3F.2U]', ("2R ","3U ","2R'","3U'"), legacy = 'Oblique-Center-6A')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3U>R@3F.2D>U@2R.3F;D@2R.3F>F@2R.3U>L@3F.2D]', ("2R ","3U'","2R'","3U "), legacy = 'Oblique-Center-6B')
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>D@2R.3F>F@2R.3U;B@2R.3U>F@2L.3U>U@2R.3F]', ("2R ","3U2","2R'","3U2"), legacy = 'Oblique-Center-6C')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3D>F@2R.3U>R@3F.2U;B@2R.3U>L@3F.2U>F@2R.3D]', ("2R2","3U ","2R2","3U'"), legacy = 'Oblique-Center-6D')
                self._add_myperm2('CtrObl10p[5x2]~v01', ("2R'", "3F'", "2U'", '3F ', '2U ', '2R ', '3L ', '2F2', '3L ', '2F2', '3L2'), legacy = 'Oblique-Center-6E')
                self._add_myperm2('CtrObl10p[5x2]~v02', ('3L ', "2F'", '3D ', '2F ', "3D'", '2L ', "3L'", '3B2', '2L ', '3B2', '2L2'), legacy = 'Oblique-Center-6F')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3D>R@3F.2U>D@2R.3F;F@2R.3D>L@3F.2U>U@2R.3F]', ("2R ","3U ","2R ","3U'","2R2"), legacy = 'Oblique-Center-6G')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3D>L@3F.2D>D@2R.3F;F@2R.3D>R@3F.2D>U@2R.3F]', ("2R ","3U'","2R ","3U ","2R2"), legacy = 'Oblique-Center-6H')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3D>D@2R.3F>R@3F.2U;F@2R.3D>U@2R.3F>L@3F.2U]', ("2R2","3U ","2R'","3U'","2R'"), legacy = 'Oblique-Center-6I')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3D>D@2R.3F>L@3F.2D;F@2R.3D>U@2R.3F>R@3F.2D]', ("2R2","3U'","2R'","3U ","2R'"), legacy = 'Oblique-Center-6J')
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>F@2R.3U>R@3F.2U;B@2R.3U>L@3F.2U>F@2L.3U]', ("2R2","3U ","2R2","3U ","2R2","3U2","2R2"), legacy = 'Oblique-Center-6K')
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>L@3F.2D>F@2R.3U;B@2R.3U>F@2L.3U>R@3F.2D]', ("2R2","3U2","2R2","3U ","2R2","3U ","2R2"), legacy = 'Oblique-Center-6L')
                self._add_myperm2('CtrObl6p[3x2][B@3R.2D>R@2F.3U>L@2F.3D;F@3R.2D>L@2F.3U>R@2F.3D]', ("2U ","3R2","2U2","3R2","2U "), legacy = 'Oblique-Center-6M')
                self._add_myperm2('CtrObl6p[3x2][B@3R.2D>R@3F.2D>L@3B.2D;F@3R.2D>L@3F.2D>R@3B.2D]', ("2F ","3R ","2F2","3R'","2F "), legacy = 'Oblique-Center-6N')
                self._add_myperm2('CtrObl6p[3x2][B@3R.2D>L@3B.2D>R@3F.2D;F@3R.2D>R@3B.2D>L@3F.2D]', ("2F'","3R ","2F2","3R'","2F'")                , legacy = 'Oblique-Center-6O')

                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>B@2R.3D>U@2R.3B;D@2R.3B>F@2L.3U>F@2R.3D]', ("2R2","3U2","2R ","3U2","2R "), legacy = 'Oblique-Center-4A')
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>D@2R.3F>B@2R.3D;F@2L.3U>U@2R.3F>F@2R.3D]', ("2R ","3U2","2R ","3U2","2R2"), legacy = 'Oblique-Center-4B')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3D>B@3R.2U>D@2R.3F;F@2R.3D>F@3R.2U>U@2R.3F]', ("2U ","3F ","2R ","3F'","2R'","2U'"), legacy = 'Oblique-Center-4C')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3D>D@2R.3F>D@3L.2F;F@2R.3D>U@2R.3F>U@3L.2F]', ("2F'", '3D ', "2R'", "3D'", '2R ', '2F '), legacy = 'Oblique-Center-4D')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3D>D@2L.3B>D@2R.3F;F@2R.3D>U@2L.3B>U@3R.2B]', ("2L'"," U ","2R'","3D2","2R "," U'","3D2","2L "), legacy = 'Oblique-Center-4XA')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3D>D@2L.3B>D@3R.2B;F@2R.3D>U@2L.3B>U@2R.3F]', ("2L'"," D'","2R'","3D2","2R "," D ","3D2","2L "), legacy = 'Oblique-Center-4XB')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3U>U@3R.2B>U@3L.2F;D@2R.3F>F@2R.3U>D@3R.2B]', ("2B "," U'","2R ","3U ","2R'"," U ","3U'","2B'"), legacy = 'Oblique-Center-4YA')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3U>U@3R.2B>U@2R.3F;D@3L.2F>F@2R.3U>D@3R.2B]', ("2B "," D ","2R ","3U ","2R'"," D'","3U'","2B'"), legacy = 'Oblique-Center-4YB')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3U>U@2L.3B>U@3L.2F;D@2L.3B>D@2R.3F>F@2R.3U]', ('2L ', " U'", '2R ', '3U2', "2R'", " U ", '3U2', "2L'"), legacy = 'Oblique-Center-4YC')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3U>U@2L.3B>U@2R.3F;D@2L.3B>D@3L.2F>F@2R.3U]', ('2L ', ' D ', '2R ', '3U2', "2R'", " D'", '3U2', "2L'"), legacy = 'Oblique-Center-4YD')
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>D@2R.3F>B@2R.3D;F@2L.3U>U@2L.3B>F@2R.3D]', (' U2', '2R ', '3U2', '2R ', '3U2', '2R2', ' U2'), legacy = 'Oblique-Center-4ZA')
                self._add_myperm2('CtrObl6p[3x2][B@2L.3U>D@2L.3B>B@2R.3D;F@2L.3U>U@2R.3F>F@2R.3D]', (' D2', '2R ', '3U2', '2R ', '3U2', '2R2', ' D2'), legacy = 'Oblique-Center-4ZB')
                self._add_myperm2('CtrObl8s~v01', (' U2', '2R ', '3U2', "2R'", ' U2', '2L2', '3U2', '2L2'), legacy = 'Oblique-Center-4ZC')
                self._add_myperm2('CtrObl8s~v02', (' D2', '2R ', '3U2', "2R'", ' D2', '2L2', '3U2', '2L2'), legacy = 'Oblique-Center-4ZD')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3U>U@2R.3F>U@2L.3B;D@2L.3B>F@2R.3U>D@3L.2F]', ('2L ', '3U2', ' D ', '2R ', '3U2', "2R'", " D'", "2L'"), legacy = 'Oblique-Center-4WA')
                self._add_myperm2('CtrObl6p[3x2][B@2R.3U>U@3L.2F>U@2L.3B;D@2L.3B>F@2R.3U>D@2R.3F]', ('2L ', '3U2', " U'", '2R ', '3U2', "2R'", " U ", "2L'"), legacy = 'Oblique-Center-4WB')
                

                self._add_myperm2('CtrObl3[D@2R.3F>U@2L.3F>U@3L.2B]', (" U ", '2R2', " U'", '3F2', ' U ', '2R2', " U'", '3F2'), legacy = 'Oblique-Center-Opp2X')
                self._add_myperm2('CtrObl6p[3x2][D@2R.3F>U@2R.3B>U@3L.2B;U@2L.3B>U@3R.2B>U@3L.2F]', ('3B2', ' U ', '3B2', '2R2', '3B2', '2R2', " U'", '3B2'), legacy = 'Oblique-Center-Opp2X01')
                self._add_myperm2('CtrObl3[D@2R.3F>U@2R.3B>U@3L.2B]', (" D'", '3L2', ' D ', '2R2', " D'", '3L2', ' D ', '2R2'), legacy = 'Oblique-Center-Opp2X02')
                self._add_myperm2('CtrObl3[D@2R.3F>U@3L.2B>U@2R.3B]', ('2R2', " U'", '3F2', ' U ', '2R2', " U'", '3F2', ' U '), legacy = 'Oblique-Center-Opp2Y')
                self._add_myperm2('CtrObl3[D@2R.3F>U@3R.2F>U@2R.3B]', ('2R2', ' U ', '3F2', " U'", '2R2', ' U ', '3F2', " U'"), legacy = 'Oblique-Center-Opp2Y01')
                self._add_myperm2('CtrObl6p[3x2][D@2L.3F>D@3L.2B>D@3R.2F;D@2R.3F>U@2R.3B>D@3L.2F]', ('3F2', ' D ', '2R2', '3F2', '2R2', '3F2', " D'", '3F2'), legacy = 'Oblique-Center-Opp2Y02')
                self._add_myperm2('CtrObl6p[3x2][D@2R.3F>U@3R.2F>U@2R.3B;U@2L.3B>U@3R.2B>U@3L.2F]', ('3B2', " U'", '2R2', '3B2', '2R2', '3B2', ' U ', '3B2'), legacy = 'Oblique-Center-Opp2Y03')
                self._add_myperm2('CtrObl3[D@2R.3F>U@2R.3B>D@3L.2F]', ('2R2', " U'", '3L2', ' U ', '2R2', " U'", '3L2', ' U '), legacy = 'Oblique-Center-Opp2Y04')
                self._add_myperm2('CtrObl3[D@2R.3F>U@2R.3B>D@3R.2B]', ('2R2', ' U ', '3R2', " U'", '2R2', ' U ', '3R2', " U'"), legacy = 'Oblique-Center-Opp2Y05')
                self._add_myperm2('CtrObl3[D@2R.3F>U@2R.3B>U@3R.2F]', (' U ', '3F2', " U'", '2R2', ' U ', '3F2', " U'", '2R2'), legacy = 'Oblique-Center-Opp2Z')
                self._add_myperm2('CtrObl6p[3x2][D@2R.3F>U@2R.3B>U@3R.2F;U@2L.3B>U@3L.2F>U@3R.2B]', ('3B2', " U'", '3B2', '2R2', '3B2', '2R2', ' U ', '3B2'), legacy = 'Oblique-Center-Opp2Z01')
                self._add_myperm2('CtrObl3[D@2R.3F>U@3R.2F>D@3L.2F]', (' U ', '3F2', " U'", '2F2', ' U ', '3F2', " U'", '2F2'), legacy = 'Oblique-Center-Opp2Z02')


                self._add_myperm2('CtrObl3[F@3R.2U>U@2R.3F>U@2L.3B]~v01', ("3F'", '2U ', '3F ', " U2", "3F'", "2U'", '3F ', " U2"), legacy = 'Oblique-Center-InOut-Diagonal')
                self._add_myperm2('CtrObl3[B@3R.2U>D@2R.3F>D@2L.3B]~v01', ("3F'", '2U ', '3F ', " D2", "3F'", "2U'", '3F ', " D2"), legacy = 'Oblique-Center-InOut-Diagonal01')
                self._add_myperm2('CtrObl3[F@2L.3U>U@2R.3F>U@2L.3B]', ('2R ', '3U2', "2R'", " U2", '2R ', '3U2', "2R'", " U2"), legacy = 'Oblique-Center-InOut-Vertical')
                self._add_myperm2('CtrObl3[B@2L.3U>D@2R.3F>D@2L.3B]', ('2R ', '3U2', "2R'", " D2", '2R ', '3U2', "2R'", " D2"), legacy = 'Oblique-Center-InOut-Vertical01')
                self._add_myperm2('CtrObl3[F@2R.3D>U@2R.3F>F@3L.2D]', ("2R "," U ","3L'"," U'","2R'"," U ","3L "," U'"), legacy = 'Oblique-Center-InOut-Vertical02')
                self._add_myperm2('CtrObl3[F@2L.3U>U@2L.3B>F@3R.2U]', ("2L'"," U ","3R "," U'","2L "," U ","3R'"," U'"), legacy = 'Oblique-Center-InOut-Vertical03')
                self._add_myperm2('CtrObl3[F@2R.3U>U@2R.3B>F@3R.2D]', ("2R "," U ","3R "," U'","2R'"," U ","3R'"," U'"), legacy = 'Oblique-Center-InOut-Vertical04')
                self._add_myperm2('CtrObl3[F@2L.3D>U@2L.3F>F@3L.2U]', ("2L'"," U ","3L'"," U'","2L "," U ","3L "," U'")                , legacy = 'Oblique-Center-InOut-Vertical05')
                self._add_myperm2('CtrObl3[F@2R.3D>F@3L.2D>U@2R.3F]', (' U ', "3L'", " U'", '2R ', ' U ', '3L ', " U'", "2R'"), legacy = 'Oblique-Center-InOut-Diagonal02')
                self._add_myperm2('CtrObl3[F@2L.3U>F@3R.2U>U@2L.3B]', (' U ', "3R ", " U'", "2L'", ' U ', "3R'", " U'", "2L "), legacy = 'Oblique-Center-InOut-Diagonal03')
                
                self._add_myperm2('CtrObl3[F@2L.3U>U@2L.3B>U@2R.3F]', (' U2', '2R ', '3U2', "2R'", " U2", '2R ', '3U2', "2R'"), legacy = 'Oblique-Center-InIn-Diagonal')
                self._add_myperm2('CtrObl3[B@2L.3U>D@2L.3B>D@2R.3F]', (" D2", '2R ', '3U2', "2R'", " D2", '2R ', '3U2', "2R'"), legacy = 'Oblique-Center-OutOut-Diagonal')
                self._add_myperm2('CtrObl3[F@3R.2U>U@2L.3B>U@2R.3F]', (" U2", "3F'", '2U ', '3F ', " U2", "3F'", "2U'", '3F '), legacy = 'Oblique-Center-InIn-Vertical')
                self._add_myperm2('CtrObl3[B@3R.2U>D@2L.3B>D@2R.3F]', (" D2", "3F'", '2U ', '3F ', " D2", "3F'", "2U'", '3F '), legacy = 'Oblique-Center-OutOut-Vertical')

                self._add_myperm2('CtrObl3[F@3R.2U>U@2R.3F>U@2L.3B]~v02', (' U2', "3B'", "2U'", '3B ', ' U2', "3B'", '2U ', '3B '), legacy = 'Oblique-Center-InOut-Diagonal04')
                self._add_myperm2('CtrObl3[B@3R.2U>D@2R.3F>D@2L.3B]~v02', (' D2', "3B'", "2U'", '3B ', ' D2', "3B'", '2U ', '3B '), legacy = 'Oblique-Center-InOut-Diagonal05')

                

                self._add_myperm2('CtrObl3[F@2R.3U>U@3R.2F>R@2F.3D]', (" F ","2D "," R2","2D'","3R'","2D "," R2","2D'","3R "," F'"), legacy = 'Oblique-Center-Adjacent3Center-AAA')
                self._add_myperm2('CtrObl3[F@2L.3D>U@3L.2B>R@2B.3U]', (" F ","2U'"," R2","2U ","3L ","2U'"," R2","2U ","3L'"," F'"), legacy = 'Oblique-Center-Adjacent3Center-CCC')
                self._add_myperm2('CtrObl3[F@3L.2U>U@2R.3B>R@3B.2D]', (" F'","3D'","2L'"," U2","2L ","3D ","2L'"," U2","2L "," F "), legacy = 'Oblique-Center-Adjacent3Center-BBB')
                self._add_myperm2('CtrObl3[F@3R.2D>U@2L.3F>R@3F.2U]', (" F'","3U ","2R "," U2","2R'","3U'","2R "," U2","2R'"," F "), legacy = 'Oblique-Center-Adjacent3Center-DDD')

                self._add_myperm2('CtrObl3[F@2L.3D>U@3R.2F>R@2F.3D]', (" F'","2D "," R2","2D'","3R'","2D "," R2","2D'","3R "," F "), legacy = 'Oblique-Center-Adjacent3Center-AAC')
                self._add_myperm2('CtrObl3[F@2R.3U>U@3L.2B>R@2B.3U]', (" F'","2U'"," R2","2U ","3L ","2U'"," R2","2U ","3L'"," F "), legacy = 'Oblique-Center-Adjacent3Center-CCA')
                self._add_myperm2('CtrObl3[F@3R.2D>U@2R.3B>R@3B.2D]', (" F ","3D'","2L'"," U2","2L ","3D ","2L'"," U2","2L "," F'"), legacy = 'Oblique-Center-Adjacent3Center-BBD')
                self._add_myperm2('CtrObl3[F@3L.2U>U@2L.3F>R@3F.2U]', (" F ","3U ","2R "," U2","2R'","3U'","2R "," U2","2R'"," F'"), legacy = 'Oblique-Center-Adjacent3Center-DDB')
                
                self._add_myperm2('CtrObl3[F@3R.2D>U@3R.2F>R@2F.3D]', ("2D "," R2","2D'","3R'","2D "," R2","2D'","3R "), legacy = 'Oblique-Center-Adjacent3Center-AAD')
                self._add_myperm2('CtrObl3[F@3L.2U>U@3L.2B>R@2B.3U]', ("2U'"," R2","2U ","3L ","2U'"," R2","2U ","3L'"), legacy = 'Oblique-Center-Adjacent3Center-CCB')
                self._add_myperm2('CtrObl3[F@2L.3D>U@2R.3B>R@3B.2D]', ("3D'","2L'"," U2","2L ","3D ","2L'"," U2","2L "), legacy = 'Oblique-Center-Adjacent3Center-BBC')
                self._add_myperm2('CtrObl3[F@2R.3U>U@2L.3F>R@3F.2U]', ("3U ","2R "," U2","2R'","3U'","2R "," U2","2R'"), legacy = 'Oblique-Center-Adjacent3Center-DDA')

                self._add_myperm2('CtrObl3[F@2L.3U>U@2R.3F>R@3F.2D]', ('3U ', "2L'", ' U2', '2L ', "3U'", "2L'", ' U2', '2L '), legacy = 'Oblique-Center-Adjacent3Center-AAB')
                self._add_myperm2('CtrObl3[F@2R.3D>U@2L.3B>R@3B.2U]', ("3D'", '2R ', ' U2', "2R'", '3D ', '2R ', ' U2', "2R'"), legacy = 'Oblique-Center-Adjacent3Center-CCD')
                self._add_myperm2('CtrObl3[F@3L.2D>U@3L.2F>R@2F.3U]', ('2D ', ' R2', "2D'", '3L ', '2D ', ' R2', "2D'", "3L'"), legacy = 'Oblique-Center-Adjacent3Center-DDC')
                self._add_myperm2('CtrObl3[F@3R.2U>U@3R.2B>R@2B.3D]', ("2U'", ' R2', '2U ', "3R'", "2U'", ' R2', '2U ', '3R '), legacy = 'Oblique-Center-Adjacent3Center-BBA')

                self._add_myperm2('CtrObl3[F@2L.3D>U@2L.3F>R@2F.3D]', ("2L'"," D'","3F'",' D ','2L '," D'",'3F ',' D '), legacy = 'Oblique-Center-Adjacent3Center-ADC')
                self._add_myperm2('CtrObl3[F@2R.3U>U@2R.3B>R@2B.3U]', ("2R "," D'","3B ",' D ',"2R'"," D'","3B'",' D '), legacy = 'Oblique-Center-Adjacent3Center-CBA')
                self._add_myperm2('CtrObl3[F@2R.3D>U@2R.3F>R@2B.3D]', ("2R "," D ","3F'"," D'","2R'"," D ","3F "," D'"), legacy = 'Oblique-Center-Adjacent3Center-BAD')
                self._add_myperm2('CtrObl3[F@2L.3U>U@2L.3B>R@2F.3U]', ("2L'"," D ","3B "," D'","2L "," D ","3B'"," D'"), legacy = 'Oblique-Center-Adjacent3Center-DCB')

                self._add_myperm2('CtrObl3[F@2L.3D>U@2R.3B>R@2F.3D]', (" U2","2L'"," D'","3F'",' D ','2L '," D'",'3F ',' D '," U2"), legacy = 'Oblique-Center-Adjacent3Center-ABC')
                self._add_myperm2('CtrObl3[F@2R.3U>U@2L.3F>R@2B.3U]', (" U2","2R "," D'","3B ",' D ',"2R'"," D'","3B'",' D '," U2"), legacy = 'Oblique-Center-Adjacent3Center-CDA')
                self._add_myperm2('CtrObl3[F@2R.3D>U@2L.3B>R@2B.3D]', (" U2","2R "," D ","3F'"," D'","2R'"," D ","3F "," D'"," U2"), legacy = 'Oblique-Center-Adjacent3Center-BCD')
                self._add_myperm2('CtrObl3[F@2L.3U>U@2R.3F>R@2F.3U]', (" U2","2L'"," D ","3B "," D'","2L "," D ","3B'"," D'"," U2"), legacy = 'Oblique-Center-Adjacent3Center-DAB')

                self._add_myperm2('CtrObl3[B@3L.2U>F@2R.3D>U@2R.3F]', ("2R "," U ","3L "," U'","2R'"," U ","3L'"," U'"), legacy = 'Oblique-Center-Line3Center-AAD')
                self._add_myperm2('CtrObl3[B@3R.2D>F@2L.3U>U@2L.3B]', ("2L'"," U ","3R'"," U'","2L "," U ","3R "," U'"), legacy = 'Oblique-Center-Line3Center-CCB')
                self._add_myperm2('CtrObl3[B@2R.3U>F@3L.2D>U@2R.3F]', ("2R ", ' F ', '3U2', " F'", "2R'", ' F ', '3U2', " F'"), legacy = 'Oblique-Center-Line3Center-ADA')
                self._add_myperm2('CtrObl3[B@2L.3D>F@3R.2U>U@2L.3B]', ("2L'", ' F ', '3D2', " F'", '2L ', ' F ', '3D2', " F'"), legacy = 'Oblique-Center-Line3Center-CBC')

                self._add_myperm2('CtrObl3[B@2L.3U>F@2L.3D>U@3R.2F]', ("2L2"," F'","3R'"," F ","2L2"," F'","3R "," F "), legacy = 'Oblique-Center-Line3Center-ADD')
                self._add_myperm2('CtrObl3[B@2R.3D>F@2R.3U>U@3L.2B]', ("2R2"," F'","3L "," F ","2R2"," F'","3L'"," F "), legacy = 'Oblique-Center-Line3Center-CBB')
                self._add_myperm2('CtrObl3[B@2R.3U>F@2R.3D>U@2R.3F]', ('3U2', '2L ', '3F2', "2L'", ' F2', '2L ', '3F2', "2L'", ' F2', '3U2'), legacy = 'Oblique-Center-Line3Center-AAA')
                self._add_myperm2('CtrObl3[B@2L.3D>F@2L.3U>U@2L.3B]', ('3D2', "2R'", '3B2', '2R ', ' F2', "2R'", '3B2', '2R ', ' F2', '3D2'), legacy = 'Oblique-Center-Line3Center-CCC')
                self._add_myperm2('CtrObl3[B@2R.3U>F@2R.3D>U@2L.3B]', ('3D2', '2R ', '3B2', "2R'", ' B2', '2R ', '3B2', "2R'", ' B2', '3D2'), legacy = 'Oblique-Center-Line3Center-CAA')
                self._add_myperm2('CtrObl3[B@2L.3D>F@2L.3U>U@2R.3F]', ('3U2', "2L'", '3F2', '2L ', ' B2', "2L'", '3F2', '2L ', ' B2', '3U2'), legacy = 'Oblique-Center-Line3Center-ACC')

                self._add_myperm2('CtrObl3[B@2R.3D>F@2L.3D>U@3R.2F]', ('3D2', "2L'", ' U ', '2L ', '3D2', "2L'", " U'", '2L '), legacy = 'Oblique-Center-Line3Center-ADB')
                self._add_myperm2('CtrObl3[B@2L.3U>F@2R.3U>U@3L.2B]', ('3U2', '2R ', ' U ', "2R'", '3U2', '2R ', " U'", "2R'"), legacy = 'Oblique-Center-Line3Center-CBD')
                self._add_myperm2('CtrObl3[B@2L.3U>F@2R.3U>U@3R.2F]', ('2L ', ' U ', "2L'", '3U2', '2L ', " U'", "2L'", '3U2'), legacy = 'Oblique-Center-Line3Center-ABD')
                self._add_myperm2('CtrObl3[B@2R.3D>F@2L.3D>U@3L.2B]', ("2R'", ' U ', '2R ', '3D2', "2R'", " U'", '2R ', '3D2'), legacy = 'Oblique-Center-Line3Center-CDB')

                self._add_myperm2('CtrObl3[B@2L.3D>F@2R.3D>U@2R.3F]', ('2L ', ' U2', "2L'", '3D2', '2L ', ' U2', "2L'", '3D2'), legacy = 'Oblique-Center-Line3Center-AAC')
                self._add_myperm2('CtrObl3[B@2R.3U>F@2L.3U>U@2L.3B]', ("2R'", ' U2', '2R ', '3U2', "2R'", ' U2', '2R ', '3U2'), legacy = 'Oblique-Center-Line3Center-CCA')
                self._add_myperm2('CtrObl3[B@2L.3U>U@2R.3B>F@2R.3U]', ('3U2', '2L ', ' U2', "2L'", '3U2', '2L ', ' U2', "2L'"), legacy = 'Oblique-Center-Line3Center-ACA')
                self._add_myperm2('CtrObl3[B@2R.3D>U@2L.3F>F@2L.3D]', ('3D2', "2R'", ' U2', '2R ', '3D2', "2R'", ' U2', '2R '), legacy = 'Oblique-Center-Line3Center-CAC')

                self._add_myperm2('CtrObl3[B@3R.2D>F@2L.3U>U@2R.3F]', ('3F2', '2L ', ' D ', '3R ', " D'", "2L'", ' D ', "3R'", " D'", '3F2'), legacy = 'Oblique-Center-Line3Center-ACB')
                self._add_myperm2('CtrObl3[B@3L.2U>F@2R.3D>U@2L.3B]', ('3B2', "2R'", ' D ', "3L'", " D'", '2R ', ' D ', '3L ', " D'", '3B2'), legacy = 'Oblique-Center-Line3Center-CAD')
                self._add_myperm2('CtrObl3[B@2L.3D>F@3R.2U>U@2R.3F]', ('3F2', '2L ', ' F ', '3D2', " F'", "2L'", ' F ', '3D2', '3F2', " F'"), legacy = 'Oblique-Center-Line3Center-ABC')
                self._add_myperm2('CtrObl3[B@2R.3U>F@3L.2D>U@2L.3B]', ('3B2', "2R'", ' F ', '3U2', " F'", '2R ', ' F ', '3U2', '3B2', " F'"), legacy = 'Oblique-Center-Line3Center-CDA')
                self._add_myperm2('CtrObl3[B@2R.3D>F@2R.3U>U@3R.2F]', ('2F2', '2R2', " F'", "3L'", ' F ', '2R2', " F'", '3L ', '2F2', ' F '), legacy = 'Oblique-Center-Line3Center-ABB')
                self._add_myperm2('CtrObl3[B@2L.3U>F@2L.3D>U@3L.2B]', ('2B2', '2L2', " F'", '3R ', ' F ', '2L2', " F'", "3R'", '2B2', ' F '), legacy = 'Oblique-Center-Line3Center-CDD')

                self._add_myperm2('CtrObl3[B@2R.3D>F@3R.2D>U@3R.2F]', ('2F2', " F'", '2R2', " F'", "3L'", ' F ', '2R2', " F'", '3L ', '2F2', ' F2'), legacy = 'Oblique-Center-Line3Center-AAB')
                self._add_myperm2('CtrObl3[B@2L.3U>F@3L.2U>U@3L.2B]', ('2B2', " F'", '2L2', " F'", '3R ', ' F ', '2L2', " F'", "3R'", '2B2', ' F2'), legacy = 'Oblique-Center-Line3Center-CCD')
                self._add_myperm2('CtrObl3[B@3R.2U>F@2R.3U>U@3R.2F]', (' B2', '2F2', '3L ', ' B ', '2R2', " B'", "3L'", ' B ', '2R2', ' B ', '2F2'), legacy = 'Oblique-Center-Line3Center-ABA')
                self._add_myperm2('CtrObl3[B@3L.2D>F@2L.3D>U@3L.2B]', (' B2', '2B2', "3R'", ' B ', '2L2', " B'", '3R ', ' B ', '2L2', ' B ', '2B2'), legacy = 'Oblique-Center-Line3Center-CDC')



            self._add_myperm2('CtrBar3[F@2L>U@2R>U@2L]', (' F2', '2R ', ' F2', ' D2', ' B2', '2L ', ' B2', ' D2'), legacy = 'OuterCenterBar-A')
            self._add_myperm2('CtrBar3[F@2L>U@2L>U@2R]', (' D2', ' B2', "2L'", ' B2', ' D2', ' F2', "2R'", ' F2'), legacy = 'OuterCenterBar-B')
            
            self._add_myperm2('CtrBar3[F@2D>U@2R>U@2L]', (" F'", '2R ', ' F2', ' D2', ' B2', '2L ', ' B2', ' D2', " F'"), legacy = 'OuterCenterBar-C')
            self._add_myperm2('CtrBar3[F@2U>U@2R>U@2L]', (' F ', '2R ', ' F2', ' D2', ' B2', '2L ', ' B2', ' D2', ' F '), legacy = 'OuterCenterBar-D')

            self._add_myperm2('CtrBar3[F@2D>U@2B>U@2F]', (' U ', " F'", '2R ', ' F2', ' D2', ' B2', '2L ', ' B2', ' D2', " F'", " U'"), legacy = 'OuterCenterBar-E')
            self._add_myperm2('CtrBar3[F@2U>U@2B>U@2F]', (' U ', ' F ', '2R ', ' F2', ' D2', ' B2', '2L ', ' B2', ' D2', ' F ', " U'"), legacy = 'OuterCenterBar-F')
            self._add_myperm2('CtrBar3[F@2D>U@2F>U@2B]', (" U'", " F'", '2R ', ' F2', ' D2', ' B2', '2L ', ' B2', ' D2', " F'", ' U '), legacy = 'OuterCenterBar-G')
            self._add_myperm2('CtrBar3[F@2U>U@2F>U@2B]', (" U'", ' F ', '2R ', ' F2', ' D2', ' B2', '2L ', ' B2', ' D2', ' F ', ' U '), legacy = 'OuterCenterBar-H')
    
            self._add_myperm2('CtrBar4s[F@2L<>U@2R;F@2R<>U@2L]', ('2R ', ' F2', ' D2', ' B2', '2L ', '2R ', ' B2', ' D2', ' F2', '2L '), legacy = 'OuterCenterBar-W')
            self._add_myperm2('CtrBar4s[F@2D<>U@2R;F@2U<>U@2L]', (' F ', '2R ', ' F2', ' D2', ' B2', '2L ', '2R ', ' B2', ' D2', ' F2', '2L ', " F'"), legacy = 'OuterCenterBar-WW')

            self._add_myperm2('CtrBar3[B@2L>U@2R>F@2L]', ("2L'", ' B2', ' D2', ' F2', '2R2', ' F2', ' D2', ' B2', "2L'"), legacy = 'OuterCenterBar-KA')
            self._add_myperm2('CtrBar3[B@2L>F@2L>U@2R]', ("2L ", ' B2', ' D2', ' F2', '2R2', ' F2', ' D2', ' B2', "2L "), legacy = 'OuterCenterBar-KB')

            self._add_myperm2('CtrBar3[B@2L>U@2R>F@2R]', ("2R "," U2"," F2"," D2","2L'"," D2"," F2"," U2","2R2"), legacy = 'OuterCenterBar-JA')
            self._add_myperm2('CtrBar3[B@2L>F@2R>U@2R]', ("2R2"," U2"," F2"," D2","2L "," D2"," F2"," U2","2R'"), legacy = 'OuterCenterBar-JB')

            self._add_myperm2('CtrBar3[B@2R>U@2R>F@2R]', (" B2","2R "," U2"," F2"," D2","2L'"," D2"," F2"," U2","2R2"," B2"), legacy = 'OuterCenterBar-IA')
            self._add_myperm2('CtrBar3[B@2R>F@2R>U@2R]', (" B2","2R2"," U2"," F2"," D2","2L "," D2"," F2"," U2","2R'"," B2"), legacy = 'OuterCenterBar-IB')
            
            self._add_myperm2('CtrBar3[D@2L>U@2L>U@2R]', ("2L2"," F2"," U2"," F2","2L2"," F2"," U2"," F2"), legacy = 'OuterCenterBar-X')
            self._add_myperm2('CtrBar3[D@2L>U@2R>U@2L]', (" F2"," U2"," F2","2L2"," F2"," U2"," F2","2L2"), legacy = 'OuterCenterBar-Y')
            self._add_myperm2('CtrBar3[D@2L>U@2F>U@2B]', (" U ","2L2"," F2"," U2"," F2","2L2"," F2"," U2"," F2"," U'"), legacy = 'OuterCenterBar-Z')
            self._add_myperm2('CtrBar4s[D@2L<>U@2L;D@2R<>U@2R]', ("2R2"," F2"," B2","2L2"," F2"," B2"), legacy = 'OuterCenterBar-XX')
            self._add_myperm2('CtrBar4s[D@2L<>U@2F;D@2R<>U@2B]', (" U ","2R2"," F2"," B2","2L2"," F2"," B2"," U'"), legacy = 'OuterCenterBar-ZZ')

            

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

            self._add_myperm2('CtrX8p[4x2]+W2-2s[FL@D<>UB@R]', ("2R2", ' B ', "2D'", ' B ', '2R ', ' B2', ' U2', ' B ', '2D ', " B'", ' U2', '2R2'), legacy = 'WingSwapSkew-H')
            self._add_myperm2('CtrX8p[4x2]+W2-2s[FL@U<>UB@L]', ('2L2', ' B ', '2U ', ' B ', "2L'", ' B2', ' U2', ' B ', "2U'", " B'", ' U2', '2L2'), legacy = 'WingSwapSkew-G')
            self._add_myperm2('CtrX8p[4x2]+W2-2s[FL@D<>UF@R]', ("2R'", ' B ', "2D'", ' B ', '2R ', ' B2', ' U2', ' B ', '2D ', " B'", ' U2', '2R '), legacy = 'WingSwapSkew-D')
            self._add_myperm2('CtrX8p[4x2]+W2-2s[FL@U<>UF@L]', ('2L ', ' B ', '2U ', ' B ', "2L'", ' B2', ' U2', ' B ', "2U'", " B'", ' U2', "2L'")           , legacy = 'WingSwapSkew-C')
            self._add_myperm2('CtrX8p[4x2]+W2-2s[DF@R<>FL@D]', (' B ', "2D'", ' B ', '2R ', ' B2', ' U2', ' B ', '2D ', " B'", ' U2'), legacy = 'WingSwapSkew-Ex')
            self._add_myperm2('CtrX8p[4x2]+W2-2s[DF@L<>FL@U]', (' B ', '2U ', ' B ', "2L'", ' B2', ' U2', ' B ', "2U'", " B'", ' U2')     , legacy = 'WingSwapSkew-Ey')
            self._add_myperm2('CtrX8p[4x2]+W2-2s[DB@R<>FL@D]', ('2R ', ' B ', "2D'", ' B ', '2R ', ' B2', ' U2', ' B ', '2D ', " B'", ' U2', "2R'"), legacy = 'WingSwapSkew-Fx')
            self._add_myperm2('CtrX8p[4x2]+W2-2s[DB@L<>FL@U]', ("2L'", ' B ', '2U ', ' B ', "2L'", ' B2', ' U2', ' B ', "2U'", " B'", ' U2', '2L '), legacy = 'WingSwapSkew-Fy')

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
