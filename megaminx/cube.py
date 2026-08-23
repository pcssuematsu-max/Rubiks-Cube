"""Megaminx cube domain extracted from the legacy Megaminx GUI."""

import random
from functools import reduce
from pathlib import Path

import numpy as np

from core.scramble_selector import ScrambleSelector
from core.myperm_effects import rename_myperms_by_effect
from core.myperm_keys import make_myperm_key, normalize_myperm_registry, single_move_myperm_name
from core.myperm_points import load_myperm_points, reindex_myperms_by_points


PERFECT_VAL = 1.0e+8


DISPLAY_FACE_NAMES = {
    "U": "U",
    "F": "F",
    "L": "L",
    "R": "R",
    "B": "B",
    "D": "D",
    "P": "bL",
    "Q": "bR",
    "G": "dL",
    "H": "dR",
    "X": "sL",
    "Y": "sR",
}

DISPLAY_MOVE_SUFFIXES = {
    "+ ": "",
    "- ": "'",
    "++": "2",
    "--": "2'",
}

DISPLAY_TO_INTERNAL_FACE = {value: key for key, value in DISPLAY_FACE_NAMES.items()}
DISPLAY_TO_INTERNAL_SUFFIX = {value: key for key, value in DISPLAY_MOVE_SUFFIXES.items()}
DISPLAY_FACE_TOKENS = sorted(DISPLAY_TO_INTERNAL_FACE, key = len, reverse = True)


class MegaminxCube:
    def __init__(self,S = '', size = None, F2L = False, OLL = False, Centers = False, Edges = False, Cross = False):        
        self.myperm_point_puzzle = "megaminx"
        self.requested_size = size
        self.size = 0
        self.F2L = F2L
        self.OLL = OLL
        self.Centers = Centers
        self.Edges = Edges
        self.Cross = Cross
        self.color_to_num = {'U':0,'F':1,'L':2,'P':3,'Q':4,'R':5,'B':6,'X':7,'G':8,'H':9,'Y':10,'D':11}
        self.colors = ['U','F','L','P','Q','R','B','X','G','H','Y','D']
        
        self.move = {}

        self.opposite = {}


        self.inverse = {"+ ":"- ","- ":"+ ","++":"--","--":"++"}

        self.mult = {("+ ","+ "):"++",("+ ","++"):"--",("+ ","--"):"- ",("+ ","- "):0,
                     ("++","+ "):"--",("++","++"):"- ",("++","--"):0,("++","- "):"+ ",
                     ("--","+ "):"- ",("--","++"):0,("--","--"):"+ ",("--","- "):"++",
                     ("- ","+ "):0,("- ","++"):"+ ",("- ","--"):"++",("- ","- "):"--"}
                     

        self.flip = {}
        self.flip['UD'] = {"U":"U","F":"F","L":"R","P":"Q","Q":"P","R":"L","B":"B","Y":"X","H":"G","G":"H","X":"Y","D":"D"}


        self.rotate = {}
        self.rotate['UD'] = {"U":"U","F":"L","L":"P","P":"Q","Q":"R","R":"F","B":"Y","Y":"H","H":"G","G":"X","X":"B","D":"D"}
        
        self.rotate['FB'] = {"U":"R","R":"H","H":"G","G":"L","L":"U","D":"X","X":"P","P":"Q","Q":"Y","Y":"D","F":"F","B":"B"}
        
        self.rotate['DU'] = {}
        self.rotate['BF'] = {}
        for key in self.colors:
            self.rotate['DU'][self.rotate['UD'][key]] = key
            self.rotate['BF'][self.rotate['FB'][key]] = key
        
        TF_X = [(),('R1',),('R1','R1'),('r1',),('r1','r1'),('F',),('F','R1'),('F','R1','R1'),('F','r1'),('F','r1','r1')]
        TF_Y = [(),('R2',),('r2',),('R2','r1'),('r2','R1'),('R2','R1'),('R2','R2'),('r2','r2'),('R2','R2','r1'),('r2','r2','R1'),('R2','R2','r1','r1'),('R2','R2','r1','R2')]


        self.transformation_keys = [x + y for y in TF_Y for x in TF_X]


        self.transformation_dict = {'R1':'UD','R2':'FB','r1':'DU','r2':'BF'}
        self.tf_invert = {'R1':'r1','R2':'r2','r1':'R1','r2':'R2','F':'F'}

        self.axis = {}
        
        self.myperms = {}
        self.myperms2 = {}        
        
        self.myperms2['C4[U.bR.R>bR.sR.R;UFL>FUR;URF>UFL;sR.R.bR>R.U.bR]+E3[UF>bR.R>UR]'] = ("R'", "U'", 'R', 'U')
        self.myperms2['C4[U.bR.R>URF;UFL>R.dR.F;URF>R.U.bR;dR.F.R>LUF]+E3[RF>RU>FU]'] = ("R","U'","R'","U")
        self.myperms2['C5[U.L.bL>LUF>sR.R.bR>R.U.bR>URF]+E3[UL>bR.R>UR]'] = ("R'", "U2'", 'R', 'U2')
        self.myperms2['C5[U.L.bL>R.dR.F>URF>R.U.bR>LUF]+E3[RF>RU>LU]'] = ("R","U2'","R'","U2")
        self.myperms2['C5[U.L.bL>URF>R.U.bR>sR.R.bR>LUF]+E3[UL>UR>bR.R]'] = ("U2'", "R'", 'U2', 'R')
        self.myperms2['C5[U.L.bL>LUF>R.U.bR>URF>R.dR.F]+E3[RF>LU>RU]'] = ("U2'", 'R', 'U2', "R'")
        self.myperms2['C6[R.sR.dR>bR.R.U>FLU;U.L.bL>sR.R.bR>URF]+E3[UL>sR.R>UR]'] = ("R2'", "U2'", 'R2', 'U2')
        self.myperms2['C6[R.sR.dR>FUR>bL.U.L;U.bR.R>UFL>dR.F.R]+E3[R.dR>RU>LU]'] = ("R2","U2'","R2'","U2")

        self.myperms2['C4[U.L.bL>bR.sR.R;UFL>FUR;URF>UFL;sR.R.bR>bL.U.L]+E3[UF>UL>R.bR]'] = ("U2'", 'bR', "U'", "bR'", "U2'")
        self.myperms2['C6[U.L.bL>URF;U.bL.bR>R.dR.F;U.bR.R>U.bL.bR;UFL>U.bR.R;URF>bL.U.L;dR.F.R>LUF]+E6p[3x2][RF>UL>UR;U.bL>UF>U.bR]'] = ('U2', "F'", "U'", 'F', 'U2')
        self.myperms2['C4[U.L.bL>R.bR.sR;UFL>URF;URF>FLU;sR.R.bR>L.bL.U]+E3[UF>R.bR>UL]'] = ('U2', 'bR', 'U', "bR'", 'U2')
        self.myperms2['C6[U.L.bL>UFL;U.bL.bR>FUR;U.bR.R>R.dR.F;UFL>U.bR.R;URF>U.bL.bR;dR.F.R>bL.U.L]+E6[RF>FU>Rb.U;U.bL>UL>UR]'] = ("U2","R","U'","R'","U2")
        self.myperms2['C4[U.L.bL>UFL;UFL>L.bL.U;URF>R.bR.sR;sR.R.bR>RFU]+E3[UF>bR.R>UL]'] = ("U2'", "R'", 'U', 'R', 'U')
        self.myperms2['C4[U.bR.R>URF;UFL>bR.sR.R;URF>R.U.bR;sR.R.bR>LUF]+E3[UF>R.bR>UR]'] = ("U'", 'bR', "U'", "bR'", 'U2')
        self.myperms2['C5[U.L.bL>bL.bR.U>FLU>URF>R.bR.sR]+E3[U.bL>UF>bR.R]'] = ('U2', "R'", 'U2', 'R', 'U')
        self.myperms2['C5[L.dL.sL>bR.R.U>RFU>FLU>L.F.dL]+E3[UF>dL.L>UR]'] = ('U2', "L2'", "U'", 'L2', "U'")
        self.myperms2['C5[U.L.bL>LUF>R.U.bR>URF>bR.sR.R]+E3[UL>UR>R.bR]'] = ('U2', 'bR', 'U2', "bR'", 'U')
        self.myperms2['C5[L.dL.sL>L.sL.bL>R.U.bR>FUR>LUF]+E3[L.sL>RU>FU]'] = ('U2', 'L2', "U'", "L2'", "U'")
        self.myperms2['C5[U.L.bL>URF>R.U.bR>dR.F.R>LUF]+E3[RF>UL>UR]'] = ("U'", "F'", 'U2', 'F', "U'")
        self.myperms2['C6[F.dR.dL>FLU>bR.R.U;U.L.bL>URF>dR.F.R]+E3[UL>UR>dR.F]'] = ("U'", "F2'", 'U2', 'F2', "U'")
        self.myperms2['C5[L.dL.sL>LUF>FUR>R.U.bR>L.sL.bL]+E3[L.sL>FU>RU]'] = ('U', 'L2', 'U', "L2'", "U2'")





        self.myperms2['C2[UFL>LUF;URF>RFU]'] = ('R', 'bR', "bL'", "bR2'", "R'", 'F', 'R', 'bR2', 'bL', "bR'", "R'", "F'")
        self.myperms2['C2[U.bR.R>bR.R.U;UFL>LUF]'] = ("L'", 'bL', 'L2', 'F', "R'", "F'", "L2'", "bL'", 'L', 'F', 'R', "F'")
        self.myperms2['C2[UFL>LUF;sR.R.bR>R.bR.sR]'] = ("R2'", 'U', 'L', "U'", 'R2', 'bR', "R'", 'U', "L'", "U'", 'R', "bR'")
        self.myperms2['C2[URF>RFU;bL.sL.B>B.bL.sL]'] = ("B'", "bR2'", 'U', 'F', "U'", 'bR2', "B'", 'bR2', 'U', "F'", "U'", "bR2'", 'B2')
        self.myperms2['C2[D.B.sL>B.sL.D;URF>FUR]'] = ('B2', 'bR2', "R'", "F'", 'R', "bR2'", 'B', "bR2'", "R'", 'F', 'R', 'bR2', 'B2')
        self.myperms2['C2[U.bR.R>R.U.bR;UFL>FLU]'] = self.invert_moves(self.myperms2['C2[U.bR.R>bR.R.U;UFL>LUF]'])

        self.myperms2['C3[U.bR.R>bR.R.U;UFL>FLU;URF>RFU]'] = ('F', "U'", "bR'", 'U', "F2'", "L'", 'F', "R2'", "F'", 'L', 'F', 'R2', "U'", 'bR', 'U')
        self.myperms2['C3[U.bL.bR>bL.bR.U;UFL>FLU;URF>RFU]'] = ("bR'","R2", "sR'", "R", "F2", "R'", "sR", "R", "F2'", "R", "U", "L", "U'", "R", "U", "L'", "U'","bR")




        

        self.myperms2['C3[U.bR.R>UFL>URF]'] = ('R2', "sR'", 'R', 'F2', "R'", 'sR', 'R', "F2'", 'R2')
        self.myperms2['C3[U.bR.R>LUF>URF]'] = ('R', 'bR', 'bL', "L'", "bL'", 'bR', 'bL', 'L', "bL'", "bR2'", "R'")
        self.myperms2['C3[U.bR.R>FLU>URF]'] = ('F', 'R', 'bR', "R'", "F'", 'R', "bR'", "R'")
        self.myperms2['C3[U.bR.R>FLU>FUR]'] = ('L', 'F', "L'", 'bR', 'bL', 'L', "F2'", "L'", "bL'", "bR'", 'L', 'F', "L'")
        self.myperms2['C3[U.bR.R>UFL>FUR]'] = ("L'", 'bR', 'bL', 'L', "F'", "L'", "bL'", "bR'", 'L', 'F')
        self.myperms2['C3[U.bR.R>LUF>FUR]'] = ("R'", 'dR', 'R2', 'U', 'L', "U'", "R2'", "dR'", 'U', "L'", "U'", 'R')
        self.myperms2['C3[U.bR.R>LUF>RFU]'] = ('F', "U'", "bR'", 'U', "F'", "U'", 'bR', 'U')
        self.myperms2['C3[U.bR.R>FLU>RFU]'] = ("F'", "L'", 'F', "R'", "F'", 'L', 'F', 'R')
        self.myperms2['C3[U.bR.R>UFL>RFU]'] = ("L'", 'R', 'bR', "bL'", "bR'", "R'", 'bR', 'bL', 'L', "bR'")
        
        self.myperms2['C3[U.bL.bR>FLU>RFU]'] = ("bL", "L", "F", "R", "F'", "L'", "F", "R'", "F'", "bL'")
        self.myperms2['C3[U.bL.bR>UFL>RFU]'] = ("R'", "F", "R", "bR'", "R'", "F2'", "R", "bR", "R'", "F", "R")
        self.myperms2['C3[U.bL.bR>LUF>RFU]'] = ("F'", "L2", "bL2", "L2'", "F'", "L2", "bL2'", "L2'", "F2")
        self.myperms2['C3[U.bL.bR>LUF>URF]'] = ("L", "F", "L'", "bL", "L", "F2'", "L'", "bL'", "L", "F", "L'")
        self.myperms2['C3[U.bL.bR>FLU>URF]'] = ("L'", "bL", "L", "F'", "L'", "bL'", "L", "F")
        self.myperms2['C3[U.bL.bR>UFL>URF]'] = ('F', 'R', "bR'", "R'", "F2'", 'L', 'F2', 'R', 'bR', "R'", "F2'", "L'", 'F')
        self.myperms2['C3[U.bL.bR>UFL>FUR]'] = ("F", "R", "bR'", "R'", "F'", "R", "bR", "R'")
        self.myperms2['C3[U.bL.bR>LUF>FUR]'] = ("bR'","F'", "L'", "F", "R'", "F'", "L", "F", "R","bR")
        self.myperms2['C3[U.bL.bR>FLU>FUR]'] = ("bR'", "R", "bR", "L'", "bL'", "bR'", "R'", "bR", "bL", "L")

        self.myperms2['C3[B.bR.bL>RFU>LUF]'] = ("U2'", 'bL', 'U2', 'F', "U2'", "bL'", 'U2', "F'")
        self.myperms2['C3[UFL>bR.sR.R>FUR]'] = ("R2'", 'U', 'L', "U'", 'R2', 'U', "L'", "U'")
        self.myperms2['C3[UFL>FUR>sR.R.bR]'] = ('bR', 'U', 'L', "U'", "R'", 'U', "L'", "U'", 'R', "bR'")
        self.myperms2['C3[UFL>R.bR.sR>URF]'] = ('bR', "R'", "F'", "L'", 'F', 'R', "F'", 'L', 'F', "bR'")
        self.myperms2['C3[UFL>sR.R.bR>URF]'] = ("R2'", "F'", "L'", "F", "R2", "F'", "L", "F")
        self.myperms2['C3[UFL>sR.R.bR>RFU]'] = ("U'", 'bR', 'U', 'L', 'F2', "U'", "bR'", 'U', "F2'", "L'")
        self.myperms2['C3[B.bR.bL>RFU>FLU]'] = ("U2'", 'bR', 'B', "bR'", "U'", 'bR', "B'", "bR'", "U2'")
        self.myperms2['C3[UFL>R.bR.sR>FUR]'] = ("U'", 'bR', 'U', 'F', "U'", "bR'", 'U', "F'")
        self.myperms2['C3[UFL>RFU>bR.sR.R]'] = ("U2'", 'R', 'sR', "R'", "U2'", 'F', "U'", 'R', "sR'", "R'", 'U', "F'", "U'")
        self.myperms2['C3[UFL>URF>sR.R.bR]'] = ("F'", "L'", 'F', "R2'", "F'", 'L', 'F', 'R2')
        self.myperms2['C3[U.bR.R>UFL>L.sL.bL]'] = ("U2'", 'bL', 'sL', "bL'", "U2'", 'bL', "sL'", "bL'", "U'")
        self.myperms2['C3[UFL>FUR>R.bR.sR]'] = ('F', "U'", 'bR', 'U', "F'", "U'", "bR'", 'U')
        self.myperms2['C3[UFL>bL.sL.B>FUR]'] = ("F'", "L'", "bL2'", 'L', 'F', "L'", 'bL2', 'L')
        self.myperms2['C3[U.bR.R>UFL>sL.bL.L]'] = ("L'", "bL'", 'bR', 'bL', 'L2', "bL'", "bR'", 'bL', "L'")
        self.myperms2['C3[U.bR.R>sL.bL.L>UFL]'] = ('L', "bL'", 'bR', 'bL', "L2'", "bL'", "bR'", 'bL', 'L')
        self.myperms2['C3[UFL>sR.R.bR>FUR]'] = ('bR', "U'", "bR'", 'U', 'F', "U'", 'bR', 'U', "bR'", "F'")
        self.myperms2['C3[UFL>RFU>R.bR.sR]'] = ("L'", 'R', 'bR', "bL'", "bR'", "R2'", 'bR', 'bL', "bR'", 'R', 'L')
        self.myperms2['C3[UFL>FUR>bR.sR.R]'] = ('U', 'L', "U'", "R2'", 'U', "L'", "U'", 'R2')
        self.myperms2['C3[UFL>R.bR.sR>RFU]'] = ('F', "dR'", "F2'", "U'", 'bR', 'U', 'F2', 'dR', "U'", "bR'", 'U', "F'")
        self.myperms2['C3[U.bR.R>L.sL.bL>LUF]'] = ('L2', 'F', "R'", "F'", "L2'", 'F', 'R', "F'")
        self.myperms2['C3[UFL>RFU>sR.R.bR]'] = ("R'", 'dR', 'R2', 'L', 'U', "L'", "U'", "R2'", "dR'", "R'", 'U', 'L', "U'", 'R2', "L'")
        self.myperms2['C3[U.bR.R>B.sR.bR>LUF]'] = ("R'", 'F', 'R', 'bR2', "R'", "F'", 'R', "bR2'")
        self.myperms2['C3[U.bR.R>sL.bL.L>LUF]'] = ("R'", 'U', "bL'", "U'", "F'", 'U', 'bL', "U'", 'F', 'R')
        self.myperms2['C3[U.bR.R>bL.L.sL>FLU]'] = ('bR', 'L', "bL'", "bR'", 'bL', "L2'", "bL'", 'bR', 'bL', "bR'", 'L')
        self.myperms2['C3[UFL>URF>R.bR.sR]'] = ('bR', 'U', 'F', "U'", "bR'", 'U', "F'", "U'", "F'", "L'", 'F', "R'", "F'", 'L', 'F', 'R')
        self.myperms2['C3[U.bR.R>LUF>L.sL.bL]'] = ('F', "R'", "F'", 'L2', 'F', 'R', "F'", "L2'")
        self.myperms2['C3[B.bR.bL>URF>FLU]'] = ("U'", "bR2'", 'U', 'F', "U'", 'bR2', 'U', "F'")
        self.myperms2['C3[UFL>sL.B.bL>FUR]'] = ('sL2', 'dL', 'F', 'U', "F'", "dL'", 'F', "U'", "F'", "U'", "R'", 'U', "L'", "U'", 'R', 'U', 'L', "sL2'")
        self.myperms2['C3[D.B.sL>U.bR.R>LUF]'] = ('sL2', 'F', "R'", "F'", 'L2', 'F', 'R', "F'", "L2'", "sL2'")
        self.myperms2['C3[UFL>URF>bR.sR.R]'] = ("U2'", 'R', 'sR', "R'", 'U', 'R', "sR'", "R'", 'U')
        self.myperms2['C3[B.bR.bL>FLU>URF]'] = ('F', 'bL', 'R', "bR'", "R'", "F'", 'R', 'bR', "R'", "bL'")
        self.myperms2['C3[U.bR.R>B.sR.bR>UFL]'] = ('bR', 'bL', "L'", "bL'", 'bR2', 'bL', 'L', "bL'", 'bR2')
        self.myperms2['C3[UFL>sR.bR.B>URF]'] = ("U'", 'R', "sR'", "R'", "U'", 'R', 'sR', "R'", 'U2')
        self.myperms2['C3[D.sR.B>FLU>bR.R.U]'] = ("sR'", 'bR', 'bL', "L'", "bL'", 'bR2', 'bL', 'L', "bL'", 'bR2', 'sR')
        self.myperms2['C3[D.dR.sR>FLU>bR.R.U]'] = ('D', "sR'", 'bR', 'bL', "L'", "bL'", 'bR2', 'bL', 'L', "bL'", 'bR2', 'sR', "D'")
        self.myperms2['C3[D.B.sL>FLU>bR.R.U]'] = ("D'", "sR'", 'bR', 'bL', "L'", "bL'", 'bR2', 'bL', 'L', "bL'", 'bR2', 'sR', 'D')
        self.myperms2['C3[D.dL.dR>FLU>bR.R.U]'] = ('D2', "sR'", 'bR', 'bL', "L'", "bL'", 'bR2', 'bL', 'L', "bL'", 'bR2', 'sR', "D2'")
        self.myperms2['C3[D.sL.dL>FLU>bR.R.U]'] = ("D2'", "sR'", 'bR', 'bL', "L'", "bL'", 'bR2', 'bL', 'L', "bL'", 'bR2', 'sR', 'D2')
        self.myperms2['C3[D.sR.B>LUF>FUR]'] = ("sR2'", "F'", "U'", 'bR', 'U', 'F', "U'", "bR'", "R'", "dR'", 'R', 'U', "R'", 'dR', 'R', 'sR2')
        self.myperms2['C3[D.B.sL>LUF>FUR]'] = ("D'", "sR2'", "F'", "U'", 'bR', 'U', 'F', "U'", "bR'", "R'", "dR'", 'R', 'U', "R'", 'dR', 'R', 'sR2', 'D')
        self.myperms2['C3[UFL>B.sR.bR>FUR]'] = ("B2'", "sL2'", "U'", "F'", 'dL', 'F', 'U', "F'", "dL'", "L'", "bL'", 'L', 'F', "L'", 'bL', 'L', 'sL2', 'B2')
        self.myperms2['C3[UFL>B.bL.sL>URF]'] = ("U2'", 'bR', "B'", "bR'", "U'", 'bR', 'B', "bR'", "U2'")
        self.myperms2['C3[B.bR.bL>URF>LUF]'] = ('bL', "L'", "bL'", "bR'", 'R', 'bR', 'bL', 'L', "bR'", "R'", 'bR', "bL'")
        self.myperms2['C3[D.sR.B>FLU>R.U.bR]'] = ("sR2'", "R'", "U'", 'R', 'sR2', "R'", 'U', 'F', 'R', 'bR', "R'", "F'", 'R', "bR'")
        self.myperms2['C3[L.dL.sL>R.U.bR>LUF]'] = ("L'", 'bR', 'bL', 'L', "F'", 'dL', 'F', "L'", "bL'", 'L', "bR'", "F'", "dL'", 'F')
        self.myperms2['C3[D.dL.dR>bR.R.U>UFL]'] = ("dL2'", 'F', 'U', "F'", 'dL2', 'F', "U'", "R'", "F'", "L'", 'F', 'R', "F'", 'L')
        self.myperms2['C3[U.bR.R>UFL>bL.L.sL]'] = ("U2'", "L'", "sL'", 'L', "U2'", "L'", 'sL', 'L', "U'")
        self.myperms2['C3[UFL>sR.bR.B>FUR]'] = ("sR'", "U'", 'bR', 'U', 'F', "U'", "bR'", 'U', "F'", 'sR')
        self.myperms2['C3[D.sL.dL>sR.R.bR>FLU]'] = ("R2'", "sL2'", "U'", "R'", 'U', 'L2', "U'", 'R', 'U', "L2'", 'sL2', 'R2')
        self.myperms2['C3[UFL>bL.sL.B>sR.R.bR]'] = ("R2'", 'sL2', 'F', 'R', "F'", "L2'", 'F', "R'", "F'", 'L2', "sL2'", 'R2')
        self.myperms2['C3[D.B.sL>R.bR.sR>LUF]'] = ("R2'", 'sL2', "U'", "R'", 'U', 'L2', "U'", 'R', 'U', "L2'", "sL2'", 'R2')
        self.myperms2['C3[D.sL.dL>UFL>R.bR.sR]'] = ("U2'", "D2'", 'bR', 'U', "bR'", "sR2'", 'bR', "U'", "bR'", 'sR2', 'D2', 'U2')
        self.myperms2['C3[UFL>sR.R.bR>bL.sL.B]'] = ("bR2'", 'bL2', "U'", "F'", 'U', 'bL', "U'", 'F', 'U', 'bL2', 'bR2')
        self.myperms2['C3[D.sL.dL>FLU>sR.R.bR]'] = ("R2'", 'dL2', "L'", "U'", "R'", 'U', 'L', "U'", 'R', 'U', "dL2'", 'R2')
        self.myperms2['C3[U.bR.R>sR.bR.B>FLU]'] = ('F', "U'", 'R', "sR'", "R'", 'U', 'R', 'sR', "R'", "F'")
        self.myperms2['C3[U.bR.R>FLU>sR.bR.B]'] = ('F', 'R', "sR'", "R'", "U'", 'R', 'sR', "R'", 'U', "F'")
        self.myperms2['C3[U.bR.R>bL.L.sL>UFL]'] = ("U'", 'L', 'U', 'bR', "U'", "L'", 'U', 'bL', "L'", "bL'", "bR'", 'bL', 'L', "bL'")
        self.myperms2['C3[U.bR.R>LUF>bL.L.sL]'] = ("bL'", "U'", "F'", 'U', 'bL', "U'", 'F', 'U', 'F', "R'", "F'", 'L', 'F', 'R', "F'", "L'")
        self.myperms2['C3[U.bR.R>bL.L.sL>LUF]'] = ('U', 'bL', 'sL', "bL'", "U'", 'bR', "U'", 'bL', "sL'", "bL'", 'U', "bR'")
        self.myperms2['C3[UFL>B.bL.sL>FUR]'] = ("F'", "L2'", 'sL', 'L2', 'F', "L2'", "sL'", 'L2')
        self.myperms2['C3[UFL>FUR>bR.B.sR]'] = ('F', "U'", 'bR2', 'U', "F'", "U'", "bR2'", 'U')
        self.myperms2['C3[D.sR.B>bR.R.U>UFL]'] = ("bR'", "sR'", 'bR2', "R'", 'F', 'R', "bR2'", 'sR', 'bR', "R'", "F'", 'R')
        self.myperms2['C3[U.bR.R>sL.bL.L>FLU]'] = ('L', "bL'", 'bR', 'bL', 'F', "L'", "bL'", "bR'", 'bL', 'L', "F'", "L'")
        self.myperms2['C3[UFL>bR.B.sR>URF]'] = ('bR', "U'", "bR'", "sR'", 'bR', 'U', "bR'", "R2'", 'F', 'R2', 'sR', "R2'", "F'", 'R2')
        self.myperms2['C3[U.bR.R>L.sL.bL>UFL]'] = ('L2', 'F', "R'", "F'", 'L2', 'bL', 'L2', 'F', 'R', "F'", "L2'", "bL'", 'L')
        self.myperms2['C3[UFL>B.sR.bR>RFU]'] = ('R', 'bR2', "R'", "U'", "R'", 'U', "F'", 'R', "bR2'", "R'", 'F', "U'", 'R', 'U')
        self.myperms2['C3[U.bR.R>LUF>sR.bR.B]'] = ('F', 'bR2', "R'", "F'", 'R', "bR2'", "R'", "L'", 'bL', 'L2', 'F', 'R', "F'", "L2'", "bL'", 'L')







        self.myperms2['C2[UFL>FLU;URF>FUR]+E2[UF>FU;UR>RU]'] = ('F2', "R2'", "F'", 'R', 'F', 'R', "F2'", "U'", "R'", 'U', 'R', "U'", "R'", 'U', "L'", "U'", 'R', 'U', 'L')

        self.myperms2['C3[U.L.bL>bR.R.U>RFU]+E3[UF>LU>UR]'] = ('L', "F'", "L'", 'F', 'U', 'F', "U'", "F'")
        self.myperms2['C3[U.L.bL>RFU>bR.R.U]+E3[UF>UR>LU]'] = ('F', 'U', "F'", "U'", "F'", 'L', 'F', "L'")
        self.myperms2['C3[U.L.bL>RFU>bR.R.U]+E3[UL>dR.F>RU]'] = ("R'", "F'", 'R', "F'", "U'", 'F', 'U', 'F')
        self.myperms2['C3[U.L.bL>bR.R.U>RFU]+E3[UL>RU>dR.F]'] = ("F'", "U'", "F'", 'U', 'F', "R'", 'F', 'R')

        self.myperms2['C4[U.bR.R>R.dR.F;UFL>RFU;URF>UFL;dR.F.R>U.bR.R]+E2[UF>FU;UR>RU]'] = ("R","U'","R'","U","R'","F","R","F'")
        self.myperms2['C4[U.bR.R>dR.F.R;UFL>URF;URF>LUF;dR.F.R>bR.R.U]+E2[UF>FU;UR>RU]'] = ("F","R'","F'","R","U'","R","U","R'")

        self.myperms2['C4[L.dL.sL>L.bL.U;U.L.bL>L.dL.sL;U.bR.R>URF;URF>R.U.bR]+E3[L.sL>RU>FU]'] = ('L', 'U2', 'L', "U'", "L'", "U'", "L'")
        self.myperms2['C4[U.bL.bR>B.sR.bR;UFL>FUR;URF>UFL;bR.B.sR>U.bL.bR]+E3[B.bR>UR>UF]'] = ("bR'", "U'", "bR'", "U'", 'bR', 'U2', 'bR')
        self.myperms2['C4[U.L.bL>UFL;U.bR.R>URF;UFL>L.bL.U;URF>R.U.bR]+E3[UF>RU>UL]'] = ("R'", "U'", "F'", 'U', 'F', 'R')
        self.myperms2['C4[U.L.bL>LUF;U.bR.R>RFU;UFL>U.L.bL;URF>U.bR.R]+E3[UF>UL>RU]'] = ("R'", "F'", "U'", 'F', 'U', 'R')
        self.myperms2['C5[U.L.bL>U.bR.R>LUF>FUR>bL.bR.U]+E3[U.bL>UR>FU]'] = ('F', 'U2', 'R', "U2'", "R'", "F'")
        self.myperms2['C5[U.L.bL>bL.bR.U>FUR>LUF>U.bR.R]+E3[U.bL>FU>UR]'] = ('F', 'R', 'U2', "R'", "U2'", "F'")
        self.myperms2['C4[U.L.bL>UFL;U.bR.R>URF;UFL>L.bL.U;URF>R.U.bR]+E3[U.bL>RU>FU]'] = ('bL', 'U2', 'L', "U'", "L'", "U'", "bL'")
        self.myperms2['C4[U.L.bL>LUF;U.bR.R>RFU;UFL>U.L.bL;URF>U.bR.R]+E3[U.bL>FU>RU]'] = ('bL', 'U', 'L', 'U', "L'", "U2'", "bL'")
        self.myperms2['C4[U.L.bL>RFU;U.bR.R>FLU;UFL>U.bR.R;URF>L.bL.U]+E3[UF>UL>UR]'] = ('L', 'U2', "L'", "U'", 'L', "U'", "L'")
        self.myperms2['C4[U.L.bL>FUR;U.bR.R>LUF;UFL>U.bR.R;URF>bL.U.L]+E3[UF>LU>RU]'] = ('L', 'F', "R'", 'F', 'R', "F2'", "L'")
        self.myperms2['C4[U.L.bL>URF;U.bR.R>FLU;UFL>bR.R.U;URF>L.bL.U]+E3[UF>UL>UR]'] = ("R'", "U'", 'R', "U'", "R'", 'U2', 'R')
        self.myperms2['C5[U.L.bL>U.bR.R>URF>UFL>U.bL.bR]+E3[U.bL>UR>UF]'] = ('bL2', "U2'", "bL2'", "U'", 'bL2', "U2'", "bL2'")
        self.myperms2['C4[U.L.bL>bR.R.U;U.bR.R>URF;UFL>FLU;URF>L.bL.U]+E3[UF>UL>RU]'] = ("R'", "U'", "F'", 'U', 'F2', 'R', "F'", "U'", "F'", 'U', 'F')
        self.myperms2['C4[U.bR.R>sR.bR.B;UFL>URF;URF>FLU;bR.B.sR>U.bR.R]+E3[UF>UR>bR.R]'] = ("U'", "sR'", "R'", 'U', 'R', 'sR')
        self.myperms2['C3[U.bL.bR>LUF>U.bR.R]+E3[F.dL>RU>UL]'] = ('F', 'U', 'F', "U'", "F'", 'bL', 'L', "F'", "L'", "bL'")
        self.myperms2['C4[U.bR.R>bR.B.sR;UFL>FUR;URF>UFL;bR.B.sR>bR.R.U]+E3[UF>bR.R>UR]'] = ("sR'", "R'", "U'", 'R', 'U', 'sR')
        self.myperms2['C3[U.bL.bR>U.bR.R>LUF]+E3[F.dL>UL>RU]'] = ('bL', 'L', 'F', "L'", "bL'", 'F', 'U', "F'", "U'", "F'")
        self.myperms2['C3[U.L.bL>URF>FLU]+E4[U.bL>UF;UF>UR;UL>LU;UR>bL.U]'] = ("U'", 'bR', 'bL', 'L', 'U', "L'", "bL'", "U'", "bR'")
        self.myperms2['C3[U.bR.R>URF>UFL]+E3[UF>UL>UR]'] = ('L', 'U2', "L'", "U'", 'L', "U2'", "R'", 'U', "L'", "U'", 'R', 'U')
        self.myperms2['C5[U.L.bL>LUF>R.U.bR>sR.R.bR>FUR]+E4[UF>FU;UL>RU;UR>bR.R;bR.R>UL]'] = ("U'", "R'", 'U', "F'", "U'", 'F', 'U', 'R')
        self.myperms2['C3[U.L.bL>FLU>URF]+E4[U.bL>RU;UF>U.bL;UL>LU;UR>UF]'] = ('U', 'R', 'U', 'bR', 'bL', "U'", "bL'", "bR'", "R'")
        self.myperms2['C4[U.L.bL>UFL;U.bR.R>FUR;UFL>L.bL.U;URF>U.bR.R]+E4[U.bL>UL;UF>bL.U;UL>UF;UR>RU]'] = ("U'", 'R', "bR'", "R'", 'bR', 'U', 'bR', "U'", "bR'")
        self.myperms2['C5[U.L.bL>FUR>sR.R.bR>R.U.bR>LUF]+E4[UF>FU;UL>bR.R;UR>LU;bR.R>UR]'] = ("R'", "U'", "F'", 'U', 'F', "U'", 'R', 'U')
        self.myperms2['C3[U.bR.R>UFL>URF]+E3[UF>UR>UL]'] = ("U'", 'bR', 'U2', "bR'", "U'", 'bR', "U2'", "L'", 'U', "bR'", "U'", 'L', 'U')
        self.myperms2['C3[U.L.bL>RFU>R.U.bR]+E3[UF>RU>LU]'] = ("R'", "F'", 'L', "F'", "L'", 'F', "L'", "bL'", 'L', 'F', 'R', "L'", 'bL', 'L')
        self.myperms2['C4[U.bL.bR>LUF;U.bR.R>URF;UFL>bR.U.bL;URF>R.U.bR]+E3[UF>LU>UR]'] = ('bL', 'L', 'U', 'F', "U'", "F'", "L'", "bL'")
        self.myperms2['C3[U.L.bL>URF>R.dR.F]+E3[RF>LU>RU]'] = ("U2'", 'R', 'U2', 'bR', "L'", "bL'", "bR'", "R'", 'bR', 'bL', 'L', "bR'")
        self.myperms2['C3[U.bL.bR>RFU>dR.F.R]+E3[RF>RU>UL]'] = ("F'", "U'", 'F', 'U', "R'", "U'", "F'", 'U', 'R', "bR'", "R'", 'F', 'R', 'bR')
        self.myperms2['C4[U.L.bL>URF;U.bR.R>LUF;UFL>R.U.bR;URF>bL.U.L]+E3[UF>LU>RU]'] = ("R'", "F2'", 'L', 'F', "L'", 'F', 'R')
        self.myperms2['C3[U.L.bL>R.dR.F>URF]+E3[RF>RU>LU]'] = ('U2', "L'", "U2'", 'L', 'F2', "R'", "F2'", 'R')
        self.myperms2['C4[U.bR.R>sR.bR.B;UFL>URF;URF>FLU;bR.B.sR>U.bR.R]+E3[UF>UR>R.bR]'] = ('bR2', "U2'", "bR'", 'U', 'bR', 'U', "bR2'")
        self.myperms2['C3[U.bR.R>URF>UFL]+E3[U.bL>UR>UF]'] = ('U2', "bR2'", 'U2', 'bR2', 'U', "bR2'", 'U2', 'bR2')
        self.myperms2['C2[U.bR.R>bR.R.U;UFL>LUF]+E3[RF>UR>UF]'] = ('R', "U'", "R'", 'U2', "F'", "U'", 'F', "R'", 'F', 'R', "F'")
        self.myperms2['C3[U.L.bL>UFL>URF]+E4s[U.bL<>UL;UF<>UR]'] = ('U', "L2'", 'U2', 'L2', 'U', "L2'", 'U2', 'L2')
        self.myperms2['C3[U.bR.R>LUF>RFU]+E3[RF>RU>UF]'] = ("R'", "dR'", 'R', "U'", "R'", 'dR', 'R', 'U', 'F', "R'", "F'", 'R')
        self.myperms2['C4[UFL>B.sR.bR;URF>R.bR.sR;bR.B.sR>UFL;sR.R.bR>RFU]+E3[UF>sR.bR>UR]'] = ('bR', "U'", 'bR', "U'", "bR'", 'U2', "bR'")
        self.myperms2['C3[L.dL.sL>U.bR.R>bR.sR.R]+E3[UF>bR.R>dL.L]'] = ("R'", "L'", "U'", 'R', 'U2', "L'", 'U', 'L', "U2'", 'L')
        self.myperms2['C3[L.dL.sL>bR.sR.R>U.bR.R]+E3[UF>dL.L>bR.R]'] = ("L'", 'U2', "L'", "U'", 'L', "U2'", "R'", 'U', 'L', 'R')
        self.myperms2['C5[B.bR.bL>UFL>URF>sR.bR.B>U.bL.bR]+E3[UF>UR>UL]'] = ("bR2'", "sR'", "U'", 'sR', 'bR2', 'U2', "bR2'", "U'", 'bR2', "U'")
        self.myperms2['C2[U.bR.R>R.U.bR;UFL>FLU]+E3[RF>RU>FU]'] = ("R'", 'F', 'R', "F'", 'R', "U'", "R'", 'U2', "F'", "U'", 'F')
        self.myperms2['C4[U.bR.R>R.bR.sR;UFL>FUR;URF>LUF;sR.R.bR>bR.R.U]+E3[U.bL>R.bR>UF]'] = ('bR', 'U2', "bR'", 'U2', 'bR', 'U2', "bR'", 'U')
        self.myperms2['C4[U.bR.R>bR.sR.R;UFL>RFU;URF>FLU;sR.R.bR>R.U.bR]+E3[U.bL>UF>R.bR]'] = ("U'", 'bR', "U2'", "bR'", "U2'", 'bR', "U2'", "bR'")
        self.myperms2['C5[U.bR.R>B.bL.sL>UFL>L.sL.bL>RFU]+E3[UL>UR>sL.bL]'] = ("bL'", 'U', "bL'", 'U2', 'bL', 'U2', 'bL')
        self.myperms2['C5[F.dR.dL>UFL>L.bL.U>RFU>U.bR.R]+E3[RF>UL>UR]'] = ("U'", 'dR', "F'", 'U2', 'F', "U'", "dR'")
        self.myperms2['C2[UFL>LUF;URF>RFU]+E3[RF>UR>FU]'] = ('R', "U'", "R'", 'U', "F'", 'U', 'F', "U'", 'F', "R'", "F'", 'R')
        self.myperms2['C6[F.dR.dL>bL.U.L>dR.F.R;U.bR.R>dL.L.F>FUR]+E3[F.dL>LU>RU]'] = ("F'", "U'", "F2'", 'U2', 'F2', "U'", 'F')
        self.myperms2['C5[U.L.bL>dR.F.R>R.U.bR>URF>UFL]+E4[RF>UR;UF>UL;UL>FU;UR>FR]'] = ("F'", "U'", 'F', 'R', 'U', "R'")
        self.myperms2['C4s[U.L.bL<>dL.L.F;U.bR.R<>URF]+E4s[UF<>UR;UL<>dL.L]'] = ("L2'", 'U', 'L2', 'U', "L2'", "U'", 'L2', "U'")
        self.myperms2['C4[U.L.bL>LUF;UFL>U.L.bL;URF>bR.B.sR;bR.B.sR>RFU]+E3[UF>UL>bR.R]'] = ("sR'", "U'", "R'", "U'", 'R', 'U2', 'sR')
        self.myperms2['C4s[U.L.bL<>dR.F.R;U.bR.R<>URF]+E4s[UF<>UR;UL<>dR.F]'] = ("U'", "F2'", 'U', 'F2', 'U', "F2'", "U'", 'F2')
        self.myperms2['C4[U.bL.bR>LUF;UFL>bR.sR.R;URF>RFU;sR.R.bR>bL.bR.U]+E3[UF>R.bR>RU]'] = ("bR'", 'R', 'bR', "R'", 'bR', "U2'", "bR'", 'U2')
        self.myperms2['C4[D.dR.sR>bR.R.U;U.bR.R>dR.sR.D;UFL>URF;URF>FLU]+E3[UF>UR>bR.R]'] = ("dR'", "U'", 'sR', "R'", 'U', 'R', "sR'", 'dR')
        self.myperms2['C3[U.L.bL>R.bR.sR>UFL]+E3[UF>R.bR>LU]'] = ('L', "U'", "R'", 'U', "L'", "F'", "U'", 'F', 'R', 'U')



        self.myperms2['C3[U.bR.R>R.dR.F>UFL]'] = ("U'", 'F', 'dR', "F'", 'U2', 'F', "dR'", "F'", "U'")
        self.myperms2['C3[U.bR.R>F.R.dR>UFL]'] = ("U'", "R'", "dR'", 'R', 'U2', "R'", 'dR', 'R', "U'")
        self.myperms2['C3[U.bR.R>F.R.dR>LUF]'] = ('U2', "R2'", "U2'", "R'", 'F', 'R', 'bR', "R'", "F'", 'R', "bR'", 'U2', 'R2', "U2'")
        self.myperms2['C3[U.bR.R>R.dR.F>FLU]'] = ("U'", "R'", "dR'", 'R', "bR'", 'U', "R'", 'dR', 'R', "U'", 'bR', 'U')
        self.myperms2['C3[U.bR.R>dR.F.R>UFL]'] = ("U2'", 'R2', 'U2', "bR2'", "U2'", 'bR2', "R2'", "bR2'", 'U2', 'bR2')
        self.myperms2['C3[U.bR.R>dR.F.R>LUF]'] = ("R'", 'U', 'L', "U'", 'R2', 'U', "L'", "U'", "R'")
        self.myperms2['C3[U.bR.R>F.R.dR>FLU]'] = ("F'", 'R', 'bR', "R'", 'F2', 'R', "bR'", "R'", "F'")
        self.myperms2['C3[U.bR.R>dR.F.R>FLU]'] = ("R'", "F'", "L'", 'F', 'R2', "F'", 'L', 'F', "R'")
        self.myperms2['C3[U.bR.R>R.dR.F>LUF]'] = ("F'", "U'", "bR'", 'U', 'F2', "U'", 'bR', 'U', "F'")

        
        self.myperms2['E3[UF>UR>UL]'] = ("L","U","F","U'","F'","L'","R'","F'","U'","F","U","R")
        self.myperms2['E3[UF>RU>UL]'] = ("L", "U2", "F", "U'", "F'", "U'", "L'", "F", "R", "U", "R'", "U'", "F'")
        self.myperms2['E3[UF>RU>LU]'] = ("R'", "F'", "U'", "F", "U", "R", "L", "U", "F", "U'", "F'", "L'")
        self.myperms2['E3[UF>UR>LU]'] = ("F'", "U'", "L'", "U", "L", "F", "R'", "U'", "F'", "U'", "F", "U2", "R")

        self.myperms2['E3[U.bL>FU>RU]'] = ('F', 'U', 'R', 'U', "R'", "U2'", "F'", 'R', 'U', 'bR', "U'", "bR'", "R'")
        self.myperms2['E3[U.bL>UF>UR]'] = ("bL'", "bR'", 'F', 'U2', 'R', "U'", "R'", "U'", "F'", 'R', 'bR', 'U', "bR'", "U'", "R'", 'bR', 'bL')
        self.myperms2['E3[U.bL>FU>UR]'] = ("bL'", "bR2'", "R'", "U'", 'R', 'U', 'bR', 'F', 'U', 'R', "U'", "R'", "F'", 'bR', 'bL')
        self.myperms2['E3[U.bL>UF>RU]'] = ("bL'", "bR'", "R'", "U'", "F'", 'U', 'F', 'R', "bR'", "U'", "R'", "U'", 'R', 'U2', 'bR2', 'bL')

        self.myperms2['E3[RF>UR>UL]'] = ("F'", 'L', 'U', 'F', "U'", "F'", "L'", "R'", "F'", "U'", 'F', 'U', 'R', 'F')
        self.myperms2['E3[RF>LU>RU]'] = ("U'", 'R', "bR'", "U'", "R'", 'U', 'R', 'bR', 'F', 'R', 'U', "R'", "U'", "F'", "R'", 'U')
        self.myperms2['E3[UL>bR.R>UR]'] = ("U'", "R'", 'F', 'U', 'R', "U'", "R'", "F'", "bR'", "R'", "U'", 'R', 'U', 'bR', 'R', 'U')
        self.myperms2['E3[UL>UR>R.bR]'] = ("U2'", 'bR', "bL'", "U'", "bR'", 'U', 'bR', 'bL', 'R', 'bR', 'U', "bR'", "U'", "R'", "bR'", 'U2')
        self.myperms2['E3[U.bL>bR.R>UF]'] = ("U2'", "R'", 'F', 'U', 'R', "U'", "R'", "F'", "bR'", "R'", "U'", 'R', 'U', 'bR', 'R', 'U2')
        self.myperms2['E3[UF>UR>R.bR]'] = ('bR', 'F', 'U', 'R', "U'", "R'", "F'", "bR'", "R'", "U'", 'R', 'U')
        self.myperms2['E3[RF>FU>RU]'] = ('U', 'R', 'L', 'U', 'F', "U'", "F'", "L'", "R'", "F'", "U'", 'F')
        self.myperms2['E3[UF>R.bR>UL]'] = ("U'", 'bR', 'F', 'U', 'R', "U'", "R'", "F'", "bR'", "R'", "U'", 'R', 'U2')
        self.myperms2['E3[UF>bR.R>UR]'] = ("U2'", "R'", "bL'", "U'", "bR'", 'U', 'bR', 'bL', 'R', 'bR', 'U', "bR'", 'U')
        self.myperms2['E3[UF>UL>bR.R]'] = ('U2', "R'", "bL'", "U'", "bR'", 'U', 'bR', 'bL', 'R', 'bR', 'U', "bR'", 'U2')

        self.myperms2['E3[UF>UR>sR.bR]'] = ('bR2', 'F', 'U', 'R', "U'", "R'", "F'", "bR'", "R'", "U'", 'R', 'U', "bR'")
        self.myperms2['E3[RF>RU>LU]'] = ("R'", "U2'", "dR'", "R'", "F'", 'R', 'F', 'dR', 'U', 'F', 'R', "F'", "R'", 'U', 'R')
        self.myperms2['E3[UF>UR>B.bL]'] = ("U'", 'bL2', 'R', 'U', 'bR', "U'", "bR'", "R'", "bL'", "bR'", "U'", 'bR', 'U', "bL'", 'U')
        self.myperms2['E3[UL>UR>bR.R]'] = ("R2'", "U2'", "dR'", "R'", "F'", 'R', 'F', 'dR', 'U', 'F', 'R', "F'", "R'", 'U', 'R2')
        self.myperms2['E3[L.sL>FU>RU]'] = ("U2'", 'L2', 'bR', 'U', 'bL', "U'", "bL'", "bR'", "L'", "bL'", "U'", 'bL', 'U', "L'", 'U2')
        self.myperms2['E3[UF>UR>sR.B]'] = ("sR'", 'bR2', 'F', 'U', 'R', "U'", "R'", "F'", "bR'", "R'", "U'", 'R', 'U', "bR'", 'sR')
        self.myperms2['E3[UF>UR>dR.sR]'] = ("U","dR'","R2","L","U","F","U'","F'","L'","R'","F'","U'","F","U","R'","dR","U'")
        self.myperms2['E3[B.sL>UF>UR]'] = ("U'", "B'", 'bL2', 'R', 'U', 'bR', "U'", "bR'", "R'", "bL'", "bR'", "U'", 'bR', 'U', "bL'", 'B', 'U')
        self.myperms2['E3[UF>UR>dL.dR]'] = ('U2', "dL'", 'F2', 'bL', 'U', 'L', "U'", "L'", "bL'", "F'", "L'", "U'", 'L', 'U', "F'", 'dL', "U2'")
        self.myperms2['E3[UF>UR>sL.dL]'] = ("U2'", "sL'", 'L2', 'bR', 'U', 'bL', "U'", "bL'", "bR'", "L'", "bL'", "U'", 'bL', 'U', "L'", 'sL', 'U2')
        self.myperms2['E3[D.sR>FU>RU]'] = ("sR2'", 'bR2', 'F', 'U', 'R', "U'", "R'", "F'", "bR'", "R'", "U'", 'R', 'U', "bR'", 'sR2')
        self.myperms2['E3[D.dR>FU>RU]'] = ("U","dR2'","R2","L","U","F","U'","F'","L'","R'","F'","U'","F","U","R'","dR2","U'")
        self.myperms2['E3[DB>FU>RU]'] = ("U'", "B2'", 'bL2', 'R', 'U', 'bR', "U'", "bR'", "R'", "bL'", "bR'", "U'", 'bR', 'U', "bL'", 'B2', 'U')
        self.myperms2['E3[D.dL>FU>RU]'] = ('U2', "dL2'", 'F2', 'bL', 'U', 'L', "U'", "L'", "bL'", "F'", "L'", "U'", 'L', 'U', "F'", 'dL2', "U2'")
        self.myperms2['E3[D.sL>FU>RU]'] = ("U2'", "sL2'", 'L2', 'bR', 'U', 'bL', "U'", "bL'", "bR'", "L'", "bL'", "U'", 'bL', 'U', "L'", 'sL2', 'U2')

        self.myperms2['E3[UL>dR.F>UR]'] = ("F2'", 'L', 'U', 'F', "U'", "F'", "L'", "R'", "F'", "U'", 'F', 'U', 'R', 'F2')
        self.myperms2['E3[UL>dL.L>UR]'] = ('U', "L2'", 'bL', 'U', 'L', "U'", "L'", "bL'", "F'", "L'", "U'", 'L', 'U', 'F', 'L2', "U'")
        self.myperms2['E3[UL>sR.R>UR]'] = ("U'", "R2'", 'F', 'U', 'R', "U'", "R'", "F'", "bR'", "R'", "U'", 'R', 'U', 'bR', 'R2', 'U')
        self.myperms2['E3[UL>sL.bL>UR]'] = ('U2', "bL2'", 'bR', 'U', 'bL', "U'", "bL'", "bR'", "L'", "bL'", "U'", 'bL', 'U', 'L', 'bL2', "U2'")
        self.myperms2['E3[B.bR>UR>UL]'] = ("U2'", "bR2'", 'R', 'U', 'bR', "U'", "bR'", "R'", "bL'", "bR'", "U'", 'bR', 'U', 'bL', 'bR2', 'U2')
        self.myperms2['E3[UL>dR.dL>UR]'] = ('dR', "F2'", 'L', 'U', 'F', "U'", "F'", "L'", "R'", "F'", "U'", 'F', 'U', 'R', 'F2', "dR'")
        self.myperms2['E3[UL>dL.sL>UR]'] = ('U', 'dL', "L2'", 'bL', 'U', 'L', "U'", "L'", "bL'", "F'", "L'", "U'", 'L', 'U', 'F', 'L2', "dL'", "U'")
        self.myperms2['E3[UL>sR.dR>UR]'] = ("U'", 'sR', "R2'", 'F', 'U', 'R', "U'", "R'", "F'", "bR'", "R'", "U'", 'R', 'U', 'bR', 'R2', "sR'", 'U')
        self.myperms2['E3[B.sL>RU>LU]'] = ('U2', 'sL', "bL2'", 'bR', 'U', 'bL', "U'", "bL'", "bR'", "L'", "bL'", "U'", 'bL', 'U', 'L', 'bL2', "sL'", "U2'")
        self.myperms2['E3[UL>B.sR>UR]'] = ("U2'", 'B', "bR2'", 'R', 'U', 'bR', "U'", "bR'", "R'", "bL'", "bR'", "U'", 'bR', 'U', 'bL', 'bR2', "B'", 'U2')
        self.myperms2['E3[D.dR>RU>LU]'] = ('dR2', "F2'", 'L', 'U', 'F', "U'", "F'", "L'", "R'", "F'", "U'", 'F', 'U', 'R', 'F2', "dR2'")
        self.myperms2['E3[D.dL>RU>LU]'] = ('U', 'dL2', "L2'", 'bL', 'U', 'L', "U'", "L'", "bL'", "F'", "L'", "U'", 'L', 'U', 'F', 'L2', "dL2'", "U'")
        self.myperms2['E3[D.sR>RU>LU]'] = ("U'", 'sR2', "R2'", 'F', 'U', 'R', "U'", "R'", "F'", "bR'", "R'", "U'", 'R', 'U', 'bR', 'R2', "sR2'", 'U')
        self.myperms2['E3[D.sL>RU>LU]'] = ('U2', 'sL2', "bL2'", 'bR', 'U', 'bL', "U'", "bL'", "bR'", "L'", "bL'", "U'", 'bL', 'U', 'L', 'bL2', "sL2'", "U2'")
        self.myperms2['E3[DB>RU>LU]'] = ("U2'", 'B2', "bR2'", 'R', 'U', 'bR', "U'", "bR'", "R'", "bL'", "bR'", "U'", 'bR', 'U', 'bL', 'bR2', "B2'", 'U2')

        self.myperms2['E3[L.bL>FU>R.bR]'] = ("R'", 'L2', 'U', 'F', "U'", "F'", "L'", "R'", "F'", "U'", 'F', 'U', 'R2', "L'")
        self.myperms2['E3[UF>sR.dR>UR]'] = ('R', "U'", 'sR2', 'R', 'bR', "R'", "bR'", "sR'", "U'", "bR'", "R'", 'bR', 'R', 'U2', "sR'", "R'")
        self.myperms2['E3[UF>UR>sR.dR]'] = ('R', 'sR', "U2'", "R'", "bR'", 'R', 'bR', 'U', 'sR', 'bR', 'R', "bR'", "R'", "sR2'", 'U', "R'")
        self.myperms2['E3[UF>bR.sR>RU]'] = ("R2'", 'F', "sR2'", "R'", "dR'", 'R', 'dR', 'sR', 'F', 'dR', 'R', "dR'", "R'", "F2'", 'sR', 'R2')
        self.myperms2['E3[UF>RU>bR.sR]'] = ("R2'", "sR'", 'F2', 'R', 'dR', "R'", "dR'", "F'", "sR'", "dR'", "R'", 'dR', 'R', 'sR2', "F'", 'R2')
        self.myperms2['E3[L.sL>R.bR>FU]'] = ('L2', "R2'", "U'", "F'", 'U', 'F', 'R', 'L', 'F', 'U', "F'", "U'", 'L2', 'R')
        self.myperms2['E3[UF>dL.L>R.bR]'] = ('U2', "L2'", 'bR2', 'U', 'bL', "U'", "bL'", "bR'", "L'", "bL'", "U'", 'bL', 'U', "L2'", "bR'", "U2'")
        self.myperms2['E3[L.sL>FU>R.sR]'] = ("R2'", "L2'", 'U', 'F', "U'", "F'", "L'", "R'", "F'", "U'", 'F', 'U', "R2'", "L2'")
        self.myperms2['E3[D.sR>LU>RU]'] = ('R', 'sR2', 'U2', "R'", "bR'", 'R', 'bR', 'U', 'sR', 'bR', 'R', "bR'", "R'", 'sR2', 'U2', "R'")
        self.myperms2['E3[UF>R.bR>dL.sL]'] = ('F2', "R2'", "dL2'", 'F', 'dR', "F'", "dR'", "dL'", "R'", "dR'", "F'", 'dR', 'F', "R2'", "dL2'", "F2'")
        self.myperms2['E3[UF>dL.sL>R.bR]'] = ('F2', 'dL2', 'R2', "F'", "dR'", 'F', 'dR', 'R', 'dL', 'dR', 'F', "dR'", "F'", 'dL2', 'R2', "F2'")


        self.myperms2['E3[UL>bR.R>RU]'] = ("U'", "sR'", "R'", "U'", 'R', 'sR', 'U2', "R'", "U'", "F'", 'U', 'R', "sR'", 'bR', "R'", 'F', 'R', "bR'", 'sR')
        self.myperms2['E3[UL>R.bR>UR]'] = ("U'", "R'", 'F', 'U', 'R', "U'", "R'", "F'", "bR'", "R'", "U'", 'R', 'U', 'bR', 'R', "U'", 'bR2', 'U', "bR'", "U'", "bR'", 'U2', 'F', 'R', 'bR', "R'", "F'", "bR'")
        self.myperms2['E3[UL>RU>bR.R]'] = ("R'", 'U', "F'", "U2'", 'F', 'U', 'R', 'bR', "bL'", "bR'", "R'", 'bR', 'bL', "bR'", 'R', "F'", 'U', 'bL', "U'", 'F', 'U', "bL'", "U'")
        self.myperms2['E3[U.bL>R.bR>UF]'] = ("U2'", 'bR', "U2'", "bR'", "U'", "R2'", 'U', 'L', "U'", 'R2', 'U', "L'", "U'", "bR'", 'R', 'bR', "bL'", "bR'", "R'", 'bR', 'bL')
        self.myperms2['E3[U.bL>UF>bR.R]'] = ('bR', "U'", "bR'", "U'", "R'", "F'", 'U', 'F2', 'R', "bL'", "bR'", "R'", "F'", 'R', 'bR', "bL2'", "L2'", "bL'", 'L', 'bL', 'L', "bL2'", "bR'", "U'", "L'", 'U', 'L', 'bR')
        self.myperms2['E3[UF>RU>B.sR]'] = ("sR'", 'bR2', 'F', 'U', 'R', "U'", "R'", "F'", "bR'", "R'", "U'", 'R', 'U', "bR'", 'sR', 'F2', "R2'", "F'", 'R', 'F', 'R', "F2'", "L'", "U'", "R'", 'U', 'R', 'L')
        self.myperms2['E3[UL>RU>R.bR]'] = ('L', 'F', "U'", "R'", 'U', "F'", "L'", 'F', 'R', "F'", "bL'", "R'", 'bR', 'bL', "bR'", 'R', 'bR', "bL'", "bR'", 'bL')
        self.myperms2['E3[B.sL>LU>RU]'] = ('U', 'L', "F'", "U'", "L'", 'U', 'L', 'F', 'bL', 'L', 'U', "L'", "U'", "bL'", "L'", 'sL2', 'L', 'bL', "L'", "bL'", "U'", "sL'", "bL'", "L'", 'bL', 'L', "sL'")
        self.myperms2['E3[UL>sR.B>UR]'] = ("U'", "R'", 'F', 'U', 'R', "U'", "R'", "F'", "bR'", "R'", "U'", 'R', 'U', 'bR', 'R', "sR2'", "R'", "bR'", 'R', 'bR', 'U', 'sR', 'bR', 'R', "bR'", "R'", 'sR')
        self.myperms2['E3[UF>sL.dL>bR.R]'] = ('U2', "bL2'", "sL2'", "bR2'", 'bL', 'B', "bL'", "B'", "bR'", "sL'", "B'", "bL'", 'B', 'bL', "sL2'", "bR2'", 'bL2', "U2'")
        self.myperms2['E3[UF>R.bR>UR]'] = ("U2'", "sR'", "bR'", "R'", 'bR', 'R', 'sR', 'U2', 'F', 'bR', "F'", "R'", "U'", 'R', 'U', "bR'")
        self.myperms2['E3[RF>UR>LU]'] = ("U'", 'R', "bR'", "U'", "R'", 'U', 'R', 'bR', 'U', 'L', "U'", "R'", 'U', "L'", "R'", "F'", "L'", "bL'", 'L', 'F', 'R', "L'", 'bL', 'L')
        self.myperms2['E3[UF>LU>R.bR]'] = ('U2', "R'", "bL'", "U'", "bR'", 'U', 'bR', 'bL', 'R', 'bR', 'U', "bR'", 'U2', "F2'", 'L2', 'F', "L'", "F'", "L'", 'F2', 'R', 'U', 'L', "U'", "L'", "R'")






        self.myperms2['E2[UF>FU;UR>RU]'] = ("F2", "R2'", "F'", "R", "F", "R", "F2'",
                                 "L'", "U'", "R'", "U", "R", "L")

        self.myperms2['E2[UL>LU;UR>RU]'] = ("R'", "F'", 'L2', "F2'", "L'", 'F', 'L', 'F', "L2'", "bL'", "U'", "F'", 'U', 'F2', 'bL', 'R')
        self.myperms2['E2[UF>FU;bR.R>R.bR]'] = ("R'", "F2", "R2'", "F'", "R", "F", "R", "F2'", "L'", "U'", "R'", "U", "R2", "L")
        self.myperms2['E2[UL>LU;bR.R>R.bR]'] = ("U2'", 'bR2', "U2'", "bR'", 'U', 'bR', 'U', "bR2'", "sR'", "R'", "U'", 'R', "U2'", 'sR')
        self.myperms2['E2[UF>FU;bR.sR>sR.bR]'] = ("bR", "R'", "F2", "R2'", "F'", "R", "F", "R", "F2'", "L'", "U'", "R'", "U", "R2", "L", "bR'")
        self.myperms2['E2[UF>FU;bL.B>B.bL]'] = ("bL2'", 'L2', 'U2', "L2'", "U'", 'L', 'U', 'L', "U2'", "R'", "F'", "L'", 'F', "L'", 'R', 'bL2')
        self.myperms2['E2[B.sL>sL.B;UF>FU]'] = ('sL2', "L2'", 'U2', "L2'", "U'", 'L', 'U', 'L', "U2'", "R'", "F'", "L'", 'F', "L2'", 'R', "sL2'")
        self.myperms2['E2[DB>BD;UF>FU]'] = ("B", "sR2'", "R2'", "F2", "R2'", "F'", "R", "F", "R", "F2'", "L'", "U'", "R'", "U", "R2'", "L", "sR2", "B'")

        self.myperms2['E4[U.bL>bL.U;UF>FU;UL>LU;UR>RU]'] = ("L'", 'bL', 'L', "U2'", "bL'", "U2'", 'bL2', 'U', 'L', "U'", "L'", 'bL2', 'U', 'bL2', "U2'", "bL'")


        self.myperms2['E5[U.bL>UL>UF>UR>U.bR]'] = ("R2'","U'","R2","F2","R2'","F2'","U'","F2","R2","F2'","R2'","U2","R2","U'")
        self.myperms2['C5[U.L.bL>U.bL.bR>U.bR.R>URF>UFL]'] = ("R2'","U'","R2","F2","R2'","F2'","U'","F2","R2","F2'","R2'","U2","R2")
        self.myperms2['E5[U.bL>UF>U.bR>UL>UR]'] = ("R2","U2'","R2'","F2'","U'","F2","U'","R2","U'","R2'","F2'","U2'","F2","U2'")
        self.myperms2['C5[U.L.bL>U.bR.R>UFL>U.bL.bR>URF]'] = ("R2","U2'","R2'","F2'","U'","F2","U'","R2","U'","R2'","F2'","U2'","F2")







        
        self.myperms2 = normalize_myperm_registry(self.myperms2)

        self.myperms_group = {'Corner':set(),
                              'MidEdge':set()}


        

        for key in self.myperms2.keys():
            L = self.make_transformations(self.myperms2[key],tuple())
            for i in range(120):
                self.myperms[make_myperm_key(key, i)] = L[0][i]

        self.move_keys = [s + t for s in self.colors for t in ["+ ","++","- ","--"]]
        self.single_and_rotate = []

        for m in self.move_keys:
            single_move_key = make_myperm_key(single_move_myperm_name(m), 0)
            self.myperms[single_move_key] = (m,)
            self.single_and_rotate.append(single_move_key)



    
        self.my_scrambles = []


    

        self.surface_num = 10
        
        self.state = np.zeros(self.surface_num * 12,dtype = str)
        for i in range(12):
            self.state[i*self.surface_num:(i+1)*self.surface_num] = self.colors[i]

        
        self.state_0 = self.state.copy()






        

            
        self.move_len = len(self.move_keys)
        self.key_to_num = {}
        for i in range(self.move_len):
            self.key_to_num[self.move_keys[i]] = i

        self.my_scrambles2 = {0:{}}
        self.my_scramble_changed_piece_keys = {0:{}}
        for key in self.move_keys:
            self.my_scrambles2[0][key] = set([])
        self.my_scramble_changed_piece_keys[0] = {}
        self.counter = {1:{},2:{},3:{},4:{},5:{},6:{},7:{}}

        
        K = {"U":[(0,1,2,3,4),(5,6,7,8,9),(12,22,32,42,52),(13,23,33,43,53),(15,25,35,45,55)],
             "F":[(10,11,12,13,14),(15,16,17,18,19),(2,51,94,80,23),(3,52,90,81,24),(5,59,97,88,26)],
             "L":[(20,21,22,23,24),(25,26,27,28,29),(3,11,84,70,33),(4,12,80,71,34),(6,19,87,78,36)],
             "P":[(30,31,32,33,34),(35,36,37,38,39),(4,21,74,60,43),(0,22,70,61,44),(7,29,77,68,46)],
             "Q":[(40,41,42,43,44),(45,46,47,48,49),(0,31,64,100,53),(1,32,60,101,54),(8,39,67,108,56)],
             "R":[(50,51,52,53,54),(55,56,57,58,59),(1,41,104,90,13),(2,42,100,91,14),(9,49,107,98,16)],
             "B":[(60,61,62,63,64),(65,66,67,68,69),(112,101,44,30,73),(113,102,40,31,74),(115,109,47,38,76)],
             "X":[(70,71,72,73,74),(75,76,77,78,79),(113,61,34,20,83),(114,62,30,21,84),(116,69,37,28,86)],
             "G":[(80,81,82,83,84),(85,86,87,88,89),(114,71,24,10,93),(110,72,20,11,94),(117,79,27,18,96)],
             "H":[(90,91,92,93,94),(95,96,97,98,99),(110,81,14,50,103),(111,82,10,51,104),(118,89,17,58,106)],
             "Y":[(100,101,102,103,104),(105,106,107,108,109),(111,91,54,40,63),(112,92,50,41,64),(119,99,57,48,66)],
             "D":[(110,111,112,113,114),(115,116,117,118,119),(62,72,82,92,102),(63,73,83,93,103),(65,75,85,95,105)]}
             
        self.move = {}

        for k in K:
            A=np.arange(120,dtype = 'i')
            for T in K[k]:
                for i in range(5):
                    A[T[i]] = T[(i - 1) % 5]


            self.move[k + '+ '] = A.copy()
            self.move[k + '++'] = self.move[k + '+ '][A].copy()
            self.move[k + '- '] = np.argsort(self.move[k + '+ '])
            self.move[k + '--'] = np.argsort(self.move[k + '++'])

    
        




        self.edge_key = {'UF': 0,'FU': 1,'UL': 2,'LU': 3,
                         'UP': 4,'PU': 5,'UQ': 6,'QU': 7,
                         'UR': 8,'RU': 9,'FL':10,'LF':11,
                         'LP':12,'PL':13,'PQ':14,'QP':15,
                         'QR':16,'RQ':17,'RF':18,'FR':19,
                         'FG':20,'GF':21,'GL':22,'LG':23,
                         'LX':24,'XL':25,'XP':26,'PX':27,
                         'PB':28,'BP':29,'BQ':30,'QB':31,
                         'QY':32,'YQ':33,'YR':34,'RY':35,
                         'RH':36,'HR':37,'HF':38,'FH':39,
                         'BX':40,'XB':41,'XG':42,'GX':43,
                         'GH':44,'HG':45,'HY':46,'YH':47,
                         'YB':48,'BY':49,'DB':50,'BD':51,
                         'DX':52,'XD':53,'DG':54,'GD':55,
                         'DH':56,'HD':57,'DY':58,'YD':59}

        self.corner_key = {'UPQ': 0,'PQU': 1,'QUP': 2,
                           'UQR': 3,'QRU': 4,'RUQ': 5,
                           'URF': 6,'RFU': 7,'FUR': 8,
                           'UFL': 9,'FLU':10,'LUF':11,
                           'ULP':12,'LPU':13,'PUL':13,
                           'FHG':15,'HGF':16,'GFH':17,
                           'GLF':18,'LFG':19,'FGL':20,
                           'LGX':21,'GXL':22,'XLG':23,
                           'XPL':24,'PLX':25,'LXP':26,
                           'PXB':27,'XBP':28,'BPX':29,
                           'BQP':30,'QPB':31,'PBQ':32,
                           'QBY':33,'BYQ':34,'YQB':35,
                           'YRQ':36,'RQY':37,'QYR':38,
                           'RYH':39,'YHR':40,'HRY':41,
                           'HFR':42,'FRH':43,'RHF':44,
                           'DGH':45,'GHD':46,'HDG':47,
                           'DHY':48,'HYD':49,'YDH':50,
                           'DYB':51,'YBD':52,'BDY':53,
                           'DBX':54,'BXD':55,'XDB':56,
                           'DXG':57,'XGD':58,'GDX':59}


        self.edge_index = [(5,15),(6,25),(7,35),(8,45),(9,55),
                           (19,26),(29,36),(39,46),(49,56),(59,16),
                           (18,88),(87,27),(28,78),(77,37),(38,68),
                           (67,47),(48,108),(107,57),(58,98),(97,17),
                           (69,76),(79,86),(89,96),(99,106),(109,66),
                           (115,65),(116,75),(117,85),(118,95),(119,105)]
        self.corner_index = [(0,32,43),(1,42,53),(2,52,13),(3,12,23),(4,22,33),
                             (10,94,81),(80,24,11),(20,84,71),(70,34,21),(30,74,61),
                             (60,44,31),(40,64,101),(100,54,41),(50,104,91),(90,14,51),
                             (110,82,93),(111,92,103),(112,102,63),(113,62,73),(114,72,83)]

        self.center_index = []
        self._reindex_myperms_by_points()
        rename_myperms_by_effect(self)
        self.edge_colors = sorted(self.edge_key.keys(), key = lambda x: self.edge_key[x])
        self.corner_colors = sorted(self.corner_key.keys(), key = lambda x: self.corner_key[x])

        self.ips = 3000
        
        self.perfect_data = self.makedata()
                         

        X = np.zeros((1,self.ips),dtype = 'f')
        perfect_x = self.makedata()
        self.group_val = {}
        self.total_val = {}

        Y = X.copy()
        Y[0,:1200] = perfect_x[:1200]
        self.group_val['Corner'] = Y
        Y = X.copy()
        Y[0,1200:] = perfect_x[1200:]
        self.group_val['MidEdge'] = Y


        for key in ['Corner','MidEdge']:
            self.total_val[key] = np.sum(self.group_val[key])
        
        
        self.myperms_dict = {}
        self.piece_color_counter = {}

        self.default_color = {}

        for x in self.edge_index:
            self.default_color[x] = self.state_0[x[0]] + self.state_0[x[1]]
        
        for x in self.corner_index:
            self.default_color[x] = self.state_0[x[0]] + self.state_0[x[1]] + self.state_0[x[2]]

    
        self.num_to_piece = {}
    
        for i in range(120):
            if i % 10 < 5:
                self.num_to_piece[i] = [x for x in self.corner_index if i in x][0]
            else:
                self.num_to_piece[i] = [x for x in self.edge_index if i in x][0]


        for x in self.edge_index:
            for c in self.edge_key:
                if self.default_color[x] != c:
                    self.myperms_dict[(x,c)] = []
                    self.piece_color_counter[(x,c)] = 0


        for x in self.corner_index:
            for c in self.corner_key:
                if self.default_color[x] != c:
                    self.myperms_dict[(x,c)] = []
                    self.piece_color_counter[(x,c)] = 0


        
        
        for key in self.myperms.keys():
            X = self.myperms[key]
            for m in self.invert_moves(X):
                self.make_move(m)

            for x in self.edge_index:
                c = self.state[x[0]] + self.state[x[1]]
                if c != self.default_color[x]:
                    self.myperms_dict[(x,c)].append(key)

            for x in self.corner_index:
                c = self.state[x[0]] + self.state[x[1]] + self.state[x[2]]
                if c != self.default_color[x]:
                    self.myperms_dict[(x,c)].append(key)      

            for m in X:
                self.make_move(m)


        self.myperms_order = {}
        
        self.myperms_order['Corner'] = [i for i in range(119,-1,-1) if i % 10 < 5]
        self.myperms_order['MidEdge'] = [i for i in range(119,-1,-1) if i % 10 >= 5]
        self.scramble_selector = ScrambleSelector(self)


    def _reindex_myperms_by_points(self):
        """point最大の対称変換を各myperm系列の#00へ割り当てる。"""
        points_path = Path(__file__).resolve().parent.parent / "Points.txt"
        if not points_path.exists():
            self.myperm_transform_key_aliases = {}
            self.myperm_transform_points = {}
            return
        point_table = load_myperm_points(points_path, puzzle = "megaminx")
        reindex_myperms_by_points(self, point_table)


    def collect_single_moves_and_rotate(self):
        return self.single_and_rotate

    def collect_single_move_and_rotate(self):
        return self.collect_single_moves_and_rotate()
        
    

    def format_move(self, move):
        """Convert an internal Megaminx move token to the clearer display notation."""
        internal_move = self.normalize_move_key(move)
        return DISPLAY_FACE_NAMES[internal_move[0]] + DISPLAY_MOVE_SUFFIXES[internal_move[1:]]

    def format_moves(self, moves):
        """Convert an iterable of internal moves to display notation."""
        return tuple(self.format_move(move) for move in self.normalize_move_sequence(moves))

    def normalize_move_key(self, move):
        """Accept either legacy internal tokens or display notation and return an internal move key."""
        if isinstance(move, (tuple, list)):
            if len(move) == 1 and isinstance(move[0], str):
                move = move[0]
            else:
                raise KeyError(move)
        if move in self.move:
            return move
        if len(move) == 3 and move[0] in self.colors and move[1:] in self.inverse:
            return move

        stripped_move = move.strip()
        for face_token in DISPLAY_FACE_TOKENS:
            if stripped_move.startswith(face_token):
                suffix = stripped_move[len(face_token):]
                if suffix in DISPLAY_TO_INTERNAL_SUFFIX:
                    return DISPLAY_TO_INTERNAL_FACE[face_token] + DISPLAY_TO_INTERNAL_SUFFIX[suffix]
        raise KeyError(move)

    def normalize_move_sequence(self, moves):
        """Return a move sequence in Megaminx internal canonical notation."""
        return tuple(self.normalize_move_key(move) for move in moves)

    def create_new_set(self):
        i = len(self.my_scrambles2.keys())
        self.my_scrambles2[i] = {}
        self.my_scramble_changed_piece_keys[i] = {}
        for k in self.my_scrambles2[0].keys():
            self.my_scrambles2[i][k] = set([]) 

    def register_scramble_sequence(self, level, moves):
        """Register one scramble sequence with canonical move notation."""
        normalized_moves = self.normalize_move_sequence(moves)
        self.my_scrambles2[level][normalized_moves[-1]].add(normalized_moves)
        self.my_scramble_changed_piece_keys[level][normalized_moves] = tuple(
            self.get_chenged_pieces_keys_from_moves(normalized_moves)
        )

    def get_registered_scramble_changed_piece_keys(self, level, moves):
        """Return cached changed-piece keys for a registered scramble sequence."""
        normalized_moves = self.normalize_move_sequence(moves)
        return self.my_scramble_changed_piece_keys[level].get(normalized_moves)

    def make_move(self,key):
        internal_key = self.normalize_move_key(key)
        self.state = self.state[self.move[internal_key]]


    def scramble(self,N,Move = None,difficult_mode = False,scramble_mode = None,flip = None,rotate = None,swap = False,add_moves = None,transform_N = None,flip_inside = None,move_count_policy = 'prefer_rare'):
        if Move != None:
            normalized_moves = self.normalize_move_sequence(Move)
            for m in normalized_moves:
                self.make_move(m)

            return normalized_moves
        else:
            move_lis = []
            move_count = self._init_scramble_count()
            transform_idx = 0 if transform_N is None else transform_N
            move_count_policy = self.scramble_selector.resolve_move_count_policy(move_count_policy, add_moves)

            for level_index in range(N):
                selected_moves = self._guided_scramble_moves(level_index, move_count, move_count_policy)
                transformed_moves = self.transform(selected_moves, transform_idx, flip_inside = False, invert = False)
                move_lis += list(transformed_moves)
                for move in transformed_moves:
                    self.make_move(move)

            return self.normalize_move_sequence(move_lis)
    



    def _init_scramble_count(self):
        return self.scramble_selector.init_move_count()

    def _guided_scramble_moves(self, level_index, move_count, move_count_policy):
        return self.scramble_selector.select(level_index, move_count, move_count_policy = move_count_policy)

    def _collect_scramble_candidates(self, level_index):
        level_index = self.scramble_selector.resolve_level(level_index)
        return self.scramble_selector.collect_candidates(level_index)

    def _select_scramble_candidate(self, candidates, move_count, move_count_policy, level_index):
        if move_count_policy == 'prefer_frequent':
            return self.scramble_selector._select_candidate_max(candidates, move_count, level_index)
        return self.scramble_selector._select_candidate_min(candidates, move_count, level_index)

    def _select_candidate_max(self, candidates, move_count, level_index):
        return self.scramble_selector._select_candidate_max(candidates, move_count, level_index)

    def _select_candidate_min(self, candidates, move_count, level_index):
        return self.scramble_selector._select_candidate_min(candidates, move_count, level_index)

    def _evaluate_piece_color_value(self, changed_piece_keys):
        value = 0
        for key in changed_piece_keys:
            value += self.piece_color_counter[key]
        return value

    def _update_piece_color_counter(self, changed_piece_keys):
        self.scramble_selector.update_piece_color_counter(changed_piece_keys)

    def _update_count(self, move_count, moves):
        self.scramble_selector.update_count(move_count, moves)

    def _update_counter_stats(self, level_index, moves):
        self.scramble_selector.update_counter_stats(level_index, moves)

    def get_chenged_pieces_keys_from_moves(self, moves):
        current_state = self.state.copy()
        self.reset()
        for move in self.normalize_move_sequence(moves):
            self.make_move(move)
        changed_piece_keys = self._get_changed_pieces_keys()
        self.state = current_state
        return changed_piece_keys

    def _get_changed_pieces_keys(self):
        changed_piece_keys = self._register_changed_edge_keys()
        changed_piece_keys += self._register_changed_corner_keys()
        return changed_piece_keys

    def _register_changed_edge_keys(self):
        changed_piece_keys = []
        for piece in self.edge_index:
            color = self.state[piece[0]] + self.state[piece[1]]
            if color != self.default_color[piece]:
                changed_piece_keys.append((piece, color))
        return changed_piece_keys

    def _register_changed_corner_keys(self):
        changed_piece_keys = []
        for piece in self.corner_index:
            color = self.state[piece[0]] + self.state[piece[1]] + self.state[piece[2]]
            if color != self.default_color[piece]:
                changed_piece_keys.append((piece, color))
        return changed_piece_keys

    def flip_moves(self,Moves):
        normalized_moves = self.normalize_move_sequence(Moves)
        return tuple([self.flip['UD'][x[0]] + self.inverse[x[1:]] for x in normalized_moves])

    def rotate_moves(self,Moves,axis = None):
        normalized_moves = self.normalize_move_sequence(Moves)
        if axis in self.rotate:
            return tuple([self.rotate[axis][x[0]] + x[1:] for x in normalized_moves])
        else:
            return tuple(normalized_moves)


    def invert_str(self,s):
        internal_key = self.normalize_move_key(s)
        return internal_key[0] + self.inverse[internal_key[1:]]

    def invert_moves(self,Moves):
        normalized_moves = self.normalize_move_sequence(Moves)
        return tuple([self.invert_str(x) for x in normalized_moves[::-1]])



    def reduce(self,move_lis):
        L = []
        states = [''.join(self.state)]
        Indices = []
        idx = 0
        normalized_moves = self.normalize_move_sequence(move_lis)
        for m in normalized_moves:
            self.make_move(m)
            s = ''.join(self.state)
            if s in states:
                i = states.index(s)
                L = L[:i]
                states = states[:i+1]
                Indices = Indices[:i]

            else:
                L.append(m)
                states.append(''.join(self.state))
                Indices.append(idx)

            idx += 1

        
        for m in self.invert_moves(normalized_moves):
            self.make_move(m)

        
                


        return (tuple(L),Indices) 

    def simplify(self,move_lis):
        normalized_moves = self.normalize_move_sequence(move_lis)
        L = ()
        for m in normalized_moves:
            if len(L) > 0 and L[-1][0] == m[0]:
                k = self.mult[L[-1][1:],m[1:]]
                L = L[:-1]
                if k != 0:
                    L += (m[0] + k,)
            else:
                L += (m,)

        return tuple(L)

    def conjugate(self,A,B):
        return self.simplify(A + B + self.invert_moves(A))

    def commutator(self,A,B):
        return self.simplify(A + B + self.invert_moves(A) + self.invert_moves(B))
        
    def reset(self):
        self.state[:] = self.state_0

    def makedata(self):
        x = np.zeros(self.ips,dtype = 'f')
        idx = 0
        for p in self.corner_index:
            s = self.state[p[0]] + self.state[p[1]] + self.state[p[2]]
            x[60*idx + self.corner_key[s]] = 1
            idx += 1
        
        for p in self.edge_index:
            s = self.state[p[0]] + self.state[p[1]]
            x[60*idx + self.edge_key[s]] = 1
            idx += 1

        return x

    def is_perfect(self):
        return (self.state == self.state_0).all()


    def transform(self,s,i,flip_inside = False,invert = False):
        key = self.transformation_keys[i]
        if invert:
            key = tuple([self.tf_invert[k] for k in key[::-1]])

        New_s = self.normalize_move_sequence(s)
        for x in key:
            if x == 'F':
                New_s = self.flip_moves(New_s)
            else:
                New_s = self.rotate_moves(New_s,axis = self.transformation_dict[x])


        return New_s

    def make_transformations(self,s,Moves):
        scramble_list = []
        move_list = []
        for transform_index in range(len(self.transformation_keys)):
            transformed_scramble = self.transform(s, transform_index, invert = True)
            transformed_moves = self.transform(Moves, transform_index, invert = True)
            scramble_list.append(self.normalize_move_sequence(transformed_scramble))
            move_list.append(self.normalize_move_sequence(transformed_moves))

        return scramble_list, move_list
        
                


    def _group_name_map(self):
        """Return the long-form group names used by solve/session helpers."""
        return {'A': 'Corner', 'B': 'MidEdge'}

    def state_to_str(self):
        return reduce(lambda x,y : x+y , self.state)

    def piece_display_name(self, piece_type, piece):
        """Return a compact position-style label for a solved-state piece."""
        labels = '-'.join(DISPLAY_FACE_NAMES[self.state_0[index]] for index in piece)
        return f'{piece_type}-{labels}'

    def set_state(self,S):
        if len(S) == 12 * self.surface_num:
            self.state[:] = np.array(list(S))




Rubiks_3 = MegaminxCube
