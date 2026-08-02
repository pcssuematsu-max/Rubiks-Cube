import unittest
from types import SimpleNamespace
import gc

from core.myperm_effects import EffectComponent, MypermEffectAnalyzer, PieceTransfer
from core.myperm_keys import make_myperm_key, resolve_myperm_key
from cto.cube import CtoCube
from cube.rubiks_cube import Rubiks_3
from fto.cube import FtoCube
from megaminx.cube import MegaminxCube
from pyraminx.cube import MasterPyraminxCube, PyraminxCube
from skewb.cube import SkewbCube
from managers.solve_session import SolveSessionManager, SolveSessionState


class MypermEffectAnalyzerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cube = Rubiks_3(size = 3)
        cls.analyzer = MypermEffectAnalyzer(cls.cube)
        cls.cube7 = Rubiks_3(size = 7)
        cls.analyzer7 = MypermEffectAnalyzer(cls.cube7)

    def test_corner_permutation_includes_count_direction_and_positions(self):
        name = self.analyzer.proposed_name(make_myperm_key("C3[UBR>UFL>URF]", 0))
        self.assertEqual(name, "C3[UBR>UFL>URF]")

    def test_registered_key_uses_canonical_source_name_without_legacy_alias(self):
        current_key = make_myperm_key("C3[UBR>UFL>URF]", 0)
        legacy_key = make_myperm_key("CornerPermutation-A00", 0)
        self.assertIn(current_key, self.cube.myperms)
        self.assertNotIn(legacy_key, self.cube.myperms)
        self.assertEqual(resolve_myperm_key(self.cube, current_key), current_key)
        self.assertIsNone(resolve_myperm_key(self.cube, legacy_key))

    def test_edge_flip_and_corner_twist_include_orientation(self):
        edge_name = self.analyzer.proposed_name(make_myperm_key("E2[FL>LF;RF>FR]~v02", 0))
        corner_name = self.analyzer.proposed_name(make_myperm_key("C2[UFL>FLU;URF>FUR]", 0))
        self.assertEqual(edge_name, "E2[FL>LF;RF>FR]")
        self.assertEqual(corner_name, "C2[UFL>FLU;URF>FUR]")

    def test_mixed_corner_cycle_and_twists_are_not_collapsed_to_short_cycle(self):
        component = EffectComponent(
            group = "Corner",
            part_code = "C",
            piece_size = 3,
            transfers = (
                PieceTransfer("Corner", "C", 0, 1, "U.bL.bR", "URF", "URF"),
                PieceTransfer("Corner", "C", 1, 2, "URF", "sR.bR.B", "sR.bR.B"),
                PieceTransfer("Corner", "C", 2, 0, "sR.bR.B", "U.bL.bR", "U.bL.bR"),
                PieceTransfer("Corner", "C", 3, 3, "UFL", "UFL", "FLU", "twist"),
                PieceTransfer("Corner", "C", 4, 4, "R.U.bR", "R.U.bR", "U.bR.R", "twist"),
            ),
            cycles = (("U.bL.bR", "URF", "sR.bR.B"),),
        )

        name = component.concise_name()

        self.assertEqual(
            name,
            "C5[R.U.bR>U.bR.R;U.bL.bR>URF;UFL>FLU;URF>sR.bR.B;sR.bR.B>U.bL.bR]",
        )

    def test_transformed_key_uses_transformed_positions(self):
        original = self.analyzer.proposed_name(make_myperm_key("C3[UBR>UFL>URF]", 0))
        transformed = self.analyzer.proposed_name(make_myperm_key("C3[UBR>UFL>URF]", 1))
        self.assertEqual(transformed, "C3[DBL>DLF>DRB]")
        self.assertNotEqual(original, transformed)

    def test_large_effect_uses_compact_counts(self):
        effect = self.analyzer7.analyze(make_myperm_key("EAll12[XY>YX]", 0))
        self.assertEqual(effect.concise_name(), "EAll12[XY>YX]")
        self.assertTrue(any(component.part_code.startswith("Ctr") for component in effect.components))

    def test_unavailable_inner_layer_moves_are_skipped_by_size(self):
        cube5 = Rubiks_3(size = 5, RegisterMyperms = False)
        self.assertEqual(
            cube5._moves_available_for_size(('2U2', '3U2', ' R2', '3R2')),
            ('2U2', ' R2'),
        )

        cube3 = Rubiks_3(size = 3, RegisterMyperms = False)
        self.assertEqual(
            cube3._moves_available_for_size(
                ("2F2", "3F2", " R2", " U2", "2F2", "3F2", " U2", " R2", "2F2", "3F2")
            ),
            (),
        )

    def test_outer_center_bar_uses_bar_notation(self):
        expected_effect_names = (
            ("OuterCenterBar-A", "CtrBar3[F@2D>F@2U>R@2F]"),
            ("OuterCenterBar-ZZ", "CtrBar4s[L@2D<>R@2B;L@2U<>R@2F]"),
        )
        for source_name, effect_name in expected_effect_names:
            source_key = make_myperm_key(source_name, 0)
            self.assertIn(source_key, self.cube7.myperms)
            self.assertEqual(resolve_myperm_key(self.cube7, source_name), source_key)
            self.assertEqual(
                self.analyzer7.proposed_name(source_key),
                effect_name,
            )
            self.assertIsNone(
                resolve_myperm_key(self.cube7, effect_name),
            )

    def test_mid_center_bar_uses_mid_bar_notation(self):
        expected_effect_names = (
            ("MidCenterBar(VV)", "CtrMidBar6p[3x2][F@L>R@F>R@D;F@R>R@B>R@U]"),
            ("MidCenterBar-Adjacent3Center-OB", "CtrMidBar6p[3x2][F@L>U@F>R@F;F@R>U@B>R@B]"),
        )
        for source_name, effect_name in expected_effect_names:
            source_key = make_myperm_key(source_name, 0)
            self.assertIn(source_key, self.cube7.myperms)
            self.assertEqual(resolve_myperm_key(self.cube7, source_name), source_key)
            self.assertEqual(
                self.analyzer7.proposed_name(source_key),
                effect_name,
            )
            self.assertIsNone(resolve_myperm_key(self.cube7, effect_name))

    def test_center_midedge_and_corner_swap_source_names_are_effect_names(self):
        midedge_key = make_myperm_key("CtrCore4[B>L>F>R]+ME2[FL>FR]", 0)
        corner_key = make_myperm_key("C2[DFR>FUR]+CtrCore4[B>R>F>L]~v01", 0)

        self.assertIn(midedge_key, self.cube7.myperms)
        self.assertIn(corner_key, self.cube7.myperms)
        self.assertEqual(resolve_myperm_key(self.cube7, "CtrCore4[B>L>F>R]+ME2[FL>FR]"), midedge_key)
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'CtrCore4[B>R>F>L]+ME2s[RF<>UL]'),
            make_myperm_key("CtrCore4[B>R>F>L]+ME2s[RF<>UL]", 0),
        )
        self.assertEqual(resolve_myperm_key(self.cube7, "C2[DFR>FUR]+CtrCore4[B>R>F>L]~v01"), corner_key)
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2s[DBL<>URF]+CtrCore4[B>L>F>R]~v01'),
            make_myperm_key("C2s[DBL<>URF]+CtrCore4[B>L>F>R]", 0),
        )

    def test_commutator_source_names_resolve_by_size_specific_effects(self):
        self.assertEqual(
            resolve_myperm_key(self.cube, 'C4[DLF>FLU;UBR>RFU;UFL>LFD;URF>UBR]+E3[FL>FU>RU]'),
            make_myperm_key("C4[DLF>FLU;UBR>RFU;UFL>LFD;URF>UBR]+E3[FL>FU>RU]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C4[DLF>FLU;UBR>RFU;UFL>LFD;URF>UBR]+EAll3[FL>FU>RU]'),
            make_myperm_key("C4[DLF>FLU;UBR>RFU;UFL>LFD;URF>UBR]+EAll3[FL>FU>RU]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v01'),
            make_myperm_key("CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'CtrPlus12p[3x4]+ME5[BR>UF>FD>FL>RF]'),
            make_myperm_key("CtrPlus12p[3x4]+ME5[BR>UF>FD>FL>RF]", 0),
        )

    def test_midedge4_source_names_are_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'ME4[DF>UF;FL>FR]'),
            make_myperm_key("ME4[DF>UF;FL>FR]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'ME4[BR>RF;FL>LF;LB>RB;RF>LB]'),
            make_myperm_key("ME4[BR>RF;FL>LF;LB>RB;RF>LB]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'ME4[DF>UF;FL>LF;RF>DF;UF>FR]'),
            make_myperm_key("ME4[DF>UF;FL>LF;RF>DF;UF>FR]", 0),
        )

    def test_midedge_flip_source_names_are_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'ME2[FL>LF;RF>FR]~v01'),
            make_myperm_key("ME2[FL>LF;RF>FR]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'ME2[FL>LF;RF>FR]~v02'),
            make_myperm_key("ME2[FL>LF;RF>FR]~v02", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'ME4[DF>FD;FL>LF;RF>FR;UF>FU]~v01'),
            make_myperm_key("ME4[DF>FD;FL>LF;RF>FR;UF>FU]~v01", 0),
        )

    def test_wing3_source_names_are_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-3[FL@U>LB@U>RF@U]~v01'),
            make_myperm_key("W2-3[FL@U>LB@U>RF@U]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-3[FL@U>RF@U>LB@U]~v01'),
            make_myperm_key("W2-3[FL@U>RF@U>LB@U]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-3[DF@L>RF@U>FL@U]'),
            make_myperm_key("W2-3[DF@L>RF@U>FL@U]", 0),
        )

    def test_wing_swap_parallel_source_names_are_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-2s[FL@U<>RF@U]~v01'),
            make_myperm_key("W2-2s[FL@U<>RF@U]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-2s[RF@D<>RF@U]~v02'),
            make_myperm_key("W2-2s[RF@D<>RF@U]~v02", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-2s[FL@D<>RF@U]~v25'),
            make_myperm_key("W2-2s[FL@D<>RF@U]~v25", 0),
        )

    def test_wing_swap_skew_source_names_are_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-2s[RF@U<>UF@R]'),
            make_myperm_key("W2-2s[RF@U<>UF@R]", 0),
        )

        centers_cube = Rubiks_3(size = 4, Centers = True)
        self.assertEqual(
            resolve_myperm_key(centers_cube, 'CtrX8p[4x2]+W2-2s[FL@U<>UF@L]'),
            make_myperm_key("CtrX8p[4x2]+W2-2s[FL@U<>UF@L]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(centers_cube, 'CtrX8p[4x2]+W2-2s[DF@R<>FL@D]'),
            make_myperm_key("CtrX8p[4x2]+W2-2s[DF@R<>FL@D]", 0),
        )

    def test_l2_source_names_are_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-4[FL@D>RF@D>FL@U>RF@U]~v01'),
            make_myperm_key("W2-4[FL@D>RF@D>FL@U>RF@U]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-4[RF@D>UF@R>UF@L>RF@U]'),
            make_myperm_key("W2-4[RF@D>UF@R>UF@L>RF@U]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-4s[FL@D<>RF@U;FL@U<>RF@D]~v01'),
            make_myperm_key("W2-4s[FL@D<>RF@U;FL@U<>RF@D]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-4s[FL@D<>FL@U;UB@L<>UB@R]'),
            make_myperm_key("W2-4s[FL@D<>FL@U;UB@L<>UB@R]", 0),
        )

    def test_wing_parallel6_source_names_are_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-6p[3x2][BR@D>FL@D>RF@U;BR@U>FL@U>RF@D]'),
            make_myperm_key("W2-6p[3x2][BR@D>FL@D>RF@U;BR@U>FL@U>RF@D]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-6p[3x2][BR@D>FL@D>RF@D;BR@U>FL@U>RF@U]'),
            make_myperm_key("W2-6p[3x2][BR@D>FL@D>RF@D;BR@U>FL@U>RF@U]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-6[BR@D>FL@U>RF@D>BR@U>FL@D>RF@U]'),
            make_myperm_key("W2-6[BR@D>FL@U>RF@D>BR@U>FL@D>RF@U]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-6[BR@D>FL@U>RF@U>BR@U>FL@D>RF@D]'),
            make_myperm_key("W2-6[BR@D>FL@U>RF@U>BR@U>FL@D>RF@D]", 0),
        )

    def test_edge6p_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-4s[FL@D<>UF@R;FL@U<>UF@L]'),
            make_myperm_key("W2-4s[FL@D<>UF@R;FL@U<>UF@L]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-4s[FL@D<>UB@L;FL@U<>UB@R]~v02'),
            make_myperm_key("W2-4s[FL@D<>UB@L;FL@U<>UB@R]~v02", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-4[FL@D>UF@R>FL@U>UF@L]'),
            make_myperm_key("W2-4[FL@D>UF@R>FL@U>UF@L]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-4[FL@D>UB@R>FL@U>UB@L]'),
            make_myperm_key("W2-4[FL@D>UB@R>FL@U>UB@L]", 0),
        )

    def test_edge_flip_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'EAll2[FL>LF;RF>FR]'),
            make_myperm_key("EAll2[FL>LF;RF>FR]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'EAll2[LB>BL;RF>FR]'),
            make_myperm_key("EAll2[LB>BL;RF>FR]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'EAll4[BR>RB;FL>LF;LB>BL;RF>FR]'),
            make_myperm_key("EAll4[BR>RB;FL>LF;LB>BL;RF>FR]", 0),
        )

    def test_edge_block3cycle_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'EAll3[FL>UR>FR]'),
            make_myperm_key("EAll3[FL>UR>FR]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'EAll3[BR>LF>FR]'),
            make_myperm_key("EAll3[BR>LF>FR]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'EAll3[LB>UF>RF]'),
            make_myperm_key("EAll3[LB>UF>RF]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'EAll3[RF>FU>UR]'),
            make_myperm_key("EAll3[RF>FU>UR]", 0),
        )

    def test_edge_corner_swap_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2[DFR>RFU]+EAll2[FL>FR]'),
            make_myperm_key("C2[DFR>RFU]+EAll2[FL>FR]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2s[UFL<>URF]+EAll2s[RF<>UF]'),
            make_myperm_key("C2s[UFL<>URF]+EAll2s[RF<>UF]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2s[DFR<>URF]+EAll2s[LB<>RF]'),
            make_myperm_key("C2s[DFR<>URF]+EAll2s[LB<>RF]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2s[DLF<>UFL]+EAll2[BR>LF]'),
            make_myperm_key("C2s[DLF<>UFL]+EAll2[BR>LF]", 0),
        )

    def test_corner_edge_block_swap_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2s[DFR<>UFL]+EAll2[FL>FR]'),
            make_myperm_key("C2s[DFR<>UFL]+EAll2[FL>FR]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2s[DFR<>URF]+EAll2[FL>FR]'),
            make_myperm_key("C2s[DFR<>URF]+EAll2[FL>FR]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2s[DLF<>UFL]+EAll2[BR>LF]'),
            make_myperm_key("C2s[DLF<>UFL]+EAll2[BR>LF]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2[DBL>RUB]+EAll2[FL>FR]'),
            make_myperm_key("C2[DBL>RUB]+EAll2[FL>FR]", 0),
        )

    def test_parity_cycle_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C4[DFR>FUR>LFD>LUF]+ME2[FL>FR]'),
            make_myperm_key("C4[DFR>FUR>LFD>LUF]+ME2[FL>FR]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C4[DBL>FUR>DRB>FLU]+ME2[FL>FR]'),
            make_myperm_key("C4[DBL>FUR>DRB>FLU]+ME2[FL>FR]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C4[DBL>FLU>DLF>FUR]+ME2[FL>FR]'),
            make_myperm_key("C4[DBL>FLU>DLF>FUR]+ME2[FL>FR]", 0),
        )

    def test_parity_swap_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2[DFR>RFU]+ME2[FL>FR]~v01'),
            make_myperm_key("C2[DFR>RFU]+ME2[FL>FR]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2s[ULB<>URF]+ME2[FL>FR]~v01'),
            make_myperm_key("C2s[ULB<>URF]+ME2[FL>FR]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2[UFL>FUR]+ME2s[RF<>UF]'),
            make_myperm_key("C2[UFL>FUR]+ME2s[RF<>UF]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2[DLF>FLU]+ME2[BR>LF]~v01'),
            make_myperm_key("C2[DLF>FLU]+ME2[BR>LF]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2s[DLF<>UFL]+ME2[FL>RU]'),
            make_myperm_key("C2s[DLF<>UFL]+ME2[FL>RU]", 0),
        )

    def test_super_parity_swap_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2[DRB>LUF]+ME2s[FL<>UL]'),
            make_myperm_key("C2[DRB>LUF]+ME2s[FL<>UL]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2[DBL>RFU]+ME2[FL>LU]'),
            make_myperm_key("C2[DBL>RFU]+ME2[FL>LU]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C2[DBL>FUR]+ME2s[FL<>UB]'),
            make_myperm_key("C2[DBL>FUR]+ME2s[FL<>UB]", 0),
        )

    def test_super_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'EAll12[XY>YX]'),
            make_myperm_key("EAll12[XY>YX]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'EAll12s'),
            make_myperm_key("EAll12s", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C8[3x2]+EAll12[3x4]'),
            make_myperm_key("C8[3x2]+EAll12[3x4]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(
                self.cube7,
                "C6[DBL>FLU>RDF;DRB>BUL>RFU]+EAll6[DB>LU>FR;DR>LB>FU]",
            ),
            make_myperm_key(
                "C6[DBL>FLU>RDF;DRB>BUL>RFU]+EAll6[DB>LU>FR;DR>LB>FU]",
                0,
            ),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C8s~v01'),
            make_myperm_key("C8s~v01", 0),
        )

    def test_bigqr_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(
                self.cube7,
                "CtrCore4[B>R>F>L]+CtrObl32p[4x8]+CtrPlus32p[4x8]+CtrX32p[4x8]+EAll2[FL>FR]",
            ),
            make_myperm_key(
                "CtrCore4[B>R>F>L]+CtrObl32p[4x8]+CtrPlus32p[4x8]+CtrX32p[4x8]+EAll2[FL>FR]",
                0,
            ),
        )
        self.assertEqual(
            resolve_myperm_key(
                self.cube7,
                "C2s[DBL<>URF]+CtrCore4[B>R>F>L]+CtrObl32p[16x2]+CtrPlus32p[16x2]+CtrX32p[16x2]",
            ),
            make_myperm_key(
                "C2s[DBL<>URF]+CtrCore4[B>R>F>L]+CtrObl32p[16x2]+CtrPlus32p[16x2]+CtrX32p[16x2]",
                0,
            ),
        )

    def test_side_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-4[BR@U>RF@U>FL@D>FL@U]'),
            make_myperm_key("W2-4[BR@U>RF@U>FL@D>FL@U]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-4s[BR@U<>FL@U;LB@U<>RF@D]'),
            make_myperm_key("W2-4s[BR@U<>FL@U;LB@U<>RF@D]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-4s[FL@D<>LB@U;FL@U<>RF@U]'),
            make_myperm_key("W2-4s[FL@D<>LB@U;FL@U<>RF@U]", 0),
        )

    def test_xperm_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C4s[UBR<>URF;UFL<>ULB]'),
            make_myperm_key("C4s[UBR<>URF;UFL<>ULB]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C4[UBR>RFU;UFL>LBU]'),
            make_myperm_key("C4[UBR>RFU;UFL>LBU]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'C4s[DBL<>DRB;UFL<>URF]'),
            make_myperm_key("C4s[DBL<>DRB;UFL<>URF]", 0),
        )

    def test_edgepk_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-3[FL@D>RF@U>FL@U]~v01'),
            make_myperm_key("W2-3[FL@D>RF@U>FL@U]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-3[FL@D>FL@U>RF@U]~v01'),
            make_myperm_key("W2-3[FL@D>FL@U>RF@U]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-3[FL@U>RF@U>RF@D]'),
            make_myperm_key("W2-3[FL@U>RF@U>RF@D]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-3[BR@U>FL@U>FL@D]~v01'),
            make_myperm_key("W2-3[BR@U>FL@U>FL@D]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, 'W2-3[BR@U>FL@D>FL@U]~v02'),
            make_myperm_key("W2-3[BR@U>FL@D>FL@U]~v02", 0),
        )

    def test_partial_wing_effect_is_not_collapsed_into_full_edge_bundle(self):
        name = self.analyzer7.proposed_name(make_myperm_key("W2-2s[RF@U<>UF@R]", 0))
        self.assertTrue(name.startswith("W2-2s["))
        self.assertNotIn("EAll", name)

    def test_every_wing_axis_label_matches_its_inner_layer(self):
        wing_names = []
        for piece in self.cube7.edge_index:
            if self.cube7._edge_axis_label(piece) in {"M", "E", "S"}:
                continue
            name = self.analyzer7._rubiks_position_name("Edge", piece)
            layer = name.split("@", 1)[1]
            self.assertTrue(
                any(self.cube7.move[layer + " "][index] != index for index in piece),
                name,
            )
            wing_names.append(name)

        self.assertEqual(len(wing_names), 48)
        self.assertIn("DB@2L", wing_names)
        self.assertIn("DF@2R", wing_names)
        self.assertIn("DL@2F", wing_names)
        self.assertIn("DR@2B", wing_names)

    def test_multi_sticker_center_orientation_is_not_hidden(self):
        puzzle = CtoCube()
        name = MypermEffectAnalyzer(puzzle).proposed_name(make_myperm_key("CTO-Center1-A", 0))
        self.assertEqual(name, "Ctr1[U>ULB.UBR.URF.UFL]")

    def test_unregistered_last_perms_key_uses_effect_name(self):
        current_key = make_myperm_key("C2[UFL>FLU;URF>FUR]", 0)
        moves = self.cube.myperms[resolve_myperm_key(self.cube, current_key)]
        self.cube.reset()
        for move in self.cube.invert_moves(moves):
            self.cube.make_move(move)
        state_data = self.cube.makedata().reshape(-1, 1)
        group_names = dict.fromkeys(self.cube._group_name_map().values())
        expected_changed_number = sum(
            int(round(self.cube.total_val[group] - (self.cube.group_val[group] @ state_data)[0][0], 0))
            for group in group_names
        )
        solve_state = SolveSessionState()
        frame = SimpleNamespace(
            cube = self.cube,
            solve_state = solve_state,
            myperms_col = {},
        )

        SolveSessionManager(frame)._store_perfect_key(moves)

        self.assertEqual(solve_state.last_perfect_key, "LP:C2[UFL>FLU;URF>FUR]")
        self.assertEqual(solve_state.last_perfect_changed_number, expected_changed_number)
        self.cube.reset()

    def test_registered_last_perms_key_uses_base_transform_moves(self):
        base_key = make_myperm_key("C2[UFL>FLU;URF>FUR]", 0)
        source_key = make_myperm_key("C2[UFL>FLU;URF>FUR]", 1)
        moves = self.cube.myperms[source_key]
        myperms_col = {}
        for key, registered_moves in self.cube.myperms.items():
            self.cube.reset()
            for move in self.cube.invert_moves(registered_moves):
                self.cube.make_move(move)
            myperms_col.setdefault("".join(self.cube.state), key)

        self.cube.reset()
        for move in self.cube.invert_moves(moves):
            self.cube.make_move(move)
        solve_state = SolveSessionState()
        frame = SimpleNamespace(
            cube = self.cube,
            solve_state = solve_state,
            myperms_col = myperms_col,
        )

        returned_moves = SolveSessionManager(frame)._store_perfect_key(moves)

        self.assertEqual(solve_state.last_perfect_key, "C2[UFL>FLU;URF>FUR]")
        self.assertEqual(returned_moves, self.cube.myperms[base_key])
        self.assertEqual(solve_state.last_simplified_lis, self.cube.myperms[base_key])
        self.cube.reset()

    def test_unregistered_last_perms_key_uses_point_representative_transform(self):
        source_key = make_myperm_key("C2[UFL>FLU;URF>FUR]", 1)
        moves = self.cube.myperms[resolve_myperm_key(self.cube, source_key)]
        solve_state = SolveSessionState()
        frame = SimpleNamespace(
            cube = self.cube,
            solve_state = solve_state,
            myperms_col = {},
        )

        returned_moves = SolveSessionManager(frame)._store_perfect_key(moves)

        self.assertEqual(solve_state.last_perfect_key, "LP:C2[UFL>FLU;URF>FUR]")
        self.assertEqual(returned_moves, solve_state.last_simplified_lis)
        self.assertNotEqual(tuple(moves), solve_state.last_simplified_lis)
        self.cube.reset()

    def test_supported_puzzles_can_analyze_a_registered_myperm(self):
        puzzle_factories = (
            MegaminxCube,
            PyraminxCube,
            MasterPyraminxCube,
            SkewbCube,
            FtoCube,
            CtoCube,
        )
        for puzzle_factory in puzzle_factories:
            puzzle = puzzle_factory()
            with self.subTest(puzzle = puzzle_factory.__name__):
                base_name = next(iter(puzzle.myperms2))
                effect = MypermEffectAnalyzer(puzzle).analyze(make_myperm_key(base_name, 0))
                self.assertGreater(effect.moved_count, 0)
                self.assertNotEqual(effect.concise_name(), "Identity")
            del puzzle
            gc.collect()


if __name__ == "__main__":
    unittest.main()
