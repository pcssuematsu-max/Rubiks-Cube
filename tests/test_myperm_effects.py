import unittest
from types import SimpleNamespace
import gc

from core.myperm_effects import MypermEffectAnalyzer
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
        name = self.analyzer.proposed_name(make_myperm_key("CornerPermutation-A00", 0))
        self.assertEqual(name, "C3[UBR>UFL>URF]")

    def test_registered_key_is_renamed_and_legacy_name_resolves(self):
        new_key = make_myperm_key("C3[UBR>UFL>URF]", 0)
        old_key = make_myperm_key("CornerPermutation-A00", 0)
        self.assertIn(new_key, self.cube.myperms)
        self.assertNotIn(old_key, self.cube.myperms)
        self.assertEqual(resolve_myperm_key(self.cube, old_key), new_key)

    def test_edge_flip_and_corner_twist_include_orientation(self):
        edge_name = self.analyzer.proposed_name(make_myperm_key("EdgeFlip2-A", 0))
        corner_name = self.analyzer.proposed_name(make_myperm_key("CornerTwist-A", 0))
        self.assertEqual(edge_name, "E2[UL>LU;UR>RU]")
        self.assertEqual(corner_name, "C2[UBR>BRU;ULB>BUL]")

    def test_transformed_key_uses_transformed_positions(self):
        original = self.analyzer.proposed_name(make_myperm_key("CornerPermutation-A00", 0))
        transformed = self.analyzer.proposed_name(make_myperm_key("CornerPermutation-A00", 1))
        self.assertEqual(transformed, "C3[DBL>DLF>DRB]")
        self.assertNotEqual(original, transformed)

    def test_large_effect_uses_compact_counts(self):
        effect = self.analyzer7.analyze(make_myperm_key("SuperFlip", 0))
        self.assertEqual(effect.concise_name(), "EAll12[XY>YX]")
        self.assertTrue(any(component.part_code.startswith("Ctr") for component in effect.components))

    def test_outer_center_bar_uses_bar_notation(self):
        self.assertEqual(
            self.analyzer7.proposed_name(make_myperm_key("OuterCenterBar-A", 0)),
            "CtrBar3[F@2L>U@2R>U@2L]",
        )
        self.assertEqual(
            self.analyzer7.proposed_name(make_myperm_key("OuterCenterBar-ZZ", 0)),
            "CtrBar4s[D@2L<>U@2F;D@2R<>U@2B]",
        )

    def test_mid_center_bar_uses_mid_bar_notation(self):
        self.assertEqual(
            self.analyzer7.proposed_name(make_myperm_key("MidCenterBar(VV)", 0)),
            "CtrMidBar6p[3x2][F@D>U@F>F@L;F@R>F@U>U@B]",
        )
        self.assertEqual(
            self.analyzer7.proposed_name(make_myperm_key("MidCenterBar-Adjacent3Center-OB", 0)),
            "CtrMidBar6p[3x2][F@D>R@D>U@L;F@U>R@U>U@R]",
        )

    def test_center_midedge_and_corner_swap_source_names_are_effect_names(self):
        midedge_key = make_myperm_key("CtrCore4[D>L>U>R]+ME2s[UL<>UR]", 0)
        corner_key = make_myperm_key("C2[UBR>RFU]+CtrCore4[D>L>U>R]~v01", 0)

        self.assertIn(midedge_key, self.cube7.myperms)
        self.assertIn(corner_key, self.cube7.myperms)
        self.assertEqual(resolve_myperm_key(self.cube7, "CenterMidEdgeSwap-QA"), midedge_key)
        self.assertEqual(resolve_myperm_key(self.cube7, "CenterCornerSwap-A00"), corner_key)

    def test_commutator_source_names_resolve_by_size_specific_effects(self):
        self.assertEqual(
            resolve_myperm_key(self.cube, "OutCommutator00"),
            make_myperm_key("C4[DLF>FLU;UBR>RFU;UFL>LFD;URF>UBR]+E3[FL>FU>RU]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "OutCommutator00"),
            make_myperm_key("C4[DLF>FLU;UBR>RFU;UFL>LFD;URF>UBR]+EAll3[FL>FU>RU]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "SideCommutator00"),
            make_myperm_key("CtrObl6p[3x2]+CtrPlus3+CtrX5+W2-5~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "MidCommutator00"),
            make_myperm_key("CtrPlus12p[3x4]+ME5[DF>FU>FR>LF>BL]", 0),
        )

    def test_midedge4_source_names_are_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, "MidEdge4-H-A"),
            make_myperm_key("ME4s[UB<>UF;UL<>UR]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "MidEdge4-II-A"),
            make_myperm_key("ME4[DB>DF;DF>FU;UB>BU;UF>DB]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "MidEdge4-UU-A"),
            make_myperm_key("ME4[UB>BU;UF>UL;UL>UR;UR>FU]", 0),
        )

    def test_midedge_flip_source_names_are_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, "MidEdgeFlip-A2"),
            make_myperm_key("ME2[UB>BU;UF>FU]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "MidEdgeFlip-A2I"),
            make_myperm_key("ME2[UB>BU;UF>FU]~v02", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "MidEdgeFlip-E4"),
            make_myperm_key("ME4[DB>BD;DF>FD;DL>LD;DR>RD]~v01", 0),
        )

    def test_wing3_source_names_are_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, "Wing3-Parallel-I00"),
            make_myperm_key("W2-3[DB@R>UF@R>UB@R]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "Wing3-Parallel-I01"),
            make_myperm_key("W2-3[DB@R>UB@R>UF@R]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "Wing3-U00"),
            make_myperm_key("W2-3[UB@L>UF@L>UR@F]", 0),
        )

    def test_wing_swap_parallel_source_names_are_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, "WingSwapParallel-A0"),
            make_myperm_key("W2-2s[UB@L<>UF@L]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "WingSwapParallel-K1"),
            make_myperm_key("W2-2s[UF@L<>UF@R]~v02", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "WingSwapParallel-BY07"),
            make_myperm_key("W2-2s[UB@L<>UF@R]~v25", 0),
        )

    def test_wing_swap_skew_source_names_are_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, "WingSwapSkew-C"),
            make_myperm_key("W2-2s[RF@U<>UF@R]", 0),
        )

        centers_cube = Rubiks_3(size = 4, Centers = True)
        self.assertEqual(
            resolve_myperm_key(centers_cube, "WingSwapSkew-C"),
            make_myperm_key("CtrX8p[4x2]+W2-2s[FL@U<>UF@L]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(centers_cube, "WingSwapSkew-Ex"),
            make_myperm_key("CtrX8p[4x2]+W2-2s[DF@R<>FL@D]", 0),
        )

    def test_l2_source_names_are_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, "L2NA"),
            make_myperm_key("W2-4[UL@B>UR@B>UL@F>UR@F]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "L2E-ZB1"),
            make_myperm_key("W2-4[UF@L>UR@F>UR@B>UF@R]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "L2XA"),
            make_myperm_key("W2-4s[UL@B<>UR@F;UL@F<>UR@B]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "L2E-FC0"),
            make_myperm_key("W2-4s[DF@L<>DF@R;UL@B<>UL@F]", 0),
        )

    def test_wing_parallel6_source_names_are_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, "WingParallel6-A"),
            make_myperm_key("W2-6p[3x2][BR@D>LB@U>FL@D;BR@U>LB@D>FL@U]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "WingParallel6-C"),
            make_myperm_key("W2-6p[3x2][BR@D>LB@D>FL@D;BR@U>LB@U>FL@U]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "WingParallel6-E"),
            make_myperm_key("W2-6[BR@D>LB@U>FL@D>BR@U>LB@D>FL@U]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "WingParallel6-G"),
            make_myperm_key("W2-6[BR@D>LB@D>FL@D>BR@U>LB@U>FL@U]", 0),
        )

    def test_edge6p_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, "Edge6PAX"),
            make_myperm_key("W2-4s[FL@D<>UF@R;FL@U<>UF@L]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "Edge6PCX"),
            make_myperm_key("W2-4s[FL@D<>UB@L;FL@U<>UB@R]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "Edge6PAN"),
            make_myperm_key("W2-4[FL@D>UF@R>FL@U>UF@L]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "Edge6PCN"),
            make_myperm_key("W2-4[FL@D>UB@R>FL@U>UB@L]", 0),
        )

    def test_edge_flip_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, "EdgeFlip2-A"),
            make_myperm_key("EAll2[FL>LF;RF>FR]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "EdgeFlip2-D"),
            make_myperm_key("EAll2[LB>BL;RF>FR]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "EdgeFlip4-F"),
            make_myperm_key("EAll4[BR>RB;FL>LF;LB>BL;RF>FR]", 0),
        )

    def test_edgepk_source_names_are_point_representative_effect_names(self):
        self.assertEqual(
            resolve_myperm_key(self.cube7, "EdgePK-A00"),
            make_myperm_key("W2-3[FL@D>RF@U>FL@U]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "EdgePK-A01"),
            make_myperm_key("W2-3[FL@D>FL@U>RF@U]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "EdgePK-A08"),
            make_myperm_key("W2-3[FL@U>RF@U>RF@D]", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "EdgePK-D00"),
            make_myperm_key("W2-3[BR@U>FL@U>FL@D]~v01", 0),
        )
        self.assertEqual(
            resolve_myperm_key(self.cube7, "EdgePK-D03"),
            make_myperm_key("W2-3[BR@U>FL@D>FL@U]~v02", 0),
        )

    def test_partial_wing_effect_is_not_collapsed_into_full_edge_bundle(self):
        name = self.analyzer7.proposed_name(make_myperm_key("WingSwapSkew-C", 0))
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
        old_key = make_myperm_key("CornerTwist-A", 0)
        moves = self.cube.myperms[resolve_myperm_key(self.cube, old_key)]
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

        self.assertEqual(solve_state.last_perfect_key, "LP:C2[UBR>BRU;ULB>BUL]")
        self.assertEqual(solve_state.last_perfect_changed_number, expected_changed_number)
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
