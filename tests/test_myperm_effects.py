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
