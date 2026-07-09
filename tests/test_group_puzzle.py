import unittest
from types import SimpleNamespace

import numpy as np

from group_puzzle.cube import LinearGroupPuzzle, SymmetricGroupPuzzle
from managers.debug_analysis import DebugAnalysisManager
from managers.solve_session import SolveSessionManager


class SymmetricGroupPuzzleTest(unittest.TestCase):
    def test_left_action_and_inverse(self):
        puzzle = SymmetricGroupPuzzle()
        puzzle.make_move("s")
        puzzle.make_move("r")
        self.assertEqual(puzzle.state.tolist(), [3, 2, 1])  # r composed with s
        for move in puzzle.invert_moves(("s", "r")):
            puzzle.make_move(move)
        self.assertTrue(puzzle.is_perfect())

    def test_one_hot_block_per_position(self):
        puzzle = SymmetricGroupPuzzle(degree=3)
        puzzle.make_move("s")
        data = puzzle.makedata().reshape(3, 3)
        np.testing.assert_array_equal(
            data,
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype="f"),
        )
        self.assertEqual(puzzle.last_perms_key(), "S_3:[2,1,3]")

    def test_exact_length_scramble(self):
        puzzle = SymmetricGroupPuzzle()
        moves = puzzle.scramble(11)
        scrambled_state = puzzle.state.copy()
        self.assertEqual(len(moves), 11)
        puzzle.reset()
        for move in moves:
            puzzle.make_move(move)
        np.testing.assert_array_equal(puzzle.state, scrambled_state)

    def test_scramble_has_no_adjacent_inverse_moves(self):
        puzzle = SymmetricGroupPuzzle()
        moves = puzzle.scramble(500)
        for left, right in zip(moves, moves[1:]):
            self.assertNotEqual(right, puzzle.invert_str(left))

    def test_impossible_reduced_word_length_is_rejected(self):
        puzzle = SymmetricGroupPuzzle(degree=2)
        with self.assertRaisesRegex(ValueError, "add another generator"):
            puzzle.scramble(2)

    def test_generator_set_must_be_inverse_closed(self):
        with self.assertRaisesRegex(ValueError, "inverse-closed"):
            SymmetricGroupPuzzle(
                degree=3,
                generators={"r": [2, 3, 1]},
            )


class LinearGroupPuzzleTest(unittest.TestCase):
    def test_left_matrix_action_and_inverse(self):
        puzzle = LinearGroupPuzzle(dimension=2, modulus=3)
        puzzle.make_move("A")
        puzzle.make_move("B")
        expected = np.array([[0, 1], [1, 1]], dtype=int)  # B @ A over F_3
        np.testing.assert_array_equal(puzzle.state.reshape(2, 2), expected)
        for move in puzzle.invert_moves(("A", "B")):
            puzzle.make_move(move)
        self.assertTrue(puzzle.is_perfect())

    def test_one_hot_block_per_matrix_entry(self):
        puzzle = LinearGroupPuzzle(dimension=2, modulus=2)
        puzzle.make_move("A")
        data = puzzle.makedata().reshape(4, 2)
        np.testing.assert_array_equal(
            data,
            np.array([[0, 1], [0, 1], [1, 0], [0, 1]], dtype="f"),
        )
        self.assertEqual(puzzle.last_perms_key(), "GL_2(F_2):[[1,1],[0,1]]")

    def test_rejects_non_prime_modulus(self):
        with self.assertRaisesRegex(ValueError, "prime"):
            LinearGroupPuzzle(dimension=2, modulus=4)

    def test_sl2_f7_auto_completes_inverse_generators(self):
        puzzle = LinearGroupPuzzle(
            dimension=2,
            modulus=7,
            family="SL",
            generators={
                "A": [[1, 1], [0, 1]],
                "B": [[1, 0], [1, 1]],
            },
            auto_add_inverses=True,
            display_name="SL_2(F_7)",
        )
        self.assertEqual(puzzle.move_keys, ("A", "A^-1", "B", "B^-1"))
        np.testing.assert_array_equal(puzzle.generators["A^-1"], [[1, 6], [0, 1]])
        np.testing.assert_array_equal(puzzle.generators["B^-1"], [[1, 0], [6, 1]])
        self.assertEqual(puzzle.invert_str("A"), "A^-1")
        self.assertEqual(puzzle.invert_str("B^-1"), "B")
        puzzle.make_move("A")
        self.assertEqual(puzzle.last_perms_key(), "SL_2(F_7):[[1,1],[0,1]]")
        moves = puzzle.scramble(500)
        for left, right in zip(moves, moves[1:]):
            self.assertNotEqual(right, puzzle.invert_str(left))

    def test_sl_rejects_non_unit_determinant(self):
        with self.assertRaisesRegex(ValueError, "determinant 1"):
            LinearGroupPuzzle(
                dimension=2,
                modulus=7,
                family="SL",
                generators={"bad": [[2, 0], [0, 1]]},
                auto_add_inverses=True,
            )

    def test_solve_session_uses_matrix_as_last_perms_key(self):
        puzzle = LinearGroupPuzzle(
            dimension=2,
            modulus=7,
            family="SL",
            generators={"A": [[1, 1], [0, 1]]},
            auto_add_inverses=True,
            display_name="SL_2(F_7)",
        )
        puzzle.make_move("A")
        solve_state = SimpleNamespace(
            last_perfect_key="",
            last_perfect_changed_number=0,
            last_simplified_lis=(),
        )
        frame = SimpleNamespace(cube=puzzle, solve_state=solve_state, myperms_col={})
        SolveSessionManager(frame)._store_perfect_key(())
        self.assertEqual(solve_state.last_perfect_key, "SL_2(F_7):[[1,1],[0,1]]")
        self.assertNotIn("Entry", solve_state.last_perfect_key)

    def test_group_w1_embedding_projection_metadata(self):
        puzzle = LinearGroupPuzzle(dimension=2, modulus=7)
        rng = np.random.default_rng(7)
        ai = SimpleNamespace(
            params={
                "W1": rng.normal(size=(8, puzzle.ips)).astype("f"),
                "WQ1": rng.normal(size=(4, 8)).astype("f"),
            },
            use_transformer_attention=True,
        )
        frame = SimpleNamespace(
            cube=puzzle,
            AIs=[ai],
            grad_index=0,
            grad_mode="Grad",
            grad_layer="WO_V",
            puzzle_type="group",
        )
        manager = DebugAnalysisManager(frame)
        result = manager.w1_embedding_projection(0, embedding_source="W1")
        self.assertEqual(result["labels"][0], "a00 / value=0")
        self.assertEqual(result["piece_types"][:7], ["Diagonal"] * 7)
        self.assertEqual(result["piece_types"][7:14], ["OffDiagonal"] * 7)
        self.assertEqual(result["solve_groups"][:14], ["Row 0"] * 14)
        self.assertEqual(result["solve_groups"][14:], ["Row 1"] * 14)
        self.assertEqual(int(np.count_nonzero(result["correct_flags"])), 4)

        projected = manager.w1_embedding_projection(0, embedding_source="WQ1 @ W1")
        self.assertEqual(projected["embedding_source"], "WQ1 @ W1")
        self.assertEqual(projected["embedding_shape"], (4, puzzle.ips))


if __name__ == "__main__":
    unittest.main()
