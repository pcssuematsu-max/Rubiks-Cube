import unittest

from core.myperm_points import (
    MypermPointCalculator,
    load_myperm_points,
    parse_myperm_points_text,
)
from core.myperm_keys import resolve_myperm_key
from cube.rubiks_cube import Rubiks_3


class MypermPointTableTest(unittest.TestCase):
    def test_parser_normalizes_corner_midedge_and_wing_positions(self):
        table = parse_myperm_points_text(
            """
            #Corners
            UFR:2 DBL:0.5
            #MidEdge
            UF:20 DB:10 FR:50
            #Wing
            FR@U:500 DB@L:100 BL@D:320
            #XCenter
            R@2F.2U:2000
            Others:0
            """
        )

        self.assertEqual(table.point_for_part("C", "URF"), 2)
        self.assertEqual(table.point_for_part("C", "LDB"), 0.5)
        self.assertEqual(table.point_for_part("ME", "UF@M"), 20)
        self.assertEqual(table.point_for_part("ME", "RF@M"), 50)
        self.assertEqual(table.point_for_part("W2", "FR@2U"), 500)
        self.assertEqual(table.point_for_part("W2", "RF@2U"), 500)
        self.assertEqual(table.point_for_part("W2", "LB@2D"), 320)
        self.assertEqual(table.point_for_part("CtrX", "R@2F.2U"), 2000)
        self.assertEqual(table.point_for_part("CtrX", "R@2U.2F"), 2000)
        self.assertEqual(table.point_for_part("CtrX", "R@2F.2D"), 0)

    def test_current_points_file_scores_a_real_wing_myperm(self):
        cube = Rubiks_3(size = 7)
        table = load_myperm_points()
        calculator = MypermPointCalculator(cube, table)
        key = resolve_myperm_key(cube, "WingParallel6-A")

        self.assertEqual(calculator.point_for_key(key), 2360)
        self.assertIn("BR@D>LB@U>FL@D", key[0])

    def test_edge_bundle_scores_mid_edge_and_each_wing_layer(self):
        cube = Rubiks_3(size = 7)
        table = load_myperm_points()

        self.assertEqual(table.edge_bundle_point(cube, "UF@M"), 820)


if __name__ == "__main__":
    unittest.main()
