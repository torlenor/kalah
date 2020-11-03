import random

from kalah.kalahboard import KalahBoard

import kalah.utils as utils

import unittest

class Test_utils(unittest.TestCase):
    def test_get_best_moves(self):
        board = KalahBoard(4,4)
        self.assertEqual(board.get_board(), [4, 4, 4, 4, 0, 4, 4, 4, 4, 0])
        self.assertEqual(board.allowed_moves(), [0, 1, 2, 3])

        self.assertEqual(utils.get_best_moves(board), [0, 1, 2, 3])

        board.set_board([4, 0, 5, 5, 1, 5, 4, 4, 4, 0])
        self.assertEqual(board.allowed_moves(), [0, 2, 3])
        self.assertEqual(utils.get_best_moves(board), [0, 2, 3])

        board.set_board([4, 0, 5, 5, 1, 5, 4, 4, 4, 0])
        self.assertEqual(board.allowed_moves(), [0, 2, 3])
        self.assertEqual(utils.get_best_moves(board), [0, 2, 3])

        board.set_current_player(1)
        board.set_board([4, 0, 5, 5, 1, 5, 4, 4, 4, 0])
        self.assertEqual(board.allowed_moves(), [5, 6, 7, 8])
        self.assertEqual(utils.get_best_moves(board), [5, 6, 7, 8])

if __name__ == '__main__':
    unittest.main()
