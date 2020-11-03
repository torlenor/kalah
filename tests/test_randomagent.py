import random

from kalah.kalahboard import KalahBoard

from kalah.agents.randomagent import RandomAgent

import unittest

class Test_RandomAgent(unittest.TestCase):
    def test_randomagent(self):
        board = KalahBoard(6,4)
        self.assertEqual(board.get_board(), [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0])
        self.assertEqual(board.allowed_moves(), [0, 1, 2, 3, 4, 5])

        # We set a fixed random seed and take three known results as a test
        agent = RandomAgent(5)

        self.assertEqual(agent.get_next_move(board), 4)
        self.assertEqual(agent.get_next_move(board), 2)
        self.assertEqual(agent.get_next_move(board), 5)

if __name__ == '__main__':
    unittest.main()
