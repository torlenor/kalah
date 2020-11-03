import copy
import random

from kalah.kalahboard import KalahBoard

import kalah.utils as utils

class MaxScoreAgent:
    """
    This is a max score agent for Kalah

    It tries to maximize the score from each move
    """

    def __init__(self, seed=42):
        self.random = random.Random(seed)
        pass

    def get_next_move(self, board: KalahBoard):
        return self.random.choice(utils.get_best_moves(board))
