import copy
import random

from kalah.kalahboard import KalahBoard

import kalah.utils as utils

class MaxScoreRepeatAgent:
    """
    This is a max score repeat agent for Kalah
    
    It tries to maximize the number moves per turn and in addition it tried
    to hit its own house with the last seed, so that it gets another move.
    """

    def __init__(self, seed=42):
        self.random = random.Random(seed)
        pass

    def get_next_move(self, board: KalahBoard):
        house_id = board.get_house_id(board.current_player())

        for mv in sorted(board.allowed_moves(), reverse=True):
            seeds = board.get_board()[mv]
            if (house_id - mv) == seeds:
                return mv

        return self.random.choice(utils.get_best_moves(board))
