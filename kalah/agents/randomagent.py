import random

from kalah.kalahboard import KalahBoard

class RandomAgent:
    """This is a random agent for Kalah

    It randomly chooses from the allowed moves
    """

    def __init__(self, seed=42):
        self.random = random.Random(seed)

    def get_next_move(self, board: KalahBoard):
        return self.random.choice(board.allowed_moves())
