import argparse
import time
from random import random
from datetime import datetime, timedelta

from kalah.kalahboard import KalahBoard

from kalah.agents.randomagent import RandomAgent
from kalah.agents.maxscoreagent import MaxScoreAgent
from kalah.agents.maxscorerepeatagent import MaxScoreRepeatAgent
from kalah.agents.minimaxagent import MinimaxAgent

class KalahBattleGroundResults:
    n_games = -1
    wins_agent1 = -1
    wins_agent2 = -1
    draws = -1

    def __init__(self, n_games, wins_agent1, wins_agent2, draws):
        self.n_games = n_games
        self.wins_agent1 = wins_agent1
        self.wins_agent2 = wins_agent2
        self.draws = draws

class KalahBattleground:
    """
    A battleground to pit two agents against each other
    """

    def __init__(self, bins, seeds):
        self._bins = bins
        self._seeds = seeds

    def battle(self, lambda_agent1, lambda_agent2, n_games=1000, seed=543):
        """
        Start a battle of n_games

        lambda_agent1 and lambda_agent2 are lambda functions with seed as argument
        """

        wins_agent1 = 0
        wins_agent2 = 0
        draws = 0

        current_game = 0
        cnt = 0
        while current_game < n_games:
            board = KalahBoard(self._bins, self._seeds)

            agent1 = (lambda_agent1)(seed + cnt)
            cnt += 1
            agent2 = (lambda_agent2)(seed + cnt)
            cnt += 1

            last_invalid_player = None
            invalid_count = 0
            while not board.game_over():
                if board.current_player() == 0:
                   valid = board.move(agent1.get_next_move(board))
                else:
                    valid = board.move(agent2.get_next_move(board))
                if not valid:
                    if last_invalid_player == board.current_player():
                        invalid_count += 1
                    else:
                        invalid_count = 0
                        last_invalid_player = board.current_player()
                if invalid_count > 10:
                    break

            if invalid_count > 10:
                if last_invalid_player == 0:
                    wins_agent1 += 1
                else:
                    wins_agent2 += 1
            else:
                if board.score()[0] > board.score()[1]:
                    wins_agent1 += 1
                elif board.score()[0] < board.score()[1]:
                    wins_agent2 += 1
                else:
                    draws += 1

            current_game += 1

        return KalahBattleGroundResults(n_games, wins_agent1, wins_agent2, draws)
