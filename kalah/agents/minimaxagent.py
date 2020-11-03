import copy
import datetime
import random
import sys

class MinimaxAgent:
    """
    This is a Minimax agent with alpha-beta pruning for Kalah

    It uses the score of each player as the value of the leaf nodes.
    """

    max_depth = 4

    first_round = True
    player = 0

    def __init__(self, seed=42, depth=4):
        self.random = random.Random(seed)
        self.max_depth = depth

    def minimax(self, player, depth, board, move, alpha, beta):
        if depth == 0:
            return board.score()[player]

        test_board = board.copy()
        test_board.move(move)

        maxi = (test_board.current_player() == player)

        move_options = test_board.allowed_moves()
        best_move = -sys.maxsize if maxi else sys.maxsize

        for move_slot in move_options:
            current_value = self.minimax(
                player, depth - 1, test_board, move_slot, alpha, beta)

            if maxi:
                best_move = max(current_value, best_move)
                alpha = max(alpha, best_move)
            else:
                best_move = min(current_value, best_move)
                beta = min(beta, best_move)

            if beta <= alpha:
                return best_move

        return best_move

    def get_next_move(self, board):
        # Determine which player we are
        if self.first_round:
            self.player = board.current_player()
            self.first_round = False

        # If there is only one valid move, return that move
        allowed_moves = board.allowed_moves()
        if len(allowed_moves) == 1:
            return allowed_moves[0]

        moves_and_scores = []
        for move in board.allowed_moves():
            minimax_score = self.minimax(
                self.player, self.max_depth, board, move, -sys.maxsize, sys.maxsize)
            moves_and_scores.append([move, minimax_score])

        scores = [item[1] for item in moves_and_scores]
        max_score = max(scores)

        potential_moves = []
        for move_and_score in moves_and_scores:
            if move_and_score[1] == max_score:
                potential_moves.append(move_and_score[0])

        return self.random.choice(potential_moves)
