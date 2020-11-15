import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

import numpy as np

class TfAgent:
    """
    A wrapper around a TF reinforce model for Kalah
    """

    def __init__(self, model, seed=543):
        self.model = model

    def _get_obs(self, board):
        player_board_one = board._get_player_board(0)[:-1]
        player_board_two = board._get_player_board(1)[:-1]

        board_obs = player_board_one + player_board_two
        obs = np.array(board_obs)/(2*board.bins)
        return obs

    def select_action(self, state):
        probs = self.model(np.array([state]))
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()

        return int(action.numpy())

    def get_next_move(self, board):
        obs = self._get_obs(board)
        return self.select_action(obs)
