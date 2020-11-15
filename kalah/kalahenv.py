import gym
from gym import spaces

import numpy as np

from kalah.kalahboard import KalahBoard
from kalah.kalahagentfactory import KalahAgentFactory

class KalahEnv(gym.Env):
    """
    A Kalah environment for OpenAI gym
    """

    metadata = {'render.modes': ['human']}

    _board = None

    def __init__(self):
        pass

    def _get_obs(self):
        player_board_one = self._board._get_player_board(0)[:-1]
        player_board_two = self._board._get_player_board(1)[:-1]

        board = player_board_one + player_board_two
        obs = np.array(board)/(2*self._board.bins)
        return obs

    def set_board(self, board):
        self._board = board
        self.action_space = spaces.Discrete(board.bins)

    def set_agent_factory(self, agent_factory):
        self.agent_factory = agent_factory

    def step(self, action):
        reward = 0
        info = []

        if self._board == None:
            raise ValueError("Board not set")

        opponent = self.agent_factory.get_random_agent()

        player = self._board.current_player()
        other_player = 1 if self._board.current_player() == 0 else 0

        old_score = self._board.score()[player]

        success = self._board.move(action)
        if success == False:
            return self._get_obs(), -10, True, info

        # Perform the opponents moves
        while not self._board.game_over() and self._board.current_player() != player:
            self._board.move(opponent.get_next_move(self._board))

        if self._board.game_over():
            if self._board.score()[player] > self._board.score()[other_player]:
                reward = 100
            if self._board.score()[player] <= self._board.score()[other_player]:
                reward = -10
        else:
            reward = self._board.score()[player] - old_score

        return self._get_obs(), reward, self._board.game_over(), info

    def reset(self):
        if self._board == None:
            raise ValueError("Board is not set")

        self._board.reset()
        return self._get_obs()

    def render(self, mode='human'):
        if self._board != None:
            if mode == 'human':
               self._board.pretty_print()
            else:
                super(KalahEnv, self).render(mode=mode)
        else:
            raise ValueError("Board is not set")

    def close(self):
        pass
