import numpy as np

from kalah.agents.randomagent import RandomAgent
from kalah.agents.maxscoreagent import MaxScoreAgent
from kalah.agents.maxscorerepeatagent import MaxScoreRepeatAgent
from kalah.agents.minimaxagent import MinimaxAgent

class KalahAgentFactory:
    """
    Produces agents
    """

    _minimax_depth = 4

    def __init__(self, agents=None, weights=None, seed=543):
        if agents is None and weights is None:
            agents = np.array([
                RandomAgent(seed),
                MaxScoreAgent(seed),
                MaxScoreRepeatAgent(seed),
                MinimaxAgent(seed, self._minimax_depth)
            ])
            weights = np.array([0.1, 0.2, 0.5, 0.2])

        agents = np.array(agents)
        weights = np.array(weights)

        if len(agents) != len(weights):
            raise ValueError('Did not specify the same number of agents and weights')

        self._seed = seed
        self._agents = agents
        self._weights = weights / sum(weights)

    def get_random_agent(self):
        """Returns a randomly selected opponent agent"""
        return np.random.choice(self._agents, 1, p=self._weights)[0]
