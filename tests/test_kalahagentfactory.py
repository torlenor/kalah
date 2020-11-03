import numpy as np

from kalah.agents.randomagent import RandomAgent
from kalah.agents.maxscoreagent import MaxScoreAgent
from kalah.agents.maxscorerepeatagent import MaxScoreRepeatAgent
from kalah.agents.minimaxagent import MinimaxAgent

from kalah.kalahagentfactory import KalahAgentFactory

import unittest

class Test_KalahAgentFactory(unittest.TestCase):
    def test_get_random_agent(self):
        factory = KalahAgentFactory()

        agent = factory.get_random_agent()
        self.assertIsNotNone(agent)

        agent_classes = np.array([
            RandomAgent(),
            MaxScoreAgent(),
            MaxScoreRepeatAgent(),
            MinimaxAgent()
        ])

        weights = np.array([1, 0, 0, 0])
        factory = KalahAgentFactory(agent_classes, weights)
        agent = factory.get_random_agent()
        self.assertEqual(agent.__class__.__name__, RandomAgent.__name__)

        weights = np.array([0, 1, 0, 0])
        factory = KalahAgentFactory(agent_classes, weights)
        agent = factory.get_random_agent()
        self.assertEqual(agent.__class__.__name__, MaxScoreAgent.__name__)

        weights = np.array([0, 0, 1, 0])
        factory = KalahAgentFactory(agent_classes, weights)
        agent = factory.get_random_agent()
        self.assertEqual(agent.__class__.__name__, MaxScoreRepeatAgent.__name__)

        weights = np.array([0, 0, 0, 1])
        factory = KalahAgentFactory(agent_classes, weights)
        agent = factory.get_random_agent()
        self.assertEqual(agent.__class__.__name__, MinimaxAgent.__name__)

    def test_fail(self):
        agent_classes = np.array([
            RandomAgent,
            MaxScoreAgent,
            MaxScoreRepeatAgent,
            MinimaxAgent
        ])

        weights = np.array([1, 0, 0])
        with self.assertRaises(ValueError):
            KalahAgentFactory(agent_classes, weights)

if __name__ == '__main__':
    unittest.main()
