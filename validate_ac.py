import argparse
import datetime
import random
from collections import namedtuple
from itertools import count

from matplotlib.pylab import plt
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from kalah.agents.actorcriticagent import ActorCriticAgent
from kalah.agents.maxscoreagent import MaxScoreAgent
from kalah.agents.maxscorerepeatagent import MaxScoreRepeatAgent
from kalah.agents.minimaxagent import MinimaxAgent
from kalah.agents.randomagent import RandomAgent

from kalah.kalahboard import KalahBoard
from kalah.kalahenv import KalahEnv

from kalah.models.actorcritic import ActorCriticModel

from kalah.utils import compare_agents

parser = argparse.ArgumentParser(description='Validate an actor-critic model for Kalah')
parser.add_argument('--bins', type=int, default=6, metavar='B',
                    help='bins of the Kalah board (default: 6)')
parser.add_argument('--seeds', type=int, default=4, metavar='S',
                    help='seeds of the Kalah board (default: 4)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--validation-games', type=int, default=100, metavar='EG',
                    help='how many games to play (default: 100)')
parser.add_argument('--model-path', type=str, default='./model.pt', metavar='MP',
                    help='path to trained model (default: ./model.pt)')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

def final_compare(model):
    print("Comparisons with other agent:")

    n_games = args.validation_games

    agent_class1 = ActorCriticAgent

    agent_classes = [RandomAgent, MaxScoreAgent, MaxScoreRepeatAgent, MinimaxAgent]

    for agent_class2 in agent_classes:
        agent1 = agent_class1(model, seed=args.seed)
        agent2 = agent_class2(seed=args.seed)

        [wins_agent_one, _, draws, _] = compare_agents(args.bins, args.seeds, n_games, agent1, agent2)

        win_percentage = 100*wins_agent_one / n_games

        print(agent_class1.__name__, "won", win_percentage,"% ( n =", wins_agent_one, ") of all N =", n_games ,"games against", agent_class2.__name__, "Number of draws:", draws)

def main():
    model = torch.load(args.model_path)
    model.eval()

    final_compare(model)

if __name__ == '__main__':
    main()
