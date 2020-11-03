import argparse
import numpy as np
import random
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

import matplotlib
from matplotlib.pylab import plt

from kalah.kalahboard import KalahBoard
from kalah.kalahenv import KalahEnv

from kalah.agents.randomagent import RandomAgent
from kalah.agents.maxscoreagent import MaxScoreAgent
from kalah.agents.maxscorerepeatagent import MaxScoreRepeatAgent
from kalah.agents.minimaxagent import MinimaxAgent
from kalah.agents.pytorchagent import PyTorchAgent

from kalah.utils import compare_agents

from kalah.models.reinforce import ReinforceModel

from kalah.kalahagentfactory import KalahAgentFactory

from args import add_common_train_args

parser = argparse.ArgumentParser(description='Train a REINFORCE model to play Kalah')
add_common_train_args(parser)
parser.add_argument('--drop-out', type=float, default=0.1, metavar='DR',
                    help='drop out between layers, when 0 it is disabled (default: 0.1)')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

opponent_agent_class = MaxScoreRepeatAgent

env = KalahEnv()
env.set_board(KalahBoard(args.bins, args.seeds))
env.set_agent_factory(KalahAgentFactory(agent_classes=list([MaxScoreRepeatAgent]), weights=list([1])))
env.seed(args.seed)

c_count = 0

def run_compare(policy):
    n_games = args.evaluation_games

    global c_count
    c_count += 1

    agent1 = PyTorchAgent(policy)
    agent2 = opponent_agent_class(seed=args.seed+c_count)

    [wins_agent_one, _, draws, invalid_moves] = compare_agents(args.bins, args.seeds, n_games, agent1, agent2)

    win_percentage1 = -1
    if draws != n_games:
        win_percentage1 = 100*wins_agent_one / (n_games - draws)
        print("Agent 1 won", win_percentage1 ,"% ( n =", wins_agent_one, ") of all N =", n_games ,"games. There were", draws, "draws and ", invalid_moves, "invalid moves.")
    else:
        print("Only draws")
    
    return [n_games, win_percentage1, draws]

eps = np.finfo(np.float32).eps.item()

policy = ReinforceModel(args.bins*2, args.bins, args.neurons, args.drop_out)
optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def main():
    results_wins_agent1 = []
    results_draws = []

    running_reward = 10
    for i_episode in range(args.episodes):
        policy.train()

        state, ep_reward = env.reset(), 0
        for _ in range(1, 10000):
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()

        solved = False
        last_win_percentage = 0
        if i_episode % args.log_interval == 0:
            print("Comparing @ Episode", i_episode, end=': ')
            policy.eval()
            n_games, wins_agent_one, draws = run_compare(policy)
            results_wins_agent1.append([i_episode, wins_agent_one])
            results_draws.append([i_episode, draws])
            if wins_agent_one > args.solved:
                solved = True
                last_win_percentage = wins_agent_one

        if solved:
            print("Solved after {} episodes! The last win percentage was {:2,f}".format(i_episode, last_win_percentage))
            break

    arr_results_wins_agent1 = np.array(results_wins_agent1)

    fig = plt.plot(arr_results_wins_agent1[:,0], arr_results_wins_agent1[:,1], label="Wins Percentage Agent 1")
    plt.legend(loc='lower right')
    plt.xlabel("Episode")
    plt.ylabel("Win percentage")
    plt.ylim(0, 110)
    plt.show()

    fig.clear()

    print("")

    print("Running final comparisons:")
    policy.eval()

    n_games = 1000

    agent_class1 = PyTorchAgent

    agent_classes = [RandomAgent, MaxScoreAgent, MaxScoreRepeatAgent, MinimaxAgent]

    agents = []
    win_percentage_for_agent = []
    for agent_class2 in agent_classes:
        agent1 = agent_class1(policy)
        agent2 = agent_class2()

        [wins_agent_one, _, draws, _] = compare_agents(args.bins, args.seeds, n_games, agent1, agent2)

        win_percentage = 100*wins_agent_one / (n_games - draws)

        if draws != n_games:
            print(agent_class1.__name__, "won", win_percentage,"% ( n =", wins_agent_one, ") of all N =", n_games ,"games against", agent_class2.__name__, "Number of draws:", draws)
        else:
            print("Only draws in", agent_class1.__name__, "vs", agent_class2.__name__)

        agents.append(agent_class2.__name__)
        win_percentage_for_agent.append(win_percentage)

    plt.clf()

    fig, ax = plt.subplots()

    x = np.arange(len(agents))
    width = 0.35
    ax.bar(x - width/2, win_percentage_for_agent, width)

    ax.set_ylabel('Win percentage')
    ax.set_xlabel('Opponent')
    ax.set_title('Wins against various agents')
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylim(0,110)

    plt.show()

    fig.clear()

if __name__ == '__main__':
    main()
