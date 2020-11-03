import argparse
import datetime
import os
import random
from collections import namedtuple
from itertools import count

from matplotlib.pylab import plt
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from args import add_common_train_args

from kalah.agents.maxscorerepeatagent import MaxScoreRepeatAgent
from kalah.agents.actorcriticagent import ActorCriticAgent

from kalah.kalahagentfactory import KalahAgentFactory
from kalah.kalahbattleground import KalahBattleground, KalahBattleGroundResults
from kalah.kalahboard import KalahBoard
from kalah.kalahenv import KalahEnv

from kalah.models.actorcritic import ActorCriticModel

from kalah.utils import compare_agents

parser = argparse.ArgumentParser(description='Train an actor-critic model to play Kalah')
add_common_train_args(parser)
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# agent to validate against for determining the win rate
opponent_agent_class = MaxScoreRepeatAgent

env = KalahEnv()
env.set_board(KalahBoard(args.bins, args.seeds))
env.set_agent_factory(KalahAgentFactory(seed=args.seed))
env.seed(args.seed)

battleground = KalahBattleground(args.bins, args.seeds)

c_count = 0
def run_compare(model):
    n_games = args.evaluation_games

    global c_count
    c_count += 1

    results = battleground.battle(lambda seed: ActorCriticAgent(model, seed), lambda seed: opponent_agent_class(seed), args.evaluation_games, args.seed+c_count)

    win_percentage1 = 100*results.wins_agent1 / (results.n_games - results.draws)
    if results.draws != n_games:
        print(ActorCriticAgent.__name__, "won", win_percentage1,
            "% of all N =", results.n_games ,"games against", opponent_agent_class.__name__, "Number of draws:", results.draws)

    return [results.n_games, win_percentage1, results.draws]

model = ActorCriticModel(args.bins*2, args.bins, args.neurons)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take
    return action.item()

def finish_episode(epoch):
    """Training code. Calculates actor and critic loss and performs backprop."""
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns_unnormalized = torch.tensor(returns)
    returns = (returns_unnormalized - returns_unnormalized.mean()) / (returns_unnormalized.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]

def train():
    results_wins_agent1 = []
    results_draws = []

    solved = False
    last_win_percentage = 0
    for i_episode in range(args.episodes):
        model.train()

        state = env.reset()
        ep_reward = 0

        for _ in range(1, 10000):
            action = select_action(state)

            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        finish_episode(i_episode)

        if i_episode % args.log_interval == 0:
            model.eval()

            print("Comparing @ Episode", i_episode, end=': ')
            _, win_percentage_agent1, draws = run_compare(model)
            results_wins_agent1.append([i_episode, win_percentage_agent1])
            results_draws.append([i_episode, draws])
            avg_win_percentage_agent1 = 0.5*(win_percentage_agent1+last_win_percentage)
            if avg_win_percentage_agent1 > args.solved:
                solved = True
            last_win_percentage = win_percentage_agent1

        if solved:
            print("Solved after {} episodes! The last win percentage was {:2,f}".format(i_episode, last_win_percentage))
            break

    if not solved:
        print("Not Solved after {} episodes!".format(args.episodes))

    arr_results_wins_agent1 = np.array(results_wins_agent1)

    fig = plt.figure(figsize=(8, 6))

    plt.ylim(0, 110)
    plt.plot(arr_results_wins_agent1[:,0], arr_results_wins_agent1[:,1], label="Wins Percentage Agent 1")
    plt.legend(loc='lower right')
    plt.xlabel("Episode")
    plt.ylabel("Win percentage")
    plt.show()

    fig.savefig('train_win_percentage_' + os.path.basename(args.model_path) + '.png')

def main():
    train()
    torch.save(model, args.model_path)

if __name__ == '__main__':
    main()
