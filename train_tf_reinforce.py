import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

import numpy as np

from kalah.kalahboard import KalahBoard
from kalah.kalahagentfactory import KalahAgentFactory
from kalah.kalahenv import KalahEnv

from kalah.agents.maxscorerepeatagent import MaxScoreRepeatAgent
from kalah.agents.tfagent import TfAgent

from kalah.kalahbattleground import KalahBattleground, KalahBattleGroundResults

bins = 4
seeds = 4

board = KalahBoard(bins, seeds)
agent_factory = KalahAgentFactory()
env = KalahEnv()

seed = 42

env.set_board(board)
env.set_agent_factory(agent_factory)
env.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

print(env.action_space.n)

n_inputs = env
n_outputs = env.action_space.n

model = keras.models.Sequential([
    keras.layers.Dense(256, activation='elu'),
    keras.layers.Dense(n_outputs, activation='softmax'),
])

eps = np.finfo(np.float32).eps.item()

def a_loss(prob, action, reward): 
    dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
    log_prob = dist.log_prob(action)
    loss = -log_prob*reward
    return loss

def play_one_step(env, state, model):
    probs = model(np.array([state]))
    action_probabilities = tfp.distributions.Categorical(probs=probs)
    action = action_probabilities.sample()
    next_obs, reward, done, _ = env.step(int(action.numpy()))

    return state, action, reward, probs, next_obs, done

def play_episode(env, n_max_steps, model):
    states = []
    actions = []
    rewards = []
    probs = []
    obs = env.reset()
    for _ in range(n_max_steps):
        state, action, reward, p, obs, done = play_one_step(env, obs, model)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        probs.append(p)
        if done:
            break
    return states, actions, rewards, probs

def play_multiple_epsisodes(env, n_episodes, n_max_steps, model):
    all_states = []
    all_actions = []
    all_rewards = []
    all_probs = []

    for _  in range(n_episodes):
        states = []
        actions = []
        rewards = []
        probs = []
        obs = env.reset()
        for _ in range(n_max_steps):
            state, action, reward, p, obs, done = play_one_step(env, obs, model)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            probs.append(p)
            if done:
                break
        all_states.append(states)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_probs.append(probs)

    return all_states, all_actions, all_rewards, all_grads

def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [ ( discounted_rewards - reward_mean ) / reward_std
           for discounted_rewards in all_discounted_rewards]

def normalize_rewards(rewards):
    return (rewards - rewards.mean()) / (rewards.std() + eps)

# Hyperparameters
n_iterations = 50000
n_episodes_per_update = 10
n_max_steps = 2000
discount_factor = 0.99

optimizer = keras.optimizers.Adam(lr=1e-5)

c_count = 0
def compare(model, seed=42):
    n_games = 1000
    battleground = KalahBattleground(bins, seeds)

    print("Running comparison...")

    global c_count
    c_count += 1

    results = battleground.battle(lambda seed: TfAgent(model, seed), lambda seed: MaxScoreRepeatAgent(seed), n_games, seed+c_count)

    win_percentage1 = 100*results.wins_agent1 / results.n_games
    if results.draws != n_games:
        print(TfAgent.__name__, "won", win_percentage1,
            "% of all N =", results.n_games ,"games against", MaxScoreRepeatAgent.__name__, "Number of draws:", results.draws)

    return [results.n_games, win_percentage1, results.draws]

def train(states, discounted_rewards, actions):
    for state, reward, action in zip(states, discounted_rewards, actions):
        with tf.GradientTape() as tape:
            p = model(np.array([state]), training=True)
            loss = a_loss(p, action, reward)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

for iteration in range(n_iterations):
    all_states, all_actions, all_rewards, all_grads = play_episode(env, n_max_steps, model)
    
    discounted_rewards = discount_rewards(all_rewards, discount_factor)
    normalized_discounted_rewards = normalize_rewards(discounted_rewards)

    total_rewards = sum(all_rewards)
    print("\rIteration: {}, mean rewards: {:.1f}".format(
        iteration, total_rewards), end="")
    train(all_states, normalized_discounted_rewards, all_actions)

print("\n")

compare(model)
