import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

import numpy as np

import gym

seed = 42

env = gym.make("CartPole-v1")
env.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

print(env.action_space.n)

n_inputs = env
n_outputs = env.action_space.n

model = keras.models.Sequential([
    keras.layers.Dense(16, activation='elu'),
    keras.layers.Dense(16, activation='elu'),
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
n_iterations = 400
n_episodes_per_update = 10
n_max_steps = 400
discount_factor = 0.95

optimizer = keras.optimizers.Adam(lr=1e-4)

def render_policy_net(model, n_max_steps=200, seed=42):
    frames = []
    env = gym.make("CartPole-v1")
    env.seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    for step in range(n_max_steps):
        frames.append(env.render(mode="rgb_array"))
        probs = model.predict(obs.reshape(1, -1))

        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = int(action_probabilities.sample().numpy())

        obs, reward, done, info = env.step(action)
        if done:
            print("\nFell after steps = {}".format(step + 1))
            break
    env.close()
    return frames

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
    # if total_rewards >= 200:
    #     break

print("\n")

render_policy_net(model)
