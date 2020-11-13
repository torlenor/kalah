# Kalah

[![Build and Test](https://github.com/torlenor/kalah/workflows/Build%20and%20Test/badge.svg?branch=master)](https://github.com/torlenor/kalah/actions?query=workflow%3A%22Build+and+Test%22)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

## Description

An implementation of the game Kalah and some agents in Python. It is used to play around with reinforcement learning and currently REINFORCE and actor-critic models are provided in addition to classic agents.

## Requirements

- Python 3.6–3.8
- Numpy
- PyTorch

See *the requirements.txt* file for detailed Python requirements or use
```bash
pip install -r requirements.txt --user
```
to install them.

## Agents

You can use the script *compare_classic_agents.py* to compare the classic agents with each other.

### Random Agent

It randomly chooses from the allowed moves.

### Max Score Agent

It tries to maximze the score from each move.

### Max Score Repeat Agent

In Kalah if you hit your own house with the last seed, you can go again, this agent knows that. It tries to maximize the number moves per turn or if that it not possible maximize the score.

### Minimax Agent

This agent implements the [Minimax](https://en.wikipedia.org/wiki/Minimax) algorithm for Kalah. It uses simply the score of each player as the value of the leaf nodes. One could think of some better values, but for now it is acceptable.

### ReinforceAgent

This is a wrapper around a PyTorch reinforce model for Kalah.

### ActorCriticAgent

This is a wrapper around a PyTorch actor-critic model for Kalah.

## Reinforcement learning

We provide a few scripts and models for reinforcement learning.

### ActorCritic

A good way to start playing with the actor-critic model is using the following start parameters to train it:

```bash
python train_aa.py --bins 4 --seeds 4 --evaluation-interval 10000 --episodes 500000 --gamma 0.99 --seed 1 --solved 99 --learning-rate 0.001 --neurons 256 --evaluation-games 100 --run-id aa_256neurons
```

Example output (last two lines, may not be the same anymore, due to model/code changes):

```bash
Comparing @ Episode 61000: ActorCriticAgent won 88.77551020408163 % of all N = 100 games against MaxScoreRepeatAgent Number of draws: 2
Solved after 61000 episodes! The last win percentage was 88.775510
```

The model can then be compared to classic agents with the command

```bash
python validate_aa.py --bins 4 --seeds 4 --validation-games 1000 --model-path ./results/aa_256neurons/final_model.pt
```

Example output (may not be the same anymore, due to model/code changes):

```bash
Comparisons with other agent:
ActorCriticAgent won 87.8 % ( n = 878 ) of all N = 1000 games against RandomAgent Number of draws: 31
ActorCriticAgent won 91.0 % ( n = 910 ) of all N = 1000 games against MaxScoreAgent Number of draws: 57
ActorCriticAgent won 82.7 % ( n = 827 ) of all N = 1000 games against MaxScoreRepeatAgent Number of draws: 34
ActorCriticAgent won 53.5 % ( n = 535 ) of all N = 1000 games against MinimaxAgent Number of draws: 119
```

### REINFORCE (broken at the moment)

A good way to start playing with the REINFORCE model is using the following start parameters to train it:

```bash
python train_reinforce.py --bins 4 --seeds 4 --log-interval 500 --episodes 200000 --gamma 0.99 --seed 3 --solved 70 --learning-rate 0.001  --neurons 256 --drop-out 0.1 --model-path ./reinforce_model.pt
```

With these parameters, it should converge, i.e., reach the desired win rate (which is averaged over the last two evaluations), after 156500 played games.

Example output (last two lines, may not be the same anymore, due to model/code changes):

```bash
Comparing @ Episode 156500: ReinforceAgent won 69.87951807228916 % of all N = 100 games against MaxScoreRepeatAgent Number of draws: 17
Solved after 156500 episodes! The last win percentage was 69.879518
```

The model can then be compared to classic agents with the command

```bash
python validate_reinforce.py --bins 4 --seeds 4 --validation-games 1000 --model-path ./reinforce_model.pt
```

Example output (may not be the same anymore, due to model/code changes):

```bash
Comparisons with other agent:
ReinforceAgent won 68.3 % ( n = 683 ) of all N = 1000 games against RandomAgent Number of draws: 45
ReinforceAgent won 60.9 % ( n = 609 ) of all N = 1000 games against MaxScoreAgent Number of draws: 136
ReinforceAgent won 44.7 % ( n = 447 ) of all N = 1000 games against MaxScoreRepeatAgent Number of draws: 137
ReinforceAgent won 44.2 % ( n = 442 ) of all N = 1000 games against MinimaxAgent Number of draws: 98
```

In contrast to the actor-critic model, the REINFORCE model is harder to train and you may have to play with the parameters until you find some which do reach high win rates. Sadly even the random seed used for training matters a lot. As you can see above, these win rates are not that great. Playing around more with the parameters can lead, however, to much better win rates. A better metric for judging when the model is trained may also be useful.
