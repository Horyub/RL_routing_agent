# Deep Reinforcement Learning for Network Routing

This repository contains work developed for the **Machine and Deep Learning (Advanced)** exam at the **University of Calabria (UniCal)**.

## Project objective

Train a Deep Q-Network (DQN) to route traffic demands hop by hop while respecting link capacities, keeping routes short, and limiting peak link utilization.

## What I developed

I built a custom Gymnasium environment for a 12-node Abilene-style network and adapted a DQN agent to select feasible next hops using action masking. I compared per-episode and per-hop updates and training with and without a curriculum. I evaluated the saved policies on the same 200 synthetic traffic matrices.

## Repository structure

- **Toy model** — a simplified exercise used to practice building and interacting with a Gymnasium environment.
- **[Main project](project/)** — the routing environment, DQN agent, and training and evaluation code.

| File in `project/` | Purpose |
| --- | --- |
| [`README.md`](project/README.md) | Setup, training, and evaluation instructions |
| [`routing_env.py`](project/routing_env.py) | Gymnasium environment that simulates the network and traffic demands |
| [`routing_agent.py`](project/routing_agent.py) | Agent entry point for training and evaluating saved models |
| [`dqn.py`](project/dqn.py) | Neural network that estimates next-hop action values |
| [`experience_replay.py`](project/experience_replay.py) | Buffer that stores and samples training transitions |
| [`hyperparameters.yml`](project/hyperparameters.yml) | Training configuration |
| [`requirements.txt`](project/requirements.txt) | Python dependencies |
