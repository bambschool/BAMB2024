# Basics of Reinforcement Learning

Welcome to the first part of our interactive Reinforcement Learning (RL) tutorial. This tutorial is designed to provide an introduction to RL concepts by hands-on learning and experimentation.

You will be working with the [`tutorial_2a.ipynb`](./tutorial_2a.ipynb) - the jupyter notebook with all instructions and code. Although you don't need to know and understand it, feel free to look at the helper code present in [`agents.py`](./agents.py), [`environments.py`](./environments.py), [`simulate.py`](./simulate.py), and [`plotting.py`](./plotting.py).

## Overview

In this tutorial, we are going to explore two key RL concepts: the environment and the agent. We will be using Python and the Farama Gymnasium library, which provides a variety of pre-made environments and a standardized interface, making it easy to develop and compare RL algorithms as well as build your own environments.

We will start with a simple gridworld-like environment, `FrozenLake-v1`, and two agents: a `RandomAgent` and a `QLearningAgent`. By following the standardized approach, we will see how our agents naturally extend to another classic control environment, `CartPole-v1`.

- **Exploring the Environment:** 
  - We will initialize the environment and explore its observation and action spaces.
  - Understanding these is key to successfully training an RL agent.
- **Training a Random Agent:** 
  - We will start with a `RandomAgent`, which chooses actions randomly from the action space. 
  - This will serve as a baseline and help us understand the environment dynamics.
- **Training a Q-Learning Agent:** 
  - Next, we will train a `QLearningAgent`, which uses the Q-Learning algorithm to learn from its experiences and improve its policy over time.

The interactive nature of this tutorial allows you to see the RL process in action and understand the roles of the environment and the agent, as well as the interaction between them.

## Setup Instructions

All the setup instructions including environment setup, dependencies, and how to run the code are provided in the [parent folder's README](../README.md). Please refer to that before you begin with this tutorial.

Enjoy the journey of learning Reinforcement Learning!
