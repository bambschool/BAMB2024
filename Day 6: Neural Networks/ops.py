
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.special import erf

def get_modelpath(envid):
    """
    to create a path to save the data from the network that we will analize.
    """
    # Make a local file directories
    path = Path('.') / 'files'
    os.makedirs(path, exist_ok=True)
    path = path / envid
    os.makedirs(path, exist_ok=True)
    
    return path

def plot_trials_timestep(inputs, actions, gt, perf, env_kwargs, num_steps=None, reward=None):
    inputs = np.array(inputs)
    actions = np.array(actions)
    gt = np.array(gt)
    perf = np.array(perf)
    # Plot first 40 steps
    if num_steps is None:
        num_steps = len(gt)
    f, ax = plt.subplots(ncols=1, nrows=3, figsize=(8, 4), dpi=150, sharex=True)

    ax[0].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], inputs[:num_steps, 0], label='Fixation')
    ax[0].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], inputs[:num_steps, 1], label='Stim. L.')
    ax[0].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], inputs[:num_steps, 2], label='Stim. R.')
    ax[0].set_ylabel('Inputs')
    ax[0].legend()
    ax[1].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], gt[:num_steps], label='Targets', color='k')
    ax[1].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], actions[:num_steps], label='Choice', linestyle='--')
    ax[1].set_ylabel('Actions / Targets')
    ax[1].legend()
    ax[2].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], perf[:num_steps], label='Accuracy')
    if reward is not None:
        reward = np.array(reward)   
        ax[2].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], reward[:num_steps], label='reward')
    ax[2].set_ylabel('Performance')
    ax[1].legend()
    ax[2].set_xlabel('Time (ms)')


def probit(x, beta, alpha):
    """
    Return probit function with parameters alpha and beta.

    Parameters
    ----------
    x : float
        independent variable.
    beta : float
        sensitiviy.
    alpha : TYPE
        bias term.

    Returns
    -------
    probit : float
        probit value for the given x, beta and alpha.

    """
    probit = 1/2*(1+erf((beta*x+alpha)/np.sqrt(2)))
    return probit

class efficieNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(efficieNet, self).__init__()

        # build a recurrent neural network with a single recurrent layer and rectified linear units
        self.vanilla = nn.RNN(input_size, hidden_size, nonlinearity='relu')
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # get the output of the network for a given input
        out, _ = self.vanilla(x)
        x = self.linear(out)
        return x, out
    
