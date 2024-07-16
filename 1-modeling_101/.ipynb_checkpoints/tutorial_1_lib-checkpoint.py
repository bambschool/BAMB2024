
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.optimize as optimize


# Define a function that simulates model A
def simulate_modelA(s, c, sigma_d=1, sigma_m=1.5, T=0):
    # T is the decision threshold
    n = len(s)
    sigma = np.where(c=='control',sigma_d,sigma_m)
    s_hat = s + np.random.normal(scale=sigma, size=n)  # Noisy stimulus
    r = 1 * (s_hat > T)  # Response based on stimulus (SDT)
    return r


# Define a function that simulates model B
def simulate_modelB(s, c, sigma_d=1, Lambda=0.2, T=0, q=0.5):
    # T is the decision threshold
    # q is the probability of rightward response if lapse
    n = len(s)
    s_hat = s + np.random.normal(scale=sigma_d, size=n)  # Noisy stimulus
    r_stim = 1 * (s_hat > T)  # Response based on stimulus (SDT)
    isLapse = (c == 'manipulation') & (np.random.rand(n) < Lambda)  # Generate boolean array (True if lapse; only in Manipulation trials)
    r_lapse = np.random.choice(a=[0, 1], p=[q, 1 - q], size=n)
    r = np.where(isLapse,r_lapse,r_stim)
    return r


# Define a function that simulates model C (this is a combination of model A and B and is only used for the parameter recovery exercise)
def simulate_modelC(s, c, sigma_d=1, sigma_m=1, Lambda=0.2, T=0, q=0.5):
    # T is the decision threshold
    # q is the probability of rightward response if lapse
    n = len(s)
    sigma = np.where(c=='control',sigma_d,sigma_m)
    s_hat = s + np.random.normal(scale=sigma, size=n)  # Noisy stimulus
    r_stim = 1 * (s_hat > T)  # Response based on stimulus (SDT)
    isLapse = (c == 'manipulation') & (np.random.rand(n) < Lambda)  # Generate boolean array (True if lapse; only in Manipulation trials)
    r_lapse = np.random.choice(a=[0, 1], p=[q, 1 - q], size=n)
    r = np.where(isLapse,r_lapse,r_stim)
    return r


def p_modelA(s,c, sigma_d, sigma_m):
    # model A has a change in sigma
    
    # probability of rightward choice in control and manipulated condition
    p = np.where(c=='control', norm.cdf(s/sigma_d), norm.cdf(s/sigma_m) )
    return p


def p_modelB(s,c, sigma_d, Lambda):
    # model B has a change in lapse rate
    
    # probability of rightward choice in control and manipulated condition
    p = np.where(c=='control', norm.cdf(s/sigma_d), (1-Lambda)*norm.cdf(s/sigma_d) + Lambda/2 )
    return p


def p_modelC(s,c, sigma_d, sigma_m, Lambda):
    # model C has a change in sigma and in lapse rate
    
    # probability of rightward choice in control and manipulated condition
    p = np.where(c=='control', norm.cdf(s/sigma_d), (1-Lambda)*norm.cdf(s/sigma_m) + Lambda/2 )
    return p


# Likelihood function for model A
def LogLikelihood_modelA(df, sigma_d, sigma_m):
    
    # p of rightward choice
    p = p_modelA(df.stimulus,df.condition, sigma_d, sigma_m)
    
    lh = np.where(df.response==1, p, 1-p)
    
    # Handle division by zero
    lh = np.where(lh == 0, np.finfo(float).eps, lh)
    
    LLH = np.sum(np.log(lh))
    
    return LLH


# Likelihood function for model B
def LogLikelihood_modelB(df, sigma_d, Lambda):

    # p of rightward choice
    p = p_modelB(df.stimulus,df.condition, sigma_d, Lambda)
    
    lh = np.where(df.response==1, p, 1-p)
    
    # Handle division by zero
    lh = np.where(lh == 0, np.finfo(float).eps, lh)
    
    LLH = np.sum(np.log(lh))
    
    return LLH


# Likelihood function for model C (only used in the parameter recovery exercise)
def LogLikelihood_modelC(df, sigma_d, sigma_m, Lambda):

    # p of rightward choice
    p = p_modelC(df.stimulus,df.condition, sigma_d, sigma_m, Lambda)
    
    lh = np.where(df.response==1, p, 1-p)
    
    # Handle division by zero
    lh = np.where(lh == 0, np.finfo(float).eps, lh)
    
    LLH = np.sum(np.log(lh))
    
    return LLH


# Plot psychometric curve for hte 'control' and 'manipulation' conditions
def plot_data_psychometric(df):
    mask = df.condition=='control'
    df[mask].groupby('stimulus').response.agg(('mean','sem')).plot(yerr='sem', fmt = 'bo', ax=plt.gca(), legend=False);

    mask = df.condition=='manipulation'
    df[mask].groupby('stimulus').response.agg(('mean','sem')).plot(yerr='sem', fmt = 'ro', ax=plt.gca(), legend=False);


# Plot the model fit
def plot_model(p_model):
    # define 100 angle points linearly spaced between -3 and 3
    s_linspace = np.linspace(-3,3,100)
    
    p_standard = p_model(s_linspace,'control')
    p_manipulation = p_model(s_linspace,'manipulation')
    
    # plot fitted psychometric functions
    plt.plot(s_linspace, p_standard, 'b', label="control")
    plt.plot(s_linspace, p_manipulation, 'r', label="manipulation")
    
    plt.ylabel('p(rightward)')

