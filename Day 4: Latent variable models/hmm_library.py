## HMM library: functions for expectation mazimization in hidden Markov models

# author:       Heike Stein 
# last mod:     03 Jul 23

# code based on C. Bishop, "Pattern Recognition and Machine Learning", 2006
# code partly adapted from https://github.com/lindermanlab/ssm

import numpy as np
import scipy.special as scsp


##########################################################################################
#                                            E step                                      #
##########################################################################################


def forward(p, A, log_likes):

    N = len(log_likes.T)  # number of trials

    # we calculate log alphas instead of alphas, which is computationally more stable

    # first alpha: initial state probabilities * likelihoods of each state
    # alpha_0 = p(z_0) * p(x_0 | z_0)                 # standard equation
    # log_alpha_0 = log(p(z_0)) + log(p(x_0 | z_0))   # log transformed

    alphas = [np.log(p) + log_likes[:,0]]

    for n in range(N-1):

        # all other alphas are calculated as: likelihoods * (previous alpha @ transition probabilities)
        # alpha_n = p(x_n|z_n) * sum_k( alpha_(n-1) * p(z_n | z_(n-1)) )                # standard equation
        # log_alpha_n = log(p(x_n|z_n)) + log(sum_k( alpha_(n-1) * p(z_n | z_(n-1)) ))  # log transformed

        # we first calculate the dot product on the right: sum_k( alpha_(n-1) * p(z_n | z_(n-1)) )
        norm_dot_prod = np.dot(np.exp(alphas[n] - np.max(alphas[n])), A)
        # since alphas become smaller at each iteration, 
        # we normalize by the maximum value before taking the exponent

        # log-transform and multiply by the maximum value to "undo" the normalization
        log_dot_prod = np.log(norm_dot_prod) + np.max(alphas[n])

        # calculate new alphas by summing the log likelihoods and append to list
        alphas.append(log_dot_prod + log_likes[:,n+1])

    return np.array(alphas)

##########################################################################################


def backward(A, log_likes):
    
    N = len(log_likes.T)  # number of trials
    K = len(log_likes)    # number of states

    # as in the forward pass, we calculate log betas instead of betas

    # first beta: initialize beta_N = 1 --> becomes log_beta_N = 0
    betas = [np.zeros(K)]

    # we will fill the list in forward order and then invert
    for n in range(N-1):
        
        # all other betas are calculated as: (future beta * likelihoods) @ transition probabilities)
        # beta_n = ( beta_(n+1) * p(x_{n+1}|z_{n+1}) ) @ p(z_{n+1} | z_(n))                          # standard equation

        # we first calculate the element-wise product beta_(n+1) * p(x_{n+1}|z_{n+1})
        log_prod = log_likes[:,-(n+1)] + betas[n]
        
        # again, we normalize log_prod by its maximum before calling exp(log_prod), 
        # because betas become smaller and smaller through iterations
        norm_log_prod = log_prod - np.max(log_prod)
        
        # we then calculate the dot product with the transition matrix and 
        # undo normalization outside the exp()
        betas.append(np.log(np.dot(A, np.exp(norm_log_prod))) + np.max(log_prod))
        
    return np.array(betas)[::-1] 

##########################################################################################


def expected_states(log_alphas, log_betas):

    # upper: alpha * beta --> becomes log_alpha + log_beta
    upper = log_alphas + log_betas
    
    # subtract log of the sum over states of log_alpha + log_beta
    log_gammas = upper - scsp.logsumexp(upper, axis=1, keepdims=True)
    
    # transform back by taking exp
    gammas = np.exp(log_gammas)
    
    return gammas

##########################################################################################


def expected_transitions(alphas, betas, log_likes, As):
    
    # calculate alpha * LL * transition probabilities * betas
    # sum due to log transform
    upper = alphas[:,:,np.newaxis] + betas[:,np.newaxis,:] \
            + log_likes.T[:,np.newaxis,:] + np.log(As)[np.newaxis,:,:]

    # normalize by max. val before taking the exp
    upper -= upper.max((1,2))[:, None, None]
    expected_joints = np.exp(upper)
    
    # divide by total likelihood, which is computed by marginalizing over z_n and also z_n-1
    expected_joints /= expected_joints.sum((1,2))[:,None,None]
    
    return expected_joints

##########################################################################################


def HMM_estep(p, As, log_likes):
    
    # forward pass
    alphas = forward(p, A, log_likes)
    
    # backward pass
    betas = backward(A, log_likes)
    
    # calculate posterior: expected states
    gammas = expected_states(alphas, betas)
    
    # calculate posterior: expected transitions
    xis = expected_transitions(alphas, betas, log_likes, As)
    
    # calculate log likelihood
    log_LL = np.sum(alphas[-1] + betas[-1])
    
    return gammas, xis, log_LL


##########################################################################################
#                                            M step                                      #
##########################################################################################


def HMM_mstep(data, gammas, xis):
    K = gammas.shape[1]
    N = gammas.shape[0]
    D = data.shape[-1]
    
    # class-specific normalization constant
    Nk = np.sum(gammas, axis=0)
    
    # mu
    m = [np.sum(gammas[:,k][:,np.newaxis]*data, axis=0)/Nk[k]
         for k in range(K)]
    
    # Sigma
    s = np.zeros([K,D,D]) * np.nan
    for k in range(K):
        summation = 0
        for n in range(N):
            summation += gammas[n,k] * (data[n]-m[k])[:,np.newaxis] @ (data[n]-m[k])[:,np.newaxis].T
        s[k,:,:] = summation/Nk[k]
    
    # pi
    p = gammas[0]/np.sum(gammas[0])
    
    # A
    A = np.sum(xis[1:], axis=0) / np.sum(np.sum(xis[1:], axis=0), axis=0)
        
    return m, s, p, A

##########################################################################################
#                                     Viterbi algorithm                                  #
##########################################################################################


def log_viterbi(log_likes, A, p):
    K = log_likes.shape[0]
    T = log_likes.shape[1]
    
    ### Viterbi algorithm
    deltas = np.zeros([K, T]) * np.nan
    iotas  = np.zeros([K, T]) * np.nan

    # first sample: we calculate the un-normalized log posterior over states: log prior + logLLs
    deltas[:,0] = np.log(p) + log_likes[:,0]

    for t in range(1,T):
        for j in range(K):
            # for each sample and state: 
            deltas[j,t] = np.nanmax(deltas[:,t-1] + log_likes[j,t] + np.log(A[:,j]))
            iotas[j,t]  = np.nanargmax(deltas[:,t-1] + log_likes[j,t] + np.log(A[:,j]))

    # final state and backtracing
    sequence  = np.zeros(T) * np.nan
    sequence[-1] = np.argmax(deltas[:,-1])
    
    for t in range(1,T):
        sequence[-t-1] = iotas[int(sequence[-t]), -t]

    return sequence.astype('int')