import numpy as np
import scipy as sp

##### Functions for simulation

def simulate_M2WSLS_v1(T, mu, epsilon):
    # mu: true reward probabilities
    # epsilon: level of randomness

    # first trial: choose randomly
    a = [np.random.choice(2, 1, p=[0.5, 0.5])[0]] # choice acc. to choice probababilities [b, 1-b]
    r = [(np.random.randint(0,2) < mu[a[0]]).astype('float')]    # reward based on choice

    for t in range(T-1):
        # choice depends on last reward
        if r[t]==1:  # win stay (with probability 1-epsilon)
            p = epsilon / 2*np.ones(2)
            p[a[t]] = 1-epsilon/2
        elif r[t]==0: # lose shift (with probability 1-epsilon)
            p = (1-epsilon/2) * np.ones(2)
            p[a[t]] = epsilon/2

        a.append(np.random.choice(2, 1, p=p)[0])  # choice acc. to choice probababilities
        r.append((np.random.uniform() < mu[a[-1]]).astype('float'))     # reward based on choice

    return np.squeeze(a), np.squeeze(r)


def simulate_M3RescorlaWagner_v1(T, mu, alpha, beta):
    # mu: true reward probabilities
    # alpha: learning rate
    # beta: softmax inverse temperature

    # initialise Q values for each of the two options
    Q = np.array([0.5, 0.5])

    a = []; r=[]
    # loop over trials
    for t in range(T):
        # compute choice probabilities using softmax formula
        p = np.exp(beta*Q) / np.sum(np.exp(beta*Q)) # will be [.5,.5] for Q=[.5, .5]

        a.append(np.random.choice(2, 1, p=p)[0])                          # choice
        r.append((np.random.uniform() < mu[a[-1]]).astype('float'))     # reward based on choice

        # update values using prediction error term
        delta = r[-1] - Q[a[-1]]
        Q[a[-1]] = Q[a[-1]] + alpha*delta

    return np.squeeze(a), np.squeeze(r)


def simulate_M4ChoiceKernel_v1(T, mu, alpha, beta):
    # mu: true reward probabilities
    # alpha: choice-kernel learning rate
    # beta: choice-kernel inverse temperature

    CK = np.zeros(2)+.001 # initialize at small value
    a = []
    r = []

    for t in range(T):
        p = np.exp(beta*CK) / sum(np.exp(beta*CK)) # compute choice probabilities

        a.append(np.random.choice(2, 1, p=p)[0])  # choice acc. to choice probababilities
        r.append((np.random.uniform() < mu[a[-1]]).astype('float'))     # reward based on choice

        # update choice kernel
        CK = (1-alpha) * CK
        CK[a[t]] = CK[a[t]] + alpha

    return np.squeeze(a), np.squeeze(r)



######## Likelihood functions

def lik_M2WSLS_v1(pars, a, r):

    epsilon = pars[0]
    choiceP = [.5]

    for t in range(1,len(a)):
        if r[t-1]==1:                     # win stay (with probability 1-epsilon)
            p = epsilon / 2*np.ones(2)
            p[a[t-1]] = 1-epsilon/2
        elif r[t-1]==0:                   # lose shift (with probability 1-epsilon)
            p = (1-epsilon/2) * np.ones(2)
            p[a[t-1]] = epsilon / 2

        choiceP.append(p[a[t]])

    return -np.sum(np.log(np.array(choiceP)+10**-5))


def lik_M3RescorlaWagner_v1(pars, a, r):

    alpha, beta = pars

    Q = np.array([.5,.5])
    choiceP = []

    for t in range(len(a)):

        # compute choice probabilities
        p = np.exp(beta*Q) / np.sum(np.exp(beta*Q))

        choiceP.append(p[a[t]])

        # update values
        delta = r[t] - Q[a[t]]
        Q[a[t]] = Q[a[t]] + alpha * delta

    # return negative log-likelihood
    return - np.sum(np.log(np.array(choiceP)+10**-5))



def lik_M4ChoiceKernel_v1(pars, a, r):

    alpha, beta = pars

    CK = np.zeros(2) + 0.001 # initialize at small value
    choiceP = []

    for t in range(len(a)):

        p = np.exp(beta*CK) / sum(np.exp(beta*CK)) # compute choice probabilities

        # update choice kernel
        CK = (1-alpha) * CK
        CK[a[t]] = CK[a[t]] + alpha

        choiceP.append(p[a[t]])

    # return negative log-likelihood
    return - np.sum(np.log(np.array(choiceP)+10**-5))


####### Fitting

def fit_M2WSLS_v1(a, r):

    x0 = [np.random.uniform()]
    bounds = [(0,1)]

    res = sp.optimize.minimize(lik_M2WSLS_v1, args = (a, r), method='L-BFGS-B', x0=x0, bounds=bounds)

    return len(x0)* np.log(len(a)) + 2*res.fun, res.x, -res.fun # BIC, pars, nLL


def fit_M3RescorlaWagner_v1(a, r):

    x0 = [np.random.uniform(), np.random.exponential()]
    bounds = [(0,1), (0,np.inf)]

    res = sp.optimize.minimize(lik_M3RescorlaWagner_v1, args = (a, r), method='L-BFGS-B', x0=x0, bounds=bounds)

    return len(x0)* np.log(len(a)) + 2*res.fun, res.x, -res.fun # BIC, pars, nLL


def fit_M4ChoiceKernel_v1(a, r):

    x0 = [np.random.uniform(), .5+np.random.exponential()]
    bounds = [(0,1), (0,np.inf)]

    res = sp.optimize.minimize(lik_M4ChoiceKernel_v1, args=(a, r), method='L-BFGS-B', x0=x0, bounds=bounds)

    return len(x0)* np.log(len(a)) + 2*res.fun, res.x, -res.fun # BIC, pars, nLL
