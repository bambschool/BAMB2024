import numpy as np
from scipy import optimize
from abc import ABC, abstractmethod


class RLModel(ABC):
    @abstractmethod
    def simulate(self, T, mu):
        pass

    @abstractmethod
    def likelihood(self, pars, a, r):
        pass

    def fit(self, a, r):
        x0 = self.initial_parameters()
        bounds = self.parameter_bounds()
        
        res = optimize.minimize(self.likelihood, args=(a, r), method='L-BFGS-B', x0=x0, bounds=bounds)
        
        bic = len(x0) * np.log(len(a)) + 2 * res.fun
        return bic, res.x, -res.fun  # BIC, parameters, negative log-likelihood

    @abstractmethod
    def initial_parameters(self):
        pass

    @abstractmethod
    def parameter_bounds(self):
        pass


class WinStayLoseSwitch(RLModel):
    def simulate(self, T, mu, epsilon):
        a = [np.random.choice(2)]
        r = [(np.random.random() < mu[a[0]]).astype(float)]

        for _ in range(T - 1):
            if r[-1] == 1:  # win-stay
                p = [epsilon / 2, epsilon / 2]
                p[a[-1]] = 1 - epsilon / 2
            else:  # lose-shift
                p = [1 - epsilon / 2, 1 - epsilon / 2]
                p[a[-1]] = epsilon / 2

            a.append(np.random.choice(2, p=p))
            r.append((np.random.random() < mu[a[-1]]).astype(float))

        return np.array(a), np.array(r)

    def likelihood(self, pars, a, r):
        epsilon = pars[0]
        choice_p = [0.5]

        for t in range(1, len(a)):
            if r[t-1] == 1:
                p = [epsilon / 2, epsilon / 2]
                p[a[t-1]] = 1 - epsilon / 2
            else:
                p = [1 - epsilon / 2, 1 - epsilon / 2]
                p[a[t-1]] = epsilon / 2

            choice_p.append(p[a[t]])

        return -np.sum(np.log(np.array(choice_p) + 1e-5))

    def initial_parameters(self):
        return [np.random.random()]

    def parameter_bounds(self):
        return [(0, 1)]


class RescorlaWagner(RLModel):
    def simulate(self, T, mu, alpha, beta):
        Q = np.array([0.5, 0.5])
        a, r = [], []

        for _ in range(T):
            p = np.exp(beta * Q) / np.sum(np.exp(beta * Q))
            a.append(np.random.choice(2, p=p))
            r.append((np.random.random() < mu[a[-1]]).astype(float))

            delta = r[-1] - Q[a[-1]]
            Q[a[-1]] += alpha * delta

        return np.array(a), np.array(r)

    def likelihood(self, pars, a, r):
        alpha, beta = pars
        Q = np.array([0.5, 0.5])
        choice_p = []

        for t in range(len(a)):
            p = np.exp(beta * Q) / np.sum(np.exp(beta * Q))
            choice_p.append(p[a[t]])

            delta = r[t] - Q[a[t]]
            Q[a[t]] += alpha * delta

        return -np.sum(np.log(np.array(choice_p) + 1e-5))

    def initial_parameters(self):
        return [np.random.random(), np.random.exponential()]

    def parameter_bounds(self):
        return [(0, 1), (0, np.inf)]


class ChoiceKernel(RLModel):
    def simulate(self, T, mu, alpha, beta):
        CK = np.full(2, 0.001)
        a, r = [], []

        for _ in range(T):
            p = np.exp(beta * CK) / np.sum(np.exp(beta * CK))
            a.append(np.random.choice(2, p=p))
            r.append((np.random.random() < mu[a[-1]]).astype(float))

            CK *= (1 - alpha)
            CK[a[-1]] += alpha

        return np.array(a), np.array(r)

    def likelihood(self, pars, a, r):
        alpha, beta = pars
        CK = np.full(2, 0.001)
        choice_p = []

        for t in range(len(a)):
            p = np.exp(beta * CK) / np.sum(np.exp(beta * CK))
            choice_p.append(p[a[t]])

            CK *= (1 - alpha)
            CK[a[t]] += alpha

        return -np.sum(np.log(np.array(choice_p) + 1e-5))

    def initial_parameters(self):
        return [np.random.random(), 0.5 + np.random.exponential()]

    def parameter_bounds(self):
        return [(0, 1), (0, np.inf)]


# Usage example:
# wsls = WinStayLoseShift()
# rw = RescorlaWagner()
# ck = ChoiceKernel()

# a, r = wsls.simulate(100, [0.7, 0.3], 0.1)
# bic, params, nll = wsls.fit(a, r)
