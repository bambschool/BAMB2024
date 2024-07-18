import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf


def load_data(path):
    "load csv data"
    df = pd.read_csv(path)
    return df

def probit_lapse_rates(x, beta, alpha, piL, piR):
    """
    Return probit with lapse rates.

    Parameters
    ----------
    x : float
        independent variable.
    beta : float
        sensitiviy.
    alpha : TYPE
        bias term.
    piL : float
        lapse rate for left side.
    piR : TYPE
        lapse rate for right side.

    Returns
    -------
    probit : float
        probit value for the given x, beta and alpha and lapse rates.

    """
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



    piL = 0
    piR = 0
    probit_lr = piR + (1 - piL - piR) * probit(x, beta, alpha)
    return probit_lr


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df = load_data(path='/home/molano/Downloads/CSH_ZAD_019.csv')
    # sort df by session start time
    df = df.sort_values(by=['session_start_time'])

    # get sessions
    sessions = df['session'].unique()
    # get number of sessions
    n_sessions = len(sessions)
    # get session start and end
    session_start = df['session_start_time'].unique()
    # get session numbers
    session_number = df['session_number'].values
    # get protocol
    protocol = df['task_protocol']
    # print first 1000 rows of session, session start time, session number, protocol
    print(df[['probabilityLeft', 'contrastRight', 'contrastLeft', 'response_times', 'feedbackType', 'choice', 'intervals_0', 'intervals_1', 'session_start_time']].head(20))
    # create a new column with the performance
    performance = np.zeros(len(df))
    performance[df['feedbackType']==1] = 1
    df['performance'] = performance
    # plot performance per session
    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    ax = ax.flatten()
    ax[0].plot(np.arange(n_sessions), df['performance'].groupby(df['session_start_time']).mean())
    ax[0].set_xlabel('Session')
    ax[0].set_ylabel('Mean performance')
    # plot reaction times histogram
    ax[1].hist(df['response_times']-df['stimOn_times'], bins=100)
    ax[1].set_xlabel('Reaction times')
    ax[1].set_ylabel('Counts')
    # plot psychometric curve
    # get the values of contrast right
    cr = df['contrastRight'].values
    cr[np.isnan(cr)] = 0
    # get the values of contrast left
    cl = df['contrastLeft'].values
    cl[np.isnan(cl)] = 0
    ev = cl - cr
    # get the choices
    choices = df['choice'].values
    choices[choices==-1] = 0
    # fit the psychometric curve using only the last 10000 trials
    ev = ev[-10000:]
    choices = choices[-10000:]
    x_fit = np.linspace(np.min(ev), np.max(ev), 20)
    popt, pcov = curve_fit(probit_lapse_rates, ev,
                            choices, maxfev=10000)
    y_fit = probit_lapse_rates(x_fit, popt[0], popt[1], popt[2], popt[3])
    # plot the psychometric curve
    ax[2].plot(x_fit, y_fit, 'k')
    ax[2].scatter(ev, choices, color='k', s=1)
    ax[2].set_xlabel('Contrast')
    ax[2].set_ylabel('Probability of choosing right')
    plt.show()

    
        
