import numpy as np
from scipy.integrate import solve_ivp

# The SIR model differential equations.    
def sir(t, y, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def sir_model(t: np.ndarray, N: int, beta: float, gamma: float, **kwargs):
    """
    Solves the Susceptible-Infected-Removed ODE
    
    Parameters
    ----------
    t: np.ndarray
        The time (days)
    N: int
        The total population
    beta: float
        The contact rate beta (1/days)
    gamma: float
        The mean recovery rate (1/days)
    kwargs: keyword arguments
        I0: int
            The initial number of infected individuals
        R0: int
            The initial number of recovered individuals
    """

    I0 = kwargs.get('I0',1)
    R0 = kwargs.get('R0',0)

#    # Total population, N.
#    N = 1000
#    # Initial number of infected and recovered individuals, I0 and R0.
#    I0, R0 = 1, 0
#    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
#    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
#    beta, gamma = 0.2, 1./10 
#    # A grid of time points (in days)

    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    #ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    sol = solve_ivp(sir, [np.amin(t), np.amax(t)], y0, t_eval=t,
                    args=(N, beta, gamma), 
                    method='DOP853',
                    dense_output=True)
    
    
    #S, I, R = ret.T
#    y = sol.sol(t)
#    S, I, R = y
    return sol

