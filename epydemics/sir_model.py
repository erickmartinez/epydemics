import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult
from typing import List, Union


# The SIR model differential equations.
def sir(t: float, y: List[float, float, float], N: Union[int, float], beta: float, gamma: float):
    """
    System of ODE for the Susceptible-Infected-Recovered model without vital kinetics
    Parameters
    ----------
    t: float
        The time in days
    y: List[float, float, float]
    N: int
        The total population
    beta: float
        The contact rate of the disease (1/days)
    gamma: float
        The recovery rate of the disease (1/days)

    Returns
    -------
    List[float, float, flaot]
        dS/dt, dI/dt, dR/dt
    """
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
        The contact rate of the disease (1/days)
    gamma: float
        The recovery rate of the disease (1/days)
    kwargs: keyword arguments
        I0: int
            The initial number of infected individuals
        R0: int
            The initial number of recovered individuals

    Returns
    -------
    OptimizeResult:
        The solution
    """

    I0 = kwargs.get('I0', 1)
    R0 = kwargs.get('R0', 0)

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
    # ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    sol = solve_ivp(sir, [np.amin(t), np.amax(t)], y0, t_eval=t,
                    args=(N, beta, gamma),
                    method='DOP853',
                    dense_output=True)

    return sol
