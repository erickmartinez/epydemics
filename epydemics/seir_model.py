from typing import List, Union
from scipy.optimize import OptimizeResult
import numpy as np
from scipy.integrate import solve_ivp


# https://www.idmod.org/docs/hiv/model-seir.html
# https://www.idmod.org/docs/typhoid/model-seir.html
# The SIR model differential equations. (without vital kinetics)
def seir(t: float, y: List[float], N: Union[int, float], beta: float, gamma: float, sigma: float):
    """
    System of ODE for the Susceptible-Exposed-Infected-Recovered model without vital kinetics

    Parameters
    ----------
    t: float
        Time in days
    y: List[float, float, float, float]
        A list with the current values of
        S: float
            The susceptible population
        E: float
            The exposed population
        I: float
            The infected population
        R: float
            The removed population
    N: int
        The total population
    beta: float
        The contact rate of the disease (1/days)
    gamma: float
        The recovery rate of the disease (1/days)
    sigma: float
        The rate of latent individuals becoming infectious in 1/days (average duration of incubation is 1/sigma)

    Returns
    -------
    OptimizeResult:
        The solution
    """
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


def seir_model(t: np.ndarray, N: Union[int, float], beta: float, gamma: float, sigma: float,
               **kwargs):
    """
    Solves the Susceptible-Exposed-Infected-Removed ODE
    
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
    sigma: float
        The rate of latent individuals becoming infectious in 1/days (average duration of incubation is 1/sigma)
    kwargs: keyword arguments
        I0: int
            The initial number of infected individuals
        R0: int
            The initial number of recovered individuals
        E0: float
            The initial number of exposed individuals
    """

    I0 = kwargs.get('I0', 1)
    R0 = kwargs.get('R0', 0)
    E0 = kwargs.get('E0', 1)

    #    # Total population, N.
    #    N = 1000
    #    # Initial number of infected and recovered individuals, I0 and R0.
    #    I0, R0 = 1, 0
    #    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0 - E0
    #    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    #    beta, gamma = 0.2, 1./10
    #    # A grid of time points (in days)

    # Initial conditions vector
    y0 = S0, E0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    # ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    sol = solve_ivp(seir, [np.amin(t), np.amax(t)], y0, t_eval=t,
                    args=(N, beta, gamma, sigma),
                    method='DOP853',
                    dense_output=True)

    return sol
