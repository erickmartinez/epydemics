from typing import Union
from scipy.optimize import OptimizeResult
import numpy as np
from scipy.linalg import svd
from scipy import optimize
from epydemics import seir_model, sir_model, confidence as cf
import datetime

epidemic_type = np.dtype([('days', 'i8'),
                          ('confirmed', 'i8'),
                          ('recovered', 'i8'),
                          ('dead', 'i8'),
                          ('infected', 'i8'),
                          ('date', 'M8')])

class Fitting:
    __models: list = ['SIR', 'SEIR']
    __data: np.ndarray
    __population: Union[int, float]
    __filteredData: np.ndarray

    def __init__(self, data: np.ndarray, population: Union[int, float]):
        if not isinstance(data, np.ndarray):
            raise ValueError('Input data must be an array')
        if len(data) < 4:
            raise  ValueError('Not enough data to fit')
        if population <= 0:
            raise ValueError('Invalid value for the population: {0}'.format(population))
        self.__data = data
        self.__filteredData = data
        self.__population = population

    @staticmethod
    def fobj(p: np.ndarray, time: np.ndarray, infected: np.ndarray, removed: np.ndarray = None,
             population: Union[int, float] = 1000, I0: int = 1, R0: int = 0, model: str = 'SEIR',
             fit_removed: bool = True):
        """
        The objective function to fit

        Parameters
        ----------
        p: np.ndarray
            The parameters to fit
        time: np.ndarray
            The independent variable
        infected: np.ndarray
            The infected population
        removed: np.ndarray
            The removed population
        population: int
            The total population
        I0: int
            The initial number of infections
        R0: int
            The initial number of recoveries
        model: str
            The model to use (SIR or SEIR)
        fit_removed: bool
            True if fitting the removed population false otherwise

        Returns
        -------
        np.ndarray
            The residuals
        """
        if model not in ['SIR', 'SEIR']:
            raise ValueError('Model \'{0}\' is unavailable. Valid models are:\n{1}'.format(
                model, ', '.join(['SIR', 'SEIR'])
            ))
        p = np.power(10, p)
        if model == 'SEIR':
            sol = seir_model.seir_model(time, N=population, beta=p[0], gamma=p[1], sigma=p[2], I0=I0, R0=R0, E0=p[3])
        else:
            sol = sir_model.sir_model(time, N=population, beta=p[0], gamma=p[1], I0=I0, R0=R0)

        y = sol.sol(time)
        S, E, I, R = y
        if fit_removed:
            n = len(infected)
            residual = np.empty(n * 2)
            for i in range(n):
                residual[i] = I[i] - infected[i]
                residual[i + n] = R[i] - removed[i]
            return residual
        else:
            return I - infected

    def fit(self, model: str = 'SEIR', fit_removed: bool = False, **kwargs):
        if model not in self.__models:
            raise ValueError('Model \'{0}\' is not implemented. Valid options are:\n{1}'.format(
                model, ', '.join(self.__models)))



        all_tol = np.finfo(np.float64).eps
        days = self.__filteredData['days']
        infected = self.__filteredData['infected']
        removed = self.__filteredData['removed']
        I0 = infected[0]
        R0 = removed[0]

        b0 = kwargs.get('b0', np.log10(np.array([1E-1, 1E-2, 1E0, I0])))
        max_nfev = kwargs.get('max_nfev', 200*len(infected))

        res = optimize.least_squares(
            self.fobj, b0,
            jac='3-point',
            # bounds=np.log10([1E-50,1E-50,1E-15,1], [np.inf, np.inf,np.inf, np.inf]),
            args=(days, infected, removed,
                    self.__population, I0, R0, model, fit_removed),
            xtol=all_tol,
            ftol=all_tol,
            gtol=all_tol,
            # x_scale='jac',
            # loss='soft_l1', f_scale=0.1,
            # loss='cauchy', f_scale=0.1,
            max_nfev=max_nfev,
            verbose=2
        )

        result = {
            'res': res,
            'model': model,

        }

    @property
    def data(self) -> np.ndarray:
        return self.__data

    @data.setter
    def data(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise ValueError('Input data must be an array')
        if len(data) < 4:
            raise  ValueError('Not enough data to fit')

