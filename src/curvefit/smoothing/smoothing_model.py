# -*- coding: utf-8 -*-
"""
    smoothing model
    ~~~~~~~~~~~~~~~

    Kalman smoothing model.
"""
import numpy as np
from scipy.optimize import minimize


class SimpleKalmanSmoothing:
    """Kalman smoothing class.
    """
    def __init__(self, t, y, w):
        """Constructor function for Simple Kalman Smoothing.

        Args:
            t (np.ndarray): Independent variables.
            y (np.ndarray): Dependent variables.
            w (float): Weights on the process model, ranging from 0 to 1.
        """
        assert isinstance(t, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert t.size == y.size
        assert 0 <= w <= 1

        self.t = t
        self.y = y
        self.w = w
        self.n = t.size

        # all the states
        self.s = None
        self.result = None
        # all the time differences
        self.dt = self.t[1:] - self.t[:-1]

    def objective(self, s):
        """Objective function.
        """
        s = s.reshape(self.n, 3)
        x = s[:, 0]
        v = s[:, 1]
        a = s[:, 2]

        # process residuals
        rx = x[1:] - (x[:-1] + self.dt*v[:-1] + 0.5*self.dt**2*a[:-1])
        rv = v[1:] - (v[:-1] + self.dt*a[:-1])
        ra = a[1:] - a[:-1]

        # measurement residuals
        rm = self.y - x

        return 0.5*self.w*(np.sum(rx**2) + np.sum(rv**2) + np.sum(ra**2)) + \
            0.5*(1.0 - self.w)*np.sum(rm**2)

    def gradient(self, s):
        """Gradient of the objective.
        """
        finfo = np.finfo(float)
        step = finfo.tiny/finfo.eps
        s_c = s + 0j
        grad = np.zeros(s.size)
        for i in range(s.size):
            s_c[i] += step*1j
            grad[i] = self.objective(s_c).imag/step
            s_c[i] -= step*1j

        return grad

    def smooth_observation(self, s0=None, options=None):
        """Apply optimizer smooth the observations.
        """
        if s0 is None:
            s0 = np.zeros(self.n*3)
        if options is None:
            options = {}
        result = minimize(
            fun=self.objective,
            x0=s0,
            jac=self.gradient,
            method='L-BFGS-B',
            options=options
        )

        self.result = result
        self.s = result.x.reshape(self.n, 3)
