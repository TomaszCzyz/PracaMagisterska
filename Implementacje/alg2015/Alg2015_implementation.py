import logging
import math

import numpy as np
from scipy import interpolate
from scipy.optimize import minimize_scalar

from Examples import ExampleFunction
from Utilis import interp_newton

logger = logging.getLogger(__name__)


class Alg2015:
    """
        f - approximated function
        r, rho - smoothness constants
        f__a, f__b - interval
        n_knots - initial mesh resolution
        noise - None lub 1/2 noise amplitude
    """

    def __init__(self, func: ExampleFunction, n_knots, noise=None):
        self.rng = np.random.default_rng()

        self.f = func.fun
        self.f__a = func.f__a
        self.f__b = func.f__b
        self.r = func.f__r
        self.rho = func.f__rho
        self.m = n_knots
        self.noise = noise
        self.h = (self.f__b - self.f__a) / self.m
        # self.d = (self.r + 1) * self.h
        self.d = self.h ** (self.r + self.rho)
        self.t = np.linspace(self.f__a, self.f__b, self.m + 1, dtype='float64')
        if noise is not None:
            e = self.rng.uniform(-self.noise, self.noise, self.m + 1)
            self.y = [func.fun(self.t[j]) + e[j] for j in range(0, self.m + 1)]
        else:
            self.y = func.fun(self.t)

        # following values could be local, but they are defined as class values
        # to make easier monitoring of algorithm
        self.u_1 = None
        self.v_1 = None
        self.i_max = None
        self.u_2 = None
        self.v_2 = None
        self.p_neg = None
        self.p_pos = None
        self.u_3 = None
        self.v_3 = None
        self.ksi = None

    def run(self):
        self.step1()
        self.step2()
        self.step3()

        polynomial1 = interpolate.interp1d(
            self.t[:self.i_max + 1],
            self.y[:self.i_max + 1])

        polynomial2 = interpolate.interp1d(
            self.t[self.i_max + self.r + 1:],
            self.y[self.i_max + self.r + 1:])

        def final_approximation(x):
            def raw_f(xx):
                if self.t[0] <= xx < self.u_1:
                    return polynomial1(xx)
                if self.u_1 <= xx < self.ksi:
                    return self.p_neg(xx)
                if self.ksi <= xx < self.v_1:
                    return self.p_pos(xx)
                if self.v_1 <= xx <= self.t[self.m]:
                    return polynomial2(xx)
                print("WATCH OUT!!!")
                return -1

            if isinstance(x, (list, np.ndarray)):
                return [raw_f(elem) for elem in x]
            return raw_f(x)

        return final_approximation

    def step1(self):
        i_max = np.argmax(self.divided_diff_2())
        self.u_1 = self.t[i_max]
        self.v_1 = self.t[i_max + self.r + 1]
        self.i_max = i_max

    def step2(self):
        p_neg = interp_newton(
            self.t[self.i_max - self.r:self.i_max + 1],
            self.y[self.i_max - self.r:self.i_max + 1]
            # fill_value="extrapolate"
        )
        p_pos = interp_newton(
            self.t[self.i_max + self.r + 1: self.i_max + 2 * self.r + 1 + 1],
            self.y[self.i_max + self.r + 1: self.i_max + 2 * self.r + 1 + 1]
            # fill_value="extrapolate"
        )

        u = self.u_1
        v = self.v_1
        while v - u > self.d:
            z = [u + j * (v - u) / (self.r + 2) for j in range(1, self.r + 2)]
            dif = np.abs(p_pos(z) - p_neg(z))
            j_max = np.argmax(dif).item()

            if self.noise is None:
                f_value = self.f(z[j_max])
            else:
                f_value = self.f(z[j_max]) + self.rng.uniform(-self.noise, self.noise)

            if np.abs(f_value - p_neg(z[j_max])) <= np.abs(f_value - p_pos(z[j_max])):
                u = z[j_max]
            else:
                v = z[j_max]

        self.u_2 = u.item()
        self.v_2 = v.item()
        self.p_neg = p_neg
        self.p_pos = p_pos

    def step3(self):
        u = self.u_2
        v = self.v_2

        while True:
            # logger.info('u={}   v={}'.format(u, v))
            res = minimize_scalar(
                fun=lambda x: -1 * np.abs(self.p_neg(x) - self.p_pos(x)),
                # "-1" because we are looking for maximum, not minimum
                bounds=(u, v),
                method='bounded'
            )
            if not res.success:
                break

            z_max = res['x']

            if math.isclose(z_max, u, rel_tol=1e-14) or math.isclose(z_max, v, rel_tol=1e-14):  # => no local maximum
                logger.info('minimum was close ot interval edge')
                break

            f_value = self.f(z_max) if self.noise is None else self.f(z_max) + self.rng.uniform(-self.noise, self.noise)

            if np.abs(f_value - self.p_neg(z_max)) <= np.abs(f_value - self.p_pos(z_max)):
                u = z_max
            else:
                v = z_max

        u_3 = u
        v_3 = v

        res = minimize_scalar(
            fun=lambda x: np.abs(self.p_neg(x) - self.p_pos(x)),
            bounds=(u_3, v_3),
            method='bounded')

        if not res.success:
            raise Exception('could not minimize function form step 3')

        ksi = res['x']
        self.u_3 = u_3
        self.v_3 = v_3
        self.ksi = ksi

    def divided_diff(self, index):
        products = []
        for j in range(index, index + self.r + 1):
            product = 1.0
            for k in range(index, index + self.r + 1):
                if k == j:
                    continue
                product *= self.t[k] - self.t[j]
            products.append(self.y[j] * (1 / product))

        return math.fsum(products)

    def divided_diff_2(self):
        table = [self.y]
        for i in range(self.r + 1):
            next_row = [[(table[i][j + 1] - table[i][j]) / (self.t[j + i + 1] - self.t[j])
                         for j in range(0, len(self.y) - i - 1)]]
            table = table + next_row

        return table[-1]
