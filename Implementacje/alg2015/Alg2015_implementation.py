import bisect
import logging
import math

import numpy as np
from scipy.optimize import minimize_scalar

from Examples import ExampleFunction
from Utilis import interp_newton

logger = logging.getLogger(__name__)


class Alg2015:
    """
    func - approximated function
    n_knots - initial mesh resolution
    """

    def __init__(self, example: ExampleFunction, n_knots):
        self.example = example
        self.m = n_knots

        self.t = np.linspace(self.example.f__a, self.example.f__b, self.m + 1, dtype='float64')
        self.y = example.fun(self.t)
        self.h = (self.example.f__b - self.example.f__a) / self.m

        # "d" can easily reach edge precision!!! hence condition in step2
        # self.d = (self.r + 1) * self.h
        self.d = self.h ** (self.example.f__r + self.example.f__rho)

        # following values could be local, but they are defined as class values
        # to make monitoring of algorithm easier
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
        logger.info("executing alg2015 dla m={}".format(self.m))
        self.step1()
        self.step2()
        self.step3()

        approx = []
        current_knot = self.t[0]
        # before (u_1, v_1)
        for i in range(self.i_max):
            next_knot = self.t[i + 1]

            knots = np.linspace(current_knot, next_knot, self.example.f__r + 1)
            values = self.example.fun(knots)
            polynomial = interp_newton(knots, values)

            approx.append((current_knot, polynomial))

            current_knot = next_knot

        # on (u_1, v_1)
        approx.append((self.u_1, self.p_neg))
        approx.append((self.ksi, self.p_pos))

        v_1_index = self.i_max + self.example.f__r + 1
        current_knot = self.t[v_1_index]
        # after (u_1, v_1)
        for i in range(v_1_index, len(self.t) - 1):
            next_knot = self.t[i + 1]

            knots = np.linspace(current_knot, next_knot, self.example.f__r + 1)
            values = self.example.fun(knots)
            polynomial = interp_newton(knots, values)

            approx.append((current_knot, polynomial))

            current_knot = next_knot

        # no the right edge of initial interval
        approx.append((self.t[-1], lambda x: self.example.fun(self.t[-1])))

        np_approx = np.array(approx)

        def final_approximation(t):
            ii = bisect.bisect_right(np_approx[:, 0], t) - 1
            return np_approx[ii, 1](t)

        return final_approximation

    def step1(self):
        i_max = np.argmax(self.divided_diff())
        self.u_1 = self.t[i_max]
        self.v_1 = self.t[i_max + self.example.f__r + 1]
        self.i_max = i_max

    def step2(self):
        r = self.example.f__r
        p_neg = interp_newton(
            self.t[self.i_max - r:self.i_max + 1],
            self.y[self.i_max - r:self.i_max + 1]
        )
        p_pos = interp_newton(
            self.t[self.i_max + r + 1: self.i_max + 2 * r + 1 + 1],
            self.y[self.i_max + r + 1: self.i_max + 2 * r + 1 + 1]
        )

        u = self.u_1
        v = self.v_1
        while v - u > self.d:
            if v - u < 1e-14:  # to avoid infinite loop caused by max precision
                break

            z = [u + j * (v - u) / (r + 2) for j in range(1, r + 2)]
            dif = [np.abs(p_pos(z_j) - p_neg(z_j)) for z_j in z]
            j_max = np.argmax(dif)
            f_value = self.example.fun(z[j_max])

            if abs(f_value - p_neg(z[j_max])) <= abs(f_value - p_pos(z[j_max])):
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
            res = minimize_scalar(
                fun=lambda x: -1 * abs(self.p_neg(x) - self.p_pos(x)),
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

            f_value = self.example.fun(z_max)

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

    def divided_diff(self):
        table = [self.y]
        for i in range(self.example.f__r + 1):
            next_row = [[(table[i][j + 1] - table[i][j]) / (self.t[j + i + 1] - self.t[j])
                         for j in range(0, len(self.y) - i - 1)]]
            table = table + next_row

        return table[-1]
