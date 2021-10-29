import bisect
import logging

import numpy as np

from Examples import ExampleFunction
from Utilis import interp_newton

logger = logging.getLogger(__name__)


class Alg2014:
    """
    example - function to approximate (containing data about class parameters, interval and noise)
    n_knots - initial mesh resolution
    """

    def __init__(self, example: ExampleFunction, n_knots):
        self.example = example
        self.m = n_knots

        self.t = np.linspace(self.example.f__a, self.example.f__b, self.m + 1, dtype='float64')
        self.h = (self.example.f__b - self.example.f__a) / self.m
        # self.d = (self.r + 1) * self.h
        self.d = self.h ** (self.example.f__r + self.example.f__rho)

        # following values could be local, but they are defined as class values
        # to make monitoring of algorithm easier
        self.lagrangePoly = None
        self.step1_interval = None
        self.b_set = None
        self.m_set = np.array(self.t)

    def run(self):
        logger.info("executing alg2014 dla m={}".format(self.m))
        self.step1()
        self.step2()
        return self.step3()

    def step1(self):
        """
        step 1 - detecting interval with the biggest A_test (with singularity)
        1. check if exist intervals with diameters greater than 4*d in initial mesh
        2. if yes, find interval with the biggest A_test (A_test result has to be unique)
        """
        max_diam = np.max([self.t[i + 1] - self.t[i] for i in range(len(self.t) - 1)])

        if max_diam <= 4 * self.d:
            return
        else:
            largest_result = 0
            second_largest_result = 0
            largest_result_index = 0

            for i in range(len(self.t) - 1):
                if self.t[i + 1] - self.t[i] > 4 * self.d:
                    test_result = self.a_test(
                        self.t[i],
                        self.t[i] + self.d,
                        self.t[i + 1] - self.d,
                        self.t[i + 1]
                    )

                    if test_result > largest_result:
                        largest_result = test_result
                        largest_result_index = i
                    elif largest_result > test_result > second_largest_result:
                        second_largest_result = test_result

            if largest_result - second_largest_result < 1e-14:
                return

        self.step1_interval = (self.t[largest_result_index], self.t[largest_result_index + 1])
        logger.info('interval located in step1: {}'.format(self.step1_interval))

    def step2(self):
        """
        step 2(bisection) - create set of points that need to be added to initial mesh to make singularity harmless
        bisection is based on A_test values
        """

        if self.step1_interval is None:
            return []

        self.b_set = list(self.step1_interval)
        a_new, b_new = self.step1_interval

        while b_new - a_new > 4 * self.d:

            v = (a_new + b_new) / 2
            self.b_set.append(v)
            a1 = self.a_test(a_new, a_new + self.d, v - self.d, v)
            a2 = self.a_test(v, v + self.d, b_new - self.d, b_new)

            if a1 - a2 < 1e-14:
                return

            if a1 > a2:
                b_new = v
            else:
                a_new = v

    def step3(self):
        """
        step 3 - creating final approximation using initial mesh with appended points from step2
        """
        if self.b_set is not None and len(self.b_set) >= 2:
            b_set_sorted = np.sort(self.b_set)
            index = bisect.bisect_right(self.t, b_set_sorted[1])
            self.m_set = np.insert(self.m_set, index, b_set_sorted[1:-1])

        approx = []

        current_knot = self.m_set[0]
        for i in range(len(self.m_set) - 1):

            next_knot = self.m_set[i + 1]

            approx.append((current_knot, self.example.fun(current_knot)))

            if next_knot - current_knot < 4 * self.d:
                continue  # interval is small... no need for extra points

            # each "big" interval we divide for 3 sub-intervals
            # on sub-intervals near edge approx has constant value
            # and on the middle sub-interval we use interpolating polynomial
            knot1, knot2 = current_knot + self.d, next_knot - self.d

            knots = np.linspace(knot1, knot2, self.example.f__r + 1)
            values = self.example.fun(knots)
            polynomial = interp_newton(knots, values)

            # extra knot with approximating polynomial on interval [knot1, knot2)
            approx.append((knot1, polynomial))

            # extra knot with constant function on interval [knot2, next_knot)
            approx.append((knot2, self.example.fun(knot2)))

            current_knot = next_knot

        approx.append((self.m_set[-1], self.example.fun(self.m_set[-1])))
        np_approx = np.array(approx)

        def final_approximation(t):
            ii = bisect.bisect_right(np_approx[:, 0], t) - 1

            if callable(np_approx[ii, 1]):
                return np_approx[ii, 1](t)
            else:
                return np_approx[ii, 1]

        return final_approximation

    def a_test(self, a0, a1, b1, b0):
        knots = np.linspace(b1, b0, self.example.f__r + 1)
        values = self.example.fun(knots)
        w1 = interp_newton(knots, values)

        knots = np.linspace(a0, a1, self.example.f__r + 1)
        values = self.example.fun(knots)
        w2 = interp_newton(knots, values)

        z_arr = np.linspace(a1, b1, self.example.f__r + 1)
        test_values = [(np.abs(w1(z_i) - w2(z_i))) / (b0 - a0) for z_i in z_arr]
        #  ** (self.r + self.rho)  <-- no need for this operation (the same operation in each test)

        return np.max(test_values)
