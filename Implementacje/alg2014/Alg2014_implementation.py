import numpy as np
import math
from scipy import interpolate
from Examples import ExampleFunction
import logging

logger = logging.getLogger(__name__)


class Alg2014:
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
        # self.d = self.rng.uniform(self.h ** (self.r + self.rho), (self.r + 1) * self.h)
        # self.d = (self.r + 1) * self.h
        self.d = self.h ** (self.r + self.rho)
        self.t = np.linspace(self.f__a, self.f__b, self.m + 1, dtype='float64')
        if noise is not None:
            e = self.rng.uniform(-self.noise, self.noise, self.m + 1)
            self.y = [func.fun(self.t[j]) + e[j] for j in range(0, self.m + 1)]
        else:
            self.y = func.fun(self.t)

        self.m_set = np.array(self.t)

    def run(self):
        temp1, temp2 = self.step1()
        b_set = self.step2(temp1, temp2)
        result = self.step3(b_set)

        return result

    def step1(self):
        """
        step 1 - detecting interval with the biggest A_test (with singularity)
        1. check if exists interval with diameter greater than 4*d in initial mesh
        2. if yes, find interval with the biggest A_test (A_test result has to be unique)
        """
        max_diam = np.max([self.t[i + 1] - self.t[i] for i in range(len(self.t) - 1)])

        if max_diam <= 4 * self.d:
            return 0, 0
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

            if math.isclose(largest_result, second_largest_result, rel_tol=1e-7):
                return 0, 0

        return self.t[largest_result_index], self.t[largest_result_index + 1]

    def step2(self, a, b):
        """
        step 2(bisection) - create set of points that need to be added to initial mesh to make singularity harmless
        bisection is based on A_test values
        """

        # logger.info('a=%f b=%f', a, b)
        if a == 0 and b == 0:
            return []

        a_new, b_new = a, b
        b_set = [a, b]

        while b_new - a_new > 4 * self.d:

            v = (a_new + b_new) / 2
            b_set.append(v)
            a1 = self.a_test(a_new, a_new + self.d, v - self.d, v)
            a2 = self.a_test(v, v + self.d, b_new - self.d, b_new)

            # logger.info('step2 -> a1=%f a2=%f', a1, a2)

            if math.isclose(a1, a2, rel_tol=1e-10):  # TODO: it should depend on precision
                return b_set

            if a1 > a2:
                b_new = v
            else:
                a_new = v

        return b_set

    def step3(self, b_set):
        """
        step 3 - creating final approximation using initial mesh with appended points from step2
        """
        if b_set is not []:
            b_set_sorted = np.sort(b_set)
            index = np.searchsorted(self.t, b_set_sorted[1], side='right')
            self.m_set = np.insert(self.m_set, index, b_set_sorted[1:-1])

        def adaptive_approximate(t):
            # locate knot that is smaller or equal to t (and it is closest to t)
            i = np.searchsorted(self.m_set, t, side='right') - 1
            if i < 0:
                logger.info('i was equal less than 0')
                i = 0

            if self.m_set[i + 1] - self.m_set[i] <= 4 * self.d:
                return self.f(self.m_set[i])
            else:
                if self.m_set[i] <= t < self.m_set[i] + self.d:
                    return self.f(self.m_set[i])
                if self.m_set[i] + self.d <= t < self.m_set[i + 1] - self.d:
                    left, right = self.m_set[i] + self.d, self.m_set[i + 1] - self.d
                    knots = np.linspace(left, right, self.r + 1, endpoint=True)
                    values = np.array([self.f(x) for x in knots])
                    polynomial = interpolate.interp1d(knots, values)
                    return polynomial(t)
                if self.m_set[i + 1] - self.d <= t < self.m_set[i + 1]:
                    return self.f(self.m_set[i + 1] - self.d)

            if math.isclose(t, self.m_set[-1]):
                return self.f(self.m_set[-1])

            return -1

        return adaptive_approximate

    def a_test(self, a0, a1, b1, b0):
        knots = np.linspace(b1, b0, self.r + 1, endpoint=True)
        if self.noise is None:
            values = np.array([self.f(x) for x in knots])
        else:
            values = np.array([self.f(x) + self.rng.uniform(-self.noise, self.noise) for x in knots])
        w1 = interpolate.interp1d(knots, values, fill_value="extrapolate")

        knots = np.linspace(a0, a1, self.r + 1, endpoint=True)
        if self.noise is None:
            values = np.array([self.f(x) for x in knots])
        else:
            values = np.array([self.f(x) + self.rng.uniform(-self.noise, self.noise) for x in knots])
        w2 = interpolate.interp1d(knots, values, fill_value="extrapolate")

        z_arr = np.linspace(a1, b1, self.r + 1, endpoint=True)  # "endpoint=True" is 100% good here
        values = [(np.abs(w1(z_i) - w2(z_i))) / ((b0 - a0) ** (self.r + self.rho)) for z_i in z_arr]

        return np.max(values)
