import bisect
import logging

import mpmath
import numpy as np

from Examples import ExampleFunction
from Utilis import interp_newton, divided_diff_coeffs, newton_poly, divided_diff_coeffs_my
from Utilis_mpmath import divided_diff_coeffs_all_mpmath, newton_poly_mpmath

logger = logging.getLogger(__name__)
mpmath.mp.dps = 60


class Alg2014:
    """
    example - function to approximate (containing data about class parameters, interval and noise)
    n_knots - initial mesh resolution

    Execution example:
    alg = Alg2014(example=Example2(None), n_knots=1234)
    approximation = alg.run()
    """

    def __init__(self, example: ExampleFunction, n_knots):
        self.example = example
        self.m = n_knots

        self.t = np.linspace(self.example.f__a, self.example.f__b, self.m + 1, dtype='float64')
        self.h = (self.example.f__b - self.example.f__a) / self.m

        temp_d = np.power(self.h, (self.example.f__r + self.example.f__rho))
        self.d = temp_d if temp_d > 1e-14 else 1e-14

        # following values could be local, but they are defined as class values
        # to make monitoring of algorithm easier
        self.step1_interval = None
        self.b_set = None
        self.m_set = np.array(self.t)

    def run(self):
        logger.info("\nexecuting alg2014 dla m={} and noise={}".format(self.m, self.example.f__noise))
        self.step1()
        self.step2()
        approximation = self.step3()
        logger.info("executed alg2015")

        return approximation

    def step1(self):
        """
        step 1 - detecting interval with the biggest A_test (with singularity)
        1. check if exist intervals with diameters greater than 4*d in initial mesh
        2. if yes, find interval with the biggest A_test (A_test result has to be unique)
        """
        largest_result = 0
        second_largest_result = 0
        largest_result_index = 0
        second_largest_result_index = 0

        iter_count = 0
        for i in range(len(self.t) - 1):

            if self.t[i + 1] - self.t[i] <= 4 * self.d:
                continue
            iter_count += 1

            test_result = self.a_test_4(
                self.t[i],
                self.t[i] + self.d,
                self.t[i + 1] - self.d,
                self.t[i + 1]
            )
            if self.t[i] < self.example.singularity < self.t[i] + self.d or \
                    self.t[i + 1] - self.d < self.example.singularity < self.t[i + 1]:
                logger.info("step1 - singularity was in edge sub-interval")

            logger.info("step1 - interval(i:{:2}): [{:.14f} {:.14f}], test_result: {:.14f}".format(
                i, self.t[i], self.t[i + 1], test_result))

            if test_result > largest_result:
                second_largest_result = largest_result
                second_largest_result_index = largest_result_index

                largest_result = test_result
                largest_result_index = i

            elif test_result > second_largest_result:
                second_largest_result = test_result
                second_largest_result_index = i

        logger.info(
            "step1 - (largest(index:{}): {} second largest(index:{}): {})".format(
                largest_result_index, largest_result, second_largest_result_index, second_largest_result))

        if largest_result - second_largest_result < 1e-14:
            self.step1_interval = None
            logger.info("step1 - largest test result was not unique")
        else:
            self.step1_interval = (self.t[largest_result_index], self.t[largest_result_index + 1])
            logger.info("step1 - interval (u_1, v_1): [{:.14f} {:.14f}]".format(
                self.step1_interval[0], self.step1_interval[1]))
        pass

    def step2(self):
        """
        step 2(bisection) - create set of points that need to be added to initial mesh to make singularity harmless
        bisection is based on A_test values
        """

        if self.step1_interval is None:
            return []

        a_new, b_new = self.step1_interval
        self.b_set = [a_new, b_new]

        iter_count = 0
        while b_new - a_new > 4 * self.d:
            iter_count += 1

            v = (a_new + b_new) / 2
            self.b_set.append(v)

            a1 = self.a_test(a_new, a_new + self.d, v - self.d, v)
            a2 = self.a_test(v, v + self.d, b_new - self.d, b_new)

            if abs(a1 - a2) < 1e-14:
                break

            if a1 > a2:
                b_new = v
            else:
                a_new = v

        logger.info('step2 - iterations: {}'.format(iter_count))
        logger.info("step2 - b_set(len:{}): {}".format(len(self.b_set), sorted(self.b_set)))

    def step3(self):
        """
        step 3 - creating final approximation using initial mesh with appended points from step2
        """
        if self.b_set is not None and len(self.b_set) >= 2:
            b_set_sorted = np.sort(self.b_set)
            index = bisect.bisect_right(self.t, b_set_sorted[1])
            self.m_set = np.insert(self.m_set, index, b_set_sorted[1:-1])

        return self.create_approximation()

    def create_approximation(self):
        """
        Creating approximation with constant value on "small" sub-intervals and
        interpolating polynomial on the middle of "big" ones.
        Big interval are obtained by dividing intervals of length bigger than 4 * self.d into three parts,
        where edge sub-intervals are "small".
        """
        approx = []
        current_knot = self.m_set[0]
        for i in range(len(self.m_set) - 1):

            next_knot = self.m_set[i + 1]
            approx.append((current_knot, self.example.fun(current_knot)))

            if next_knot - current_knot < 4 * self.d:
                continue  # interval is small... no need for extra points

            knot1, knot2 = current_knot + self.d, next_knot - self.d

            knots = np.linspace(knot1, knot2, self.example.f__r + 1)
            values = self.example.fun(knots)
            polynomial = interp_newton(knots, values)

            approx.append((knot1, polynomial))
            approx.append((knot2, self.example.fun(knot2)))

            current_knot = next_knot

        approx.append((self.m_set[-1], self.example.fun(self.m_set[-1])))
        np_approx = np.array(approx)

        def final_approximation(t):
            if self.m_set[-1] < t < self.m_set[0]:
                raise Exception("value {} is outside function domain".format(t))

            ii = bisect.bisect_right(np_approx[:, 0], t)
            elem = np_approx[ii - 1, 1]

            if callable(elem):
                return elem(t)
            if isinstance(elem, (float, np.float64)):
                return elem

        return final_approximation

    def a_test(self, a0, a1, b1, b0):
        r = self.example.f__r

        knots_1 = np.linspace(a0, a1, r + 1).tolist()
        values = self.example.fun(knots_1)
        w1 = interp_newton(knots_1, values)

        knots_2 = np.linspace(b1, b0, r + 1).tolist()
        values = self.example.fun(knots_2)
        w2 = interp_newton(knots_2, values)

        z_arr = np.linspace(a1, b1, r + 1).tolist()
        test_values = [abs(w1(z_i) - w2(z_i)) for z_i in z_arr]
        # / ((b0 - a0) ** (r + self.example.f__rho)) <- no need when all studied intervals have the same length

        return max(test_values)

    def a_test_2(self, a0, a1, b1, b0):
        r = self.example.f__r

        w1_knots = np.linspace(b1, b0, r + 1)
        w1_values = self.example.fun(w1_knots)
        w1_coeffs = divided_diff_coeffs(w1_knots, w1_values)[0, :]

        w2_knots = np.linspace(a0, a1, r + 1)
        w2_values = self.example.fun(w2_knots)
        w2_coeffs = divided_diff_coeffs(w2_knots, w2_values)[0, :]

        z_arr = np.linspace(a1, b1, r + 1)
        w1_values_new = newton_poly(w1_coeffs, w1_knots, z_arr)
        w2_values_new = newton_poly(w2_coeffs, w2_knots, z_arr)

        test_values = [abs(w1_values_new[j] - w2_values_new[j]) for j in range(r + 1)]
        # / ((b0 - a0) ** (r + self.example.f__rho)) <- no need when all studied intervals have the same length

        return max(test_values)

    def a_test_3(self, a0, a1, b1, b0):
        r = self.example.f__r

        w1_knots = np.linspace(b1, b0, r + 1)
        w1_values = self.example.fun(w1_knots)
        w1_coeffs = divided_diff_coeffs_all_mpmath(w1_knots, w1_values)

        w2_knots = np.linspace(a0, a1, r + 1)
        w2_values = self.example.fun(w2_knots)
        w2_coeffs = divided_diff_coeffs_all_mpmath(w2_knots, w2_values)

        z_arr = mpmath.linspace(a1, b1, r + 1)
        w1_values_new = newton_poly_mpmath(w1_coeffs, w1_knots, z_arr)
        w2_values_new = newton_poly_mpmath(w2_coeffs, w2_knots, z_arr)

        test_values = [mpmath.fabs(w1_values_new[j] - w2_values_new[j]) for j in range(r + 1)]
        # / ((b0 - a0) ** (r + self.example.f__rho)) <- no need when all studied intervals have the same length

        return float(max(test_values))

    def a_test_4(self, a0, a1, b1, b0):
        r = self.example.f__r

        w1_knots = np.linspace(b1, b0, r + 1)
        w1_values = self.example.fun(w1_knots)
        w1_coeffs = divided_diff_coeffs_my(w1_knots, w1_values)

        w2_knots = np.linspace(a0, a1, r + 1)
        w2_values = self.example.fun(w2_knots)
        w2_coeffs = divided_diff_coeffs_my(w2_knots, w2_values)

        z_arr = np.linspace(a1, b1, r + 1)
        w1_values_new = newton_poly(w1_coeffs, w1_knots, z_arr)
        w2_values_new = newton_poly(w2_coeffs, w2_knots, z_arr)

        test_values = [abs(w1_values_new[j] - w2_values_new[j]) for j in range(r + 1)]
        # / ((b0 - a0) ** (r + self.example.f__rho)) <- no need when all studied intervals have the same length

        return max(test_values)
