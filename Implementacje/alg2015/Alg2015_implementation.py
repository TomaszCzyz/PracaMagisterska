import bisect
import logging

import numpy as np

from Examples import ExampleFunction
from Utilis import interp_newton, max_local_primitive

logger = logging.getLogger(__name__)


class Alg2015:
    """
    example - function to approximate (containing data about class parameters, interval and noise)
    n_knots - initial mesh resolution
    """

    def __init__(self, example: ExampleFunction, n_knots, p):
        self.example = example
        self.m = n_knots
        self.p = 1 if p == 'infinity' else p

        self.t = np.linspace(self.example.f__a, self.example.f__b, self.m + 1, dtype='float64')
        self.y = example.fun(self.t)
        self.h = (self.example.f__b - self.example.f__a) / self.m

        # "d" can easily reach edge precision!!! hence condition in step2
        if example.f__class == 'continuous':
            self.d = (self.example.f__r + 1) * self.h + 1e-14
        else:
            omega = self.h ** ((self.example.f__r + self.example.f__rho) * self.p + 1)
            self.d = omega if omega > 5e-15 else 5e-15

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
        logger.info("\nexecuting alg2015 dla m={} and noise={}".format(self.m, self.example.f__noise))
        self.step1()
        self.step2()
        self.step3()
        approximation = self.create_approximation_2()
        logger.info("executed alg2015")

        return approximation

    def create_approximation_1(self):
        approx = []
        r = self.example.f__r

        # before (u_1, v_1)
        # for i in range(0, self.i_max - r + 1):  # also working without step r
        for i in range(0, self.i_max - r, r):
            knots = self.t[i:i + r + 1]
            values = self.y[i:i + r + 1]
            polynomial = interp_newton(knots, values)

            approx.append((self.t[i], polynomial))

        # on (u_1, v_1)
        approx.append((self.t[self.i_max - r], self.p_neg))
        # approx.append((self.t[self.i_max - r + 1], self.p_neg))  # approx.append((self.u_1, self.p_neg))
        approx.append((self.ksi, self.p_pos))

        p_pos_end_index = self.i_max + 2 * r + 1 + 1
        # after (u_1, v_1)
        # for i in range(v_1_index, len(self.t) - r + 1):  # also working without step r
        for i in range(p_pos_end_index, len(self.t), r):
            knots = self.t[i:i + r + 1]
            values = self.y[i:i + r + 1]
            polynomial = interp_newton(knots, values)

            approx.append((self.t[i], polynomial))

        # no the right edge of initial interval
        approx.append((self.t[-1], lambda x: self.y(self.t[-1])))

        np_approx = np.array(approx)

        def final_approximation(t):
            ii = bisect.bisect_right(np_approx[:, 0], t) - 1
            return np_approx[ii, 1](t)

        return final_approximation

    def create_approximation_2(self):
        approx = {}
        r = self.example.f__r
        # on (u_1, v_1)
        approx[(self.u_1, self.ksi)] = self.p_neg
        approx[(self.ksi, self.v_1)] = self.p_pos
        # from u_1 to 0
        i = self.i_max
        while i > r:
            i -= r
            # i -= 1

            knots = self.t[i:i + r + 1]
            values = self.y[i:i + r + 1]
            polynomial = interp_newton(knots, values)

            approx[(knots[0], knots[-1])] = polynomial
        # if there is the same interval, it will be overridden with the same polynomial
        knots = self.t[0:r + 1]
        values = self.y[0:r + 1]
        polynomial = interp_newton(knots, values)
        approx[(knots[0], knots[-1])] = polynomial
        # from v_1 to T
        i = self.i_max + 2 * r + 1
        while i < len(self.t) - r:
            knots = self.t[i:i + r + 1]
            values = self.y[i:i + r + 1]
            polynomial = interp_newton(knots, values)

            approx[(knots[0], knots[-1])] = polynomial
            i += r
            # i += 1
        # if there is the same interval, it will be overridden with the same polynomial
        knots = self.t[-r:]
        values = self.y[-r:]
        polynomial = interp_newton(knots, values)
        approx[(knots[0], knots[-1])] = polynomial
        sorted_intervals = sorted(approx.keys(), key=lambda x: x[0])
        temp = []
        for interval in sorted_intervals:
            temp.append([interval[0], approx[interval]])
        np_approx = np.array(temp)

        def final_approximation(t):
            ii = bisect.bisect_right(np_approx[:, 0], t) - 1
            return np_approx[ii, 1](t)

        return final_approximation

    def step1(self):
        i_max = np.argmax(self.divided_diff())
        self.u_1 = self.t[i_max]
        self.v_1 = self.t[i_max + self.example.f__r + 1]
        self.i_max = i_max
        logger.info("step1 - interval (u_1, v_1): [{:.14f} {:.14f}]".format(self.u_1, self.v_1))

    def step2(self):
        r = self.example.f__r
        begin = self.i_max - r
        end = self.i_max + 1
        p_neg = interp_newton(
            self.t[begin:end],
            self.y[begin:end]
        )
        begin = self.i_max + r + 1
        end = self.i_max + 2 * r + 1 + 1
        p_pos = interp_newton(
            self.t[begin:end],
            self.y[begin:end]
        )

        u = self.u_1
        v = self.v_1

        iter_count = 0
        while v - u > self.d:
            iter_count += 1

            z = [u + j * (v - u) / (r + 2) for j in range(1, r + 2)]
            dif = [np.abs(p_pos(z_j) - p_neg(z_j)) for z_j in z]
            j_max = np.argmax(dif).item()
            f_value = self.example.fun(z[j_max])

            if abs(f_value - p_neg(z[j_max])) <= abs(f_value - p_pos(z[j_max])):
                u = z[j_max]
            else:
                v = z[j_max]

        logger.info('step2 - iterations: {}'.format(iter_count))
        self.u_2 = u.item()
        self.v_2 = v.item()
        self.p_neg = p_neg
        self.p_pos = p_pos
        logger.info("step2 - interval (u_2, v_2): [{:.14f} {:.14f}]".format(self.u_2, self.v_2))

    def step3(self):
        u = self.u_2
        v = self.v_2

        def inter_diff(x):
            return abs(self.p_neg(x) - self.p_pos(x))

        iter_count = 0
        while True:
            iter_count += 1

            z_max = max_local_primitive(inter_diff, u, v)

            if z_max is None:
                logger.info('step3 - local maximum not found')
                break

            if abs(z_max - u) < 1e-14 or abs(z_max - v) < 1e-14:
                logger.info('step3 - local maximum was on interval edge')
                break

            f_value = self.example.fun(z_max)

            if np.abs(f_value - self.p_neg(z_max)) <= np.abs(f_value - self.p_pos(z_max)):
                u = z_max
            else:
                v = z_max

        logger.info('step3 - iterations: {}'.format(iter_count))

        u_3 = u
        v_3 = v

        ksi = max_local_primitive(lambda x: -1.0 * inter_diff(x), u_3, v_3)
        if ksi is None:
            logger.info("step3 - assigning ksi = u_3, because interval (u3, v3) was too small")
            ksi = u_3

        self.u_3 = u_3
        self.v_3 = v_3
        self.ksi = ksi
        logger.info("step3 - interval (u_3, v_3): [{:.14f} {:.14f}]".format(self.u_3, self.v_3))
        logger.info("step3 - ksi: {:.14f}".format(self.ksi))

    def divided_diff(self):
        table = [self.y]
        for i in range(self.example.f__r + 1):
            next_row = [[(table[i][j + 1] - table[i][j]) / (self.t[j + i + 1] - self.t[j])
                         for j in range(0, len(self.y) - i - 1)]]
            table = table + next_row

        return table[-1]
