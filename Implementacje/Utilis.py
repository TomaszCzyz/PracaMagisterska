import warnings

import numpy as np
from scipy import integrate
from scipy.optimize import minimize_scalar


def norm_lp(f, interval, p):
    norm = integrate.quad(
        func=lambda x: f(x) ** p,
        a=interval[0], b=interval[1]
    )[0] ** (1 / p)
    return norm


def norm_infinity(f, interval):
    res = minimize_scalar(
        fun=lambda x: -1 * f(x),  # "-1" because we are looking for maximum, not minimum
        bounds=interval,
        method='bounded'
    )
    if not res.success:
        raise Exception("should not execute")

    return res['x']


def norm_infinity_2(f, interval):
    step = 1e-3
    max_value = 0.0
    current = interval[0]

    while current <= interval[1]:
        value = f(current)
        if value > max_value:
            max_value = value

        current = current + step

    return max_value


def worst_case_error(alg, lp_norm=2):
    approximation = alg.run()
    original_func = alg.example.raw_f
    interval = alg.example.f__a, alg.example.f__b

    def f(x):
        return abs(approximation(x) - original_func(x))

    if lp_norm == 'infinity':
        result = norm_infinity_2(f, interval)
    else:
        result = norm_lp(f, interval, lp_norm)

    return result


def worst_case_error_n(alg, repeat_count, lp_norm=2):
    warnings.filterwarnings("ignore")
    max_error = np.max([worst_case_error(alg, lp_norm) for i in range(repeat_count)])

    return max_error, alg.m, alg.example.f__noise


def interp_newton(xvals, yvals):
    """
    Return a function representing the interpolating
    polynomial determined by xvals and yvals.
    """
    assert len(xvals) == len(yvals)
    nbr_data_points = len(xvals)

    # sort inputs by xvals
    data = sorted(zip(xvals, yvals), reverse=False)
    xvals, yvals = zip(*data)

    depth = 1
    coeffs = [yvals[0]]
    iter_yvals = yvals

    while depth < nbr_data_points:

        iter_data = []

        for i in range(len(iter_yvals) - 1):

            delta_y = iter_yvals[i + 1] - iter_yvals[i]
            delta_x = xvals[i + depth] - xvals[i]
            iter_val = (delta_y / delta_x)
            iter_data.append(iter_val)

            # append top-most entries in tree to coeffs =>
            if i == 0:
                coeffs.append(iter_val)

        iter_yvals = iter_data
        depth += 1

    def f(ii):
        terms = []
        return_val = 0

        for j in range(len(coeffs)):

            iterval = coeffs[j]
            iterxvals = xvals[:j]
            for k in iterxvals:
                iterval *= (ii - k)
            terms.append(iterval)
            return_val += iterval

        return return_val

    return f
