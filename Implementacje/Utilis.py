import logging
import warnings

import numpy as np
from scipy import integrate

logger = logging.getLogger(__name__)
rng = np.random.default_rng()
omitted_part_of_interval = 0.0


def norm_lp(f, interval, p):
    norm = integrate.quad(
        func=lambda x: f(x) ** p,
        a=interval[0] + omitted_part_of_interval, b=interval[1] - omitted_part_of_interval
    )[0] ** (1 / p)
    return norm


def norm_infinity(f, interval, singularity_interval):
    step = 1e-3
    randoms = rng.random(100) * step
    max_value = 0.0
    counter = 0
    current = interval[0] + omitted_part_of_interval

    while current <= interval[1] - omitted_part_of_interval:
        value = f(current)
        if value > max_value:
            max_value = value

        # more dense close to edge and singularity
        if current < interval[0] + 0.1 or current > interval[1] - 0.1 or \
                singularity_interval[0] <= current < singularity_interval[1]:
            current += (step + randoms[counter % len(randoms)]) / 100
            # current = singularity_interval[1]
            # logger.info("skipping interval with singularity during calculating error")
        else:
            current += step + randoms[counter % len(randoms)]
        counter += 1

    return max_value


def worst_case_error(alg, lp_norm=2):
    approximation = alg.run()
    original_func = alg.example.raw_f
    interval = alg.example.f__a, alg.example.f__b

    def f(x):
        return abs(approximation(x) - original_func(x))

    if lp_norm == 'infinity':
        s = alg.example.singularity
        singularity_interval = (s - 0.1, s + 0.1) if isinstance(s, float) else None
        result = norm_infinity(f, interval, singularity_interval)
    else:
        result = norm_lp(f, interval, lp_norm)

    return result


def worst_case_error_n(alg, repeat_count, lp_norm=2):
    warnings.filterwarnings("ignore")
    max_error = np.max([worst_case_error(alg, lp_norm) for _ in range(repeat_count)])

    return max_error, alg.m, alg.example.f__noise


def divided_diff_coeffs(x, y):
    """
    function to calculate the divided differences table
    """
    n = len(y)
    coeffs = np.zeros([n, n])
    # the first column is y
    coeffs[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coeffs[i][j] = (coeffs[i + 1][j - 1] - coeffs[i][j - 1]) / (x[i + j] - x[i])

    return coeffs


def newton_poly(coeffs, x_data, x):
    """
    evaluate the newton polynomial
    at x
    """
    n = len(x_data) - 1
    p = coeffs[n]
    for k in range(1, n + 1):
        p = coeffs[n - k] + (x - x_data[n - k]) * p
    return p


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


def max_local_primitive(fun, a, b, num=200):
    fun_max = -float("inf")
    arg_max = None

    if b - a < 1e-14:
        logger.info("interval is smaller than 1e-14; skipping local maximum search")
        return None

    dx = (b - a) / num
    dx = dx if dx > 1e-14 else 1e-14
    xval = a + dx
    while xval < b:
        value = fun(xval)
        if value > fun_max:
            fun_max = value
            arg_max = xval

        xval += dx

    return arg_max
