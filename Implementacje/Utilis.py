import logging
import warnings

import numpy as np
from scipy import integrate

logger = logging.getLogger(__name__)
rng = np.random.default_rng()
omitted_part_of_interval = 1.0


def norm_lp(f, interval, p):
    """
    Calculates L_p norm for 1 < p < infinity of function f on a given interval
    """
    warnings.filterwarnings("ignore")
    norm = integrate.quad(
        func=lambda x: f(x) ** p,
        a=interval[0] + omitted_part_of_interval, b=interval[1] - omitted_part_of_interval
    )[0] ** (1 / p)
    return norm


def norm_infinity(f, interval, singularity=None):
    """
    Calculates maximum norm of function f on a given interval.
    Method iterate through interval using step 'step' with random change to avoid hitting special points in interval,
    such as points in initial mesh.
    Additionally, step is reduced near edges and singularity, if specified.
    """
    step = 1e-3
    margin = 0.1
    randoms = rng.random(100) * step  # predefine random numbers for performance
    step_reduction_ratio = 100

    start, stop = interval[0], interval[1]
    max_value = 0.0
    counter = 0

    def is_near_singularity(s, x, mar=0.1):
        if s is None or not isinstance(s, float):
            return False

        if s - mar <= x < s + mar:
            return True

    current = start + omitted_part_of_interval
    while current <= stop - omitted_part_of_interval:
        value = f(current)
        if value > max_value:
            max_value = value

        # more dense close to edge and singularity
        if current < start + margin or current > stop - margin or is_near_singularity(current, singularity, margin):
            current += (step + randoms[counter % len(randoms)]) / step_reduction_ratio

            # TODO delete below commented code. It was added for debugging purposes.
            # if is_near_singularity(current, singularity, margin):
            #     current = singularity + margin
            # logger.info("skipping interval with singularity during calculation of error")
        else:
            current += step + randoms[counter % len(randoms)]

        counter += 1

    return max_value


def worst_case_error(alg, lp_norm=2):
    """
    Calculates error of approximation for given algorithm.
    Function used for comparison is stored in algorithm as part of 'example' function
    """
    approximation = alg.run()
    original_func = alg.example.raw_f
    interval = alg.example.f__a, alg.example.f__b

    def f(x):
        return abs(approximation(x) - original_func(x))

    if lp_norm == 'infinity':
        singularity = alg.example.singularity
        result = norm_infinity(f, interval, singularity)
    else:
        result = norm_lp(f, interval, lp_norm)

    return result


def worst_case_error_n(alg, repeat_count, lp_norm=2):
    """
    Runs 'worst_case_error' method 'repeat_count' times for given norm.

    Returns tuple containing:
    maximum of calculated errors, initial mesh resolution and noise associated with the algorithm

    Execution example:
    example_function = Example2(1e-8)
    alg = Alg2015(example=example_function, n_knots=8966)
    error = worst_case_error_n(
        alg=alg,
        repeat_count=100
    )[0]  # <- if only interested in error
    """
    max_error = np.max([worst_case_error(alg, lp_norm) for _ in range(repeat_count)])

    return max_error, alg.m, alg.example.f__noise


def divided_diff_coeffs(x, y):
    """
    Function to calculate the divided differences table
    """
    n = len(y)
    coeffs = np.zeros([n, n])
    coeffs[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coeffs[i][j] = (coeffs[i + 1][j - 1] - coeffs[i][j - 1]) / (x[i + j] - x[i])

    return coeffs


def newton_poly(coeffs, x_data, x):
    """
    Evaluate the newton polynomial at x
    """
    n = len(x_data) - 1
    p = coeffs[n]
    for k in range(1, n + 1):
        p = coeffs[n - k] + (x - x_data[n - k]) * p
    return p


def interp_newton(xvals, yvals):
    """
    Return a function representing the interpolating polynomial determined by xvals and yvals.
    This method combines two steps:
    1. calculate coefficients
    2. create function depending on these coefficients
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

    def f(xx):
        return_val = 0

        for j in range(len(coeffs)):
            iterval = coeffs[j]
            iterxvals = xvals[:j]
            for k in iterxvals:
                iterval *= (xx - k)
            return_val += iterval

        return return_val

    return f
