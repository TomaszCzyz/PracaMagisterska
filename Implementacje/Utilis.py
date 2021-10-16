from scipy import integrate
import numpy as np
import warnings
import copy


def worst_case_error(alg, p=2):
    approximation = alg.run()

    result = integrate.quad(
        func=lambda x: abs(alg.f(x) - approximation(x)) ** p,
        a=alg.f__a, b=alg.f__b)[0] ** (1 / p)

    return result


def worst_case_error_n(alg, num, p=2):
    warnings.filterwarnings("ignore")

    errors = []
    for n in range(num):
        errors.append(worst_case_error(alg, p))

    alg_m = copy.copy(alg.m)
    fun_noise = copy.copy(alg.f__noise)
    return np.max(errors), alg_m, fun_noise


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
        """
        Evaluate interpolated estimate at x.
        """
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
