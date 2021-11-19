import mpmath


def divided_diff_coeffs_all_mpmath(x, y):
    """
    Function to calculate the divided differences table
    """
    n = len(y)
    coeffs = mpmath.zeros(n)
    knots = mpmath.matrix(x)
    # the first column is y
    for i in range(n):
        coeffs[i, 0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            coeffs[i, j] = (coeffs[i + 1, j - 1] - coeffs[i, j - 1]) / (knots[i + j] - knots[i])

    return coeffs


def newton_poly_mpmath(coeffs, x_data, x_arr):
    """
    Evaluate the newton polynomial at x
    """
    x_arr_mpmath = mpmath.matrix(x_arr)
    x_data_mpmath = mpmath.matrix(x_data)

    def f(xx):
        result = mpmath.mpf(0)
        for j in range(len(x_data)):
            temp = coeffs[0, j]
            for k in range(0, j):
                temp *= (xx - x_data_mpmath[k])
            result += temp

        return result

    return [f(elem) for elem in x_arr_mpmath]
