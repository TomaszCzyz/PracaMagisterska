import matplotlib.pyplot as plt
import numpy as np
from abc import ABC


class ExampleFunction(ABC):
    def __init__(self, f__a, f__b, f__r, f__rho, f__noise):
        self.f__a = f__a
        self.f__b = f__b
        self.f__r = f__r
        self.f__rho = f__rho
        self.f__noise = f__noise
        self.rng = np.random.default_rng()

    def fun(self, x):
        pass

    def plot(self):
        mesh = np.linspace(self.f__a, self.f__b, 200, dtype='float64')
        plt.scatter(mesh, self.fun(mesh), s=7)
        plt.title(type(self).__name__)
        plt.show()


class Example1(ExampleFunction):
    def __init__(self, f__noise=None):
        super().__init__(
            f__a=0,
            f__b=2 * np.pi + 0.5,
            f__r=3,
            f__rho=1,
            f__noise=f__noise
        )

    @staticmethod
    def raw_f(xx):
        if 0 <= xx < np.pi:
            return np.sin(xx - np.pi)
        if np.pi <= xx <= 2 * np.pi + 0.5:
            return np.sin(xx - np.pi - 0.5)

    def fun(self, x):
        if self.f__noise is not None:
            if isinstance(x, (float, np.float64)):
                x = x + float(self.rng.uniform(-self.f__noise, self.f__noise))
            elif isinstance(x, (list, np.ndarray)):
                x = x + list(self.rng.uniform(-self.f__noise, self.f__noise, len(x)))

        return f_values(self.raw_f, x)


class Example2(ExampleFunction):
    def __init__(self, f__noise=None):
        super().__init__(
            f__a=0,
            f__b=3 * np.pi,
            f__r=3,
            f__rho=1,
            f__noise=f__noise
        )

    def raw_f(self, xx):
        if 0 <= xx < np.pi:
            return np.sin(xx)
        if np.pi <= xx <= 3 * np.pi:
            return np.sin(xx - np.pi)

    def fun(self, x):
        values = f_values(self.raw_f, x)

        if self.f__noise is not None:
            if isinstance(values, (float, np.float64)):
                values = values + float(self.rng.uniform(-self.f__noise, self.f__noise))
            elif isinstance(values, (list, np.ndarray)):
                e = self.rng.uniform(-self.f__noise, self.f__noise, len(values))
                values = [values[i] + e[i] for i in range(len(values))]

        return values


# Barycentric interpolation of the function f(x) = |x| + x/2 − x2 in 21 and 101 Chebyshev
# points of the second kind on [−1, 1]. The dots mark the interpolated values fj .

def f_values(fun, obj):
    if isinstance(obj, (float, np.float64)):
        return fun(obj)
    elif isinstance(obj, (list, np.ndarray)):
        return [fun(elem) for elem in obj]
    raise Exception("obj has to be list or float")
