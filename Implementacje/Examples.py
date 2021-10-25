import matplotlib.pyplot as plt
import numpy as np
from abc import ABC

rng = np.random.default_rng()


class ExampleFunction(ABC):
    def __init__(self, f__a, f__b, f__r, f__rho, f__noise):
        self.f__a = f__a
        self.f__b = f__b
        self.f__r = f__r
        self.f__rho = f__rho
        self.f__noise = f__noise

    def fun(self, x):
        pass

    def plot(self):
        mesh = np.arange(self.f__a, self.f__b, 0.03, dtype='float64')
        plt.scatter(mesh, self.fun(mesh), s=7)
        plt.title(type(self).__name__)
        plt.show()


class Example1(ExampleFunction):
    def __init__(self, f__noise=None, f__r=3):
        super().__init__(
            f__a=0,
            f__b=2 * np.pi + 0.5,
            f__r=f__r,
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
        return f_values_with_noise(self.raw_f, self.f__noise, x)


class Example2(ExampleFunction):
    def __init__(self, f__noise=None, f__r=3):
        super().__init__(
            f__a=0,
            f__b=3 * np.pi,
            f__r=f__r,
            f__rho=1,
            f__noise=f__noise
        )

    @staticmethod
    def raw_f(xx):
        if 0 <= xx < np.pi:
            return np.sin(xx)
        if np.pi <= xx <= 3 * np.pi:
            return np.sin(xx - np.pi)

    def fun(self, x):
        return f_values_with_noise(self.raw_f, self.f__noise, x)


class Example3(ExampleFunction):
    def __init__(self, f__noise=None, f__r=3):
        super().__init__(
            f__a=0.0,
            f__b=2.0,
            f__r=f__r,
            f__rho=1,
            f__noise=f__noise
        )

    @staticmethod
    def raw_f(xx):
        return abs(xx - 1.0) + (xx - 1.0) / 2.0 - (xx - 1.0) ** 2

    def fun(self, x):
        return f_values_with_noise(self.raw_f, self.f__noise, x)


def f_values_with_noise(fun, noise, x):
    values = f_values(fun, x)

    if noise is not None:
        values = add_noise(values, noise)

    return values


def f_values(fun, obj):
    if isinstance(obj, (float, np.float64)):
        return fun(obj)
    elif isinstance(obj, (list, np.ndarray)):
        return [fun(elem) for elem in obj]
    raise Exception("obj has to be list or float")


def add_noise(values, noise):
    if isinstance(values, (float, np.float64)):
        values = values + float(rng.uniform(-noise, noise))
    elif isinstance(values, (list, np.ndarray)):
        e = rng.uniform(-noise, noise, len(values))
        values = [values[i] + e[i] for i in range(len(values))]
    return values
