from abc import ABC

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()


class ExampleFunction(ABC):
    def __init__(self, f__a, f__b, f__r, f__rho, f__noise, f__class, singularity=None):
        self.f__a = f__a
        self.f__b = f__b
        self.f__r = f__r
        self.f__rho = f__rho
        self.f__noise = f__noise
        self.f__class = f__class
        self.singularity = singularity

    def fun(self, x):
        pass

    def plot(self):
        mesh = np.arange(self.f__a, self.f__b, 0.005, dtype='float64')
        plt.scatter(mesh, self.fun(mesh), s=1)
        plt.title(type(self).__name__)
        plt.show()


class Example1(ExampleFunction):
    def __init__(self, f__noise=None, f__r=3):
        super().__init__(
            f__a=0,
            f__b=2 * np.pi + 0.5,
            f__r=f__r,
            f__rho=1,
            f__noise=f__noise,
            f__class='discontinuous',
            singularity=np.pi
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
            f__noise=f__noise,
            f__class='continuous',
            singularity=np.pi
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
    def __init__(self, f__noise=None, f__r=3, x_0=0.5):
        f__a = 0.0
        f__b = 4.0 - x_0
        super().__init__(
            f__a=f__a,
            f__b=f__b,
            f__r=f__r,
            f__rho=1,
            f__noise=f__noise,
            f__class='discontinuous',
            singularity=(f__b - f__a) / 2 + np.random.default_rng().uniform(-0.1, 0.1)
        )
        self.x_0 = x_0

    def raw_f(self, xx):
        def inner_f(xxx):
            return 2 * (0.5 - (1 / np.pi) * (
                    np.sin(np.pi * xxx) + (1 / 2) * np.sin(2 * np.pi * xxx) + (1 / 3) * np.sin(3 * np.pi * xxx)))

        if self.f__a <= xx <= self.singularity:
            return inner_f(xx)
        if self.singularity <= xx <= self.f__b:
            return inner_f(xx + self.x_0)

    def fun(self, x):
        return f_values_with_noise(self.raw_f, self.f__noise, x)


class Example4(ExampleFunction):
    def __init__(self, f__noise=None, f__r=3):
        f__a = 0.0
        f__b = 2 * np.pi
        super().__init__(
            f__a=f__a,
            f__b=f__b,
            f__r=f__r,
            f__rho=1,
            f__noise=f__noise,
            f__class='continuous',
            singularity=rng.uniform(f__a + 1.5, f__b - 1.5)
        )

    def raw_f(self, xx):
        return np.cos(xx) + np.e ** (-8 * abs(xx - self.singularity))

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


def plot_all_examples():
    name = 'Example'
    i = 1
    while name + str(i) in globals():
        var = globals()[name + str(i)]()
        var.plot()
        i += 1
