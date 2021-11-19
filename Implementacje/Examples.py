from abc import ABC

import matplotlib.pyplot as plt
import mpmath
import numpy as np

rng = np.random.default_rng()


def plot_all_examples():
    name = 'Example'
    i = 1
    while name + str(i) in globals():
        var = globals()[name + str(i)]()
        var.plot()
        i += 1


class ExampleFunction(ABC):
    """
    abstract class contains basic template for each example
    """

    def __init__(self, f__a, f__b, f__r, f__rho, f__noise, f__class, singularity=None):
        self.f__a = f__a
        self.f__b = f__b
        self.f__r = f__r
        self.f__rho = f__rho
        self.f__noise = f__noise
        self.f__class = f__class if f__class in ['continuous', 'discontinuous'] else None
        self.singularity = singularity

    def get_noise(self):
        return 0.0 if self.f__noise is None else rng.uniform(-self.f__noise, self.f__noise)

    def fun(self, x):
        return self.raw_f(x) + self.get_noise()

    def fun_mp(self, x):
        return self.raw_f_mp(x) + self.get_noise()

    def raw_f(self, x):
        pass

    def raw_f_mp(self, x):
        pass

    def plot(self, label=None):
        mesh = np.arange(self.f__a, self.f__b, 0.005, dtype='float64')
        plt.scatter(mesh, [self.fun(x) for x in mesh], s=1, label=label)
        # plt.title(type(self).__name__)
        if label is not None:
            plt.legend()
        plt.show()


class Example1(ExampleFunction):
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
            singularity=rng.uniform(f__a + 2.5, f__b - 2.5)
        )

    def raw_f(self, x):
        return np.cos(x) + np.e ** (-8 * abs(x - self.singularity))

    def raw_f_mp(self, x):
        return mpmath.cos(x) + mpmath.e ** (-8 * mpmath.fabs(x - self.singularity))


class Example2(ExampleFunction):
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

    @staticmethod
    def inner_f(x):
        return 2 * (0.5 - (1 / np.pi) * (
                np.sin(np.pi * x) +
                (1 / 2) * np.sin(2 * np.pi * x) +
                (1 / 3) * np.sin(3 * np.pi * x)))

    @staticmethod
    def inner_f_mp(x):
        return 2 * (0.5 - (1 / mpmath.pi) * (
                mpmath.sin(mpmath.pi * x) +
                (1 / 2) * mpmath.sin(2 * mpmath.pi * x) +
                (1 / 3) * mpmath.sin(3 * mpmath.pi * x)))

    def raw_f(self, x):
        if self.f__a <= x < self.singularity:
            return self.inner_f(x)
        if self.singularity <= x <= self.f__b:
            return self.inner_f(x + self.x_0)

    def raw_f_mp(self, x):
        if self.f__a <= x < self.singularity:
            return self.inner_f_mp(x)
        if self.singularity <= x <= self.f__b:
            return self.inner_f_mp(x + self.x_0)


class Example3(ExampleFunction):
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

    def raw_f(self, x):
        if 0 <= x < np.pi:
            return np.sin(x - np.pi)
        if np.pi <= x <= 2 * np.pi + 0.5:
            return np.sin(x - np.pi - 0.5)

    def raw_f_mp(self, x):
        if 0 <= x < mpmath.pi:
            return mpmath.sin(x - mpmath.pi)
        if mpmath.pi <= x <= 2 * mpmath.pi + 0.5:
            return mpmath.sin(x - mpmath.pi - 0.5)


class Example4(ExampleFunction):
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

    def raw_f(self, x):
        if 0 <= x < np.pi:
            return np.sin(x)
        if np.pi <= x <= 3 * np.pi:
            return np.sin(x - np.pi)

    def raw_f_mp(self, x):
        if 0 <= x < mpmath.pi:
            return mpmath.sin(x)
        if mpmath.pi <= x <= 3 * mpmath.pi:
            return mpmath.sin(x - mpmath.pi)
