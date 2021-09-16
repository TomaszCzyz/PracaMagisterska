# from abc import ABCMeta, abstractmethod, abstractproperty
from scipy import integrate
import numpy as np
import warnings
import copy


# class ApproxAlg(ABCMeta):
#     def __init__(self):
#         super().__init__()
#         self.f__a = None
#
#     @abstractmethod
#     def run(self):
#         pass


def worst_case_error(alg, p=2):
    warnings.filterwarnings("ignore")

    approximation = alg.run()

    result = integrate.quad(
        func=lambda x: abs(alg.f(x) - approximation(x)) ** p,
        a=alg.f__a, b=alg.f__b)[0] ** (1 / p)

    return result


def worst_case_error_n(alg, num, p=2):
    errors = []
    for n in range(num):
        errors.append(worst_case_error(alg, p))

    alg_m = copy.copy(alg.m)
    alg_noise = copy.copy(alg.noise)
    return np.max(errors), alg_m, alg_noise
