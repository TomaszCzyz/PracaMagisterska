import logging
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

from Examples import Example2
from Utilis import worst_case_error_n
from alg2014.Alg2014_implementation import Alg2014
from alg2015.Alg2015_implementation import Alg2015


class MyCallback:
    def __init__(self, max_count, algorithm_name):
        self.algorithm_name = algorithm_name
        self.finished_tasks = 0
        self.tasks_number = max_count
        self.log10_errors_for_noise = {}

        self.colors = ['orange', 'grey', 'green', 'b']
        self.markers = ['s', 'd', ',', 'x', 'o', '+', '.', 'v', '^', '<', '>']

    def apply_result_handler(self, args):
        error, alg_m, alg_noise = args
        self.finished_tasks += 1
        self.print_status(args)

        if alg_noise not in self.log10_errors_for_noise.keys():
            self.log10_errors_for_noise[alg_noise] = []
        self.log10_errors_for_noise[alg_noise].append([np.log10(alg_m), np.log10(error)])

        self.plot_results()

    def plot_results(self):
        if self.finished_tasks / self.tasks_number < 0.5:
            return

        fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
        plt.title(self.algorithm_name)
        axs = axs.ravel()

        temp = 0
        for key, value in self.log10_errors_for_noise.items():
            axs[temp].scatter(
                np.array(value)[:, 0], -np.array(value)[:, 1],
                c=self.colors[temp], marker=self.markers[temp], label="noise='{}'".format(key))
            axs[temp].legend(numpoints=1)
            temp += 1
        plt.show()

    def print_status(self, args=None):
        if args:
            print("finished task for m={} and noise={}".format(args[1], args[2]))
        print("finished tasks: {}/{}".format(self.finished_tasks, self.tasks_number), end='\r')


def calculate(n_times, array, deltas, algorithm_name, example_function):
    my_callback = MyCallback(len(array) * len(deltas), algorithm_name)

    errors = {}
    for elem in reversed(array):
        for delta in deltas:
            print("running algorithm({} times) for m={}, noise={}".format(n_times, elem, delta))
            if algorithm_name == 'alg2015':
                alg = Alg2015(func=example_function, n_knots=elem, noise=delta)
            elif algorithm_name == 'alg2014':
                alg = Alg2014(func=example_function, n_knots=elem, noise=delta)
            else:
                raise Exception("incorrect algorithm name")

            result_tuple = worst_case_error_n(
                alg=alg,
                num=n_times
            )
            my_callback.apply_result_handler(result_tuple)
            errors[(elem, delta)] = result_tuple[0]

    return errors


def calculate_async(num, array, deltas, algorithm_name, example_function):
    my_callback = MyCallback(len(array) * len(deltas), algorithm_name)
    errors = {}
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:

        for elem in reversed(array):

            for delta in deltas:
                print("starting processing algorithm({} times) for m={} and delta={}...".format(num, elem, delta))
                if algorithm_name == 'alg2015':
                    alg = Alg2015(func=example_function, n_knots=elem, noise=delta)
                elif algorithm_name == 'alg2014':
                    alg = Alg2014(func=example_function, n_knots=elem, noise=delta)
                else:
                    raise Exception("incorrect algorithm name")

                apply_result = pool.apply_async(
                    func=worst_case_error_n,
                    args=(alg, num),
                    callback=my_callback.apply_result_handler)

                errors[(elem, delta)] = apply_result

        my_callback.print_status()
        for r in errors.values():
            r.wait()

        errors = {k: (v.get())[0] for k, v in errors.items()}

    return errors


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info('Started')

    example_fun = Example2()
    example_fun.f__r = 2

    # be careful with parameters bellow, e.g. too small m can break an algorithm
    log10_m_array = np.linspace(1.8, 4.0, num=15)  # 10 ** 4.7 =~ 50118

    n_runs = 20
    noises = [None, 10e-4]  # [None, 10e-12, 10e-8, 10e-4]
    m_array = list(np.array(np.power(10, log10_m_array), dtype='int'))

    alg = Alg2014(func=example_fun, n_knots=10000, noise=None)
    results = worst_case_error_n(
        alg=alg,
        num=10
    )
    # results = calculate(n_runs, m_array, noises, 'alg2014', example_fun)
    # results = calculate_async(n_runs, m_array, noises, 'alg2014', example_fun)

    logging.info("results: {}".format(results))
    logging.info('Finished')
    # return results


if __name__ == '__main__':
    main()

    # %%

    # alg = Alg2014(func=example_fun, n_knots=102, noise=10e-4)
    # worst_case_error_n(
    #     alg=alg,
    #     num=10
    # )
