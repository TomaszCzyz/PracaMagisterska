import logging
import multiprocessing as mp
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from Examples import Example1
from Utilis import worst_case_error_n
from alg2014.Alg2014_implementation import Alg2014
from alg2015.Alg2015_implementation import Alg2015


class MyCallback:
    def __init__(self, max_count, extra_data):
        self.algorithm_name = extra_data[0]
        self.n_times = extra_data[1]
        self.example_function = extra_data[2]
        self.finished_tasks = 0
        self.tasks_number = max_count
        self.log10_errors_for_noise = {}
        self.log10_m_for_noise = {}

        self.colors = ['orange', 'grey', 'green', 'b']
        self.markers = ['v', '^', '<', '>', 's', 'd', ',', 'x', 'o', '+', '.', '1', '_', '.']

    def callback_handler(self, args):
        error, alg_m, alg_noise = args
        self.finished_tasks += 1
        self.print_status(args)

        if alg_noise not in self.log10_errors_for_noise.keys():
            self.log10_errors_for_noise[alg_noise] = []
            self.log10_m_for_noise[alg_noise] = []
        self.log10_errors_for_noise[alg_noise].append(-np.log10(error))
        self.log10_m_for_noise[alg_noise].append(np.log10(alg_m))

        self.plot_results()

    def plot_results(self, save=False):
        if self.finished_tasks / self.tasks_number < 0.75:
            return

        fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
        fig.text(0.5, 0.04, u'log\u2081\u2080m', ha='center')
        fig.text(0.04, 0.5, u'-log\u2081\u2080err', va='center', rotation='vertical')
        plt.suptitle("{} for {}(r={})\nbased on {} sample functions".format(
            self.algorithm_name, type(self.example_function).__name__,
            self.example_function.f__r, self.n_times))

        axs = axs.ravel()
        temp = 0
        sorted_noises = sorted(self.log10_m_for_noise.keys(), key=lambda x: (x is not None, x))
        for key_noise in sorted_noises:
            axs[temp].scatter(
                self.log10_m_for_noise[key_noise], self.log10_errors_for_noise[key_noise],
                c=self.colors[temp],
                marker=self.markers[-1],
                s=64,
                label="noise=" + "{:.0e}".format(key_noise) if key_noise is not None else "0"
            )
            axs[temp].legend(numpoints=1)
            axs[temp].grid()
            temp += 1

        if save:
            self.save_plt()
        plt.show()

    def save_plt(self):
        path = 'data/{}/'.format(self.algorithm_name)
        if not os.path.exists(path):
            os.makedirs(path)

        plot_nr = 0
        while os.path.exists("{}plot{}.jpg".format(path, plot_nr)):
            plot_nr += 1

        plt.savefig('{}plot{}.jpg'.format(path, plot_nr))

    def print_status(self, args=None):
        if args:
            print("finished task for m={} and noise={}".format(args[1], args[2]))
        print("finished tasks: {}/{}".format(self.finished_tasks, self.tasks_number), end='\r')


def calculate(n_times, array, deltas, algorithm_name, example_function):
    extra_data = algorithm_name, n_times, example_function
    my_callback = MyCallback(len(array) * len(deltas), extra_data)

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
                num=1 if delta is None else n_times
            )
            my_callback.callback_handler(result_tuple)

    return my_callback


def calculate_async(n_times, array, deltas, algorithm_name, example_function):
    extra_data = algorithm_name, n_times, example_function
    my_callback = MyCallback(len(array) * len(deltas), extra_data)

    with mp.Pool(processes=mp.cpu_count() - 2) as pool:

        apply_results = []
        for elem in reversed(array):

            for delta in deltas:
                print("starting processing algorithm({} times) for m={} and delta={}...".format(n_times, elem, delta))
                if algorithm_name == 'alg2015':
                    alg = Alg2015(func=example_function, n_knots=elem, noise=delta)
                elif algorithm_name == 'alg2014':
                    alg = Alg2014(func=example_function, n_knots=elem, noise=delta)
                else:
                    raise Exception("incorrect algorithm name")

                apply_result = pool.apply_async(
                    func=worst_case_error_n,
                    args=(alg, 1 if delta is None else n_times),
                    callback=my_callback.callback_handler)

                apply_results.append(apply_result)

        my_callback.print_status()
        for r in apply_results:
            r.wait()

    return my_callback


def main():
    example_fun = Example1()
    example_fun.plot()
    example_fun.f__r = 4

    # be careful with parameters bellow, e.g. too small m can break an algorithm
    log10_m_array = np.linspace(1.8, 4.0, num=10)  # 10 ** 4.7 =~ 50118

    n_runs = 10
    m_array = list(np.array(np.power(10, log10_m_array), dtype='int'))
    noises = [None, 1e-5, 1e-4, 1e-3]
    # [None, 10e-12, 10e-8, 10e-4] <- cannot take such small noise;
    # because of precision there is almost no difference in errors in such plot scale

    # results = calculate(n_runs, m_array, noises, 'alg2014', example_fun)
    results = calculate_async(n_runs, m_array, noises, 'alg2014', example_fun)

    # results = calculate(n_runs, m_array, noises, 'alg2015', example_fun)
    # results = calculate_async(n_runs, m_array, noises, 'alg2015', example_fun)

    # alg = Alg2015(func=example_fun, n_knots=1000, noise=10e-4)
    # worst_case_error_n(
    #     alg=alg,
    #     num=3
    # )

    return results


if __name__ == '__main__':
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    logging.basicConfig(level=logging.INFO, filename='myapp.log')

    start_datetime = datetime.now()
    logging.info('Started at %s', start_datetime.strftime("%d/%m/%Y %H:%M:%S"))

    main_callback = main()

    end_datetime = datetime.now()
    diff_datetime = end_datetime - start_datetime
    logging.info('Finished at %s (execution time: %s)',
                 end_datetime.strftime("%d/%m/%Y %H:%M:%S"),
                 str(diff_datetime))

    # %%

    main_callback.plot_results(save=True)

    # alg = Alg2014(func=example_fun, n_knots=102, noise=10e-4)
    # worst_case_error_n(
    #     alg=alg,
    #     num=10
    # )
