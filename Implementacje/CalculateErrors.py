import logging
import multiprocessing as mp
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from Examples import Example2, Example1
from Utilis import worst_case_error_n
from alg2014.Alg2014_implementation import Alg2014
from alg2015.Alg2015_implementation import Alg2015

markers = ['v', '^', '<', '>', 's', 'd', ',', 'x', 'o', '+', '.', '1', '_', '.']
colors = ['orange', 'grey', 'green', 'b']


class MyCallback:
    def __init__(self, max_count, data):
        self.data = data
        self.tasks_number = max_count

        self.example_function = create_example(data['example_fun_name'])
        self.finished_tasks = 0
        self.log10_errors_for_noise = {}
        self.log10_m_for_noise = {}

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
        if self.finished_tasks / self.tasks_number <= 0.99:
            return

        fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
        fig.text(0.5, 0.04, u'log\u2081\u2080m', ha='center')
        fig.text(0.04, 0.5, u'-log\u2081\u2080err', va='center', rotation='vertical')
        plt.suptitle("{} for {}(r={}, p={})\nbased on {} sample functions".format(
            self.data['algorithm_name'], self.data['example_fun_name'],
            self.example_function.f__r,
            self.data['p'],
            self.data['executions_number']))

        axs = axs.ravel()
        temp = 0
        sorted_noises = sorted(self.log10_m_for_noise.keys(), key=lambda x: (x is not None, x))
        for key_noise in sorted_noises:
            axs[temp].scatter(
                self.log10_m_for_noise[key_noise], self.log10_errors_for_noise[key_noise],
                c=colors[temp],
                marker=markers[-1],
                s=64,
                label="noise=" + "{:.0e}".format(key_noise) if key_noise is not None else "0"
            )
            m_original = np.array(np.floor(np.power(10, self.log10_m_for_noise[None])), dtype='float64')
            theoretical_error = np.power(m_original, -(self.example_function.f__r + 1))
            reference_line = -np.log10(theoretical_error)
            axs[temp].plot(self.log10_m_for_noise[None], reference_line, linestyle='--')

            axs[temp].legend(numpoints=1)
            axs[temp].grid()
            temp += 1

        if save:
            self.save_plt()
        plt.show()

    def save_plt(self):
        path = 'data/{}/'.format(self.data['algorithm_name'])
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


def create_example(example_name, delta=None):
    if example_name == 'example1':
        return Example1(delta)
    if example_name == 'example2':
        return Example2(delta)
    raise Exception("incorrect example function name")


def create_algorithm(algorithm_name, example_function, knots_number):
    if algorithm_name == 'alg2014':
        return Alg2014(example_function, knots_number)
    if algorithm_name == 'alg2015':
        return Alg2015(example_function, knots_number)
    raise Exception("incorrect algorithm name")


def calculate(n_times, array, deltas, algorithm_name, example_fun_name, p, parallel=False):
    extra_data = {
        'algorithm_name': algorithm_name,
        'example_fun_name': example_fun_name,
        'executions_number': n_times,
        'p': p
    }
    my_callback = MyCallback(len(array) * len(deltas), extra_data)

    if parallel:
        with mp.Pool(processes=mp.cpu_count() - 2) as pool:
            apply_results = []
            for elem in reversed(array):
                for delta in deltas:
                    print(
                        "starting processing algorithm({} times) for m={} and delta={}...".format(n_times, elem, delta))

                    example_function = create_example(example_fun_name, delta)
                    alg = create_algorithm(algorithm_name, example_function, elem)

                    apply_result = pool.apply_async(
                        func=worst_case_error_n,
                        args=(alg, 1 if delta is None else n_times, p),
                        callback=my_callback.callback_handler)

                    apply_results.append(apply_result)

            my_callback.print_status()
            for r in apply_results:
                r.wait()

    else:
        for elem in reversed(array):
            for delta in deltas:
                print("running algorithm({} times) for m={}, noise={}".format(n_times, elem, delta))

                example_function = create_example(example_fun_name, delta)
                alg = create_algorithm(algorithm_name, example_function, elem)

                result_tuple = worst_case_error_n(
                    alg=alg,
                    num=1 if delta is None else n_times,
                    p=p
                )
                my_callback.callback_handler(result_tuple)

    return my_callback


def main():
    # be careful with parameters bellow, e.g. too small m can break an algorithm
    log10_m_array = np.linspace(1.3, 4.1, num=20)  # 10 ** 4.7 =~ 50118
    n_runs = 2
    m_array = list(np.array(np.power(10, log10_m_array), dtype='int'))
    noises = [None, 1e-12, 1e-8, 1e-4]
    # noises = [None, 1e-5, 1e-4, 1e-3]

    alg, example, p_norm = 'alg2014', 'example2', 'infinity'

    results = calculate(n_runs, m_array, noises, alg, example, p_norm, parallel=False)

    # alg = Alg2015(example=Example2(None), n_knots=8966)
    # results = alg.run()

    print("FINISHED")
    return results


if __name__ == '__main__':
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    logging.basicConfig(level=logging.INFO, filename='myapp.log')

    start_datetime = datetime.now()
    logging.info('Started at %s', start_datetime.strftime("%d/%m/%Y %H:%M:%S"))

    main_callback = main()

    end_datetime = datetime.now()
    processing_time = end_datetime - start_datetime
    logging.info('Finished at %s (execution time: %s)',
                 end_datetime.strftime("%d/%m/%Y %H:%M:%S"),
                 str(processing_time))

    # %%

    main_callback.plot_results(save=False)

    # Example2(None)
    # alg = Alg2015(example=example_function, n_knots=8966)
    # worst_case_error_n(
    #     alg=alg,
    #     num=3
    # )
    # results = alg.run()
