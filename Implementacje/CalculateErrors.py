import logging
import multiprocessing as mp
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from Examples import Example1, Example2, Example3
from Utilis import worst_case_error_n
from alg2014.Alg2014_implementation import Alg2014
from alg2015.Alg2015_implementation import Alg2015

markers = ['v', '^', '<', '>', 's', 'd', ',', 'x', 'o', '+', '.', '1', '_', '.']


class ResultsCollector:
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

        # data
        axs = axs.ravel()
        x_min, x_max = min(self.log10_m_for_noise[None]), max(self.log10_m_for_noise[None])
        subplot_nr = 0
        sorted_by_noise = sorted(self.log10_m_for_noise.keys(), key=lambda x: (x is not None, x))
        for key_noise in sorted_by_noise:
            axs[subplot_nr].plot(
                sorted(self.log10_m_for_noise[key_noise]), sorted(self.log10_errors_for_noise[key_noise]),
                c=['orange', 'grey', 'green', 'b'][subplot_nr],
                marker=markers[-1],
                linewidth=1,  # s=64,
                label=u'\u03B4=' + "{:.0e}".format(key_noise if key_noise is not None else 0)
            )
            m_original = np.array(np.floor(np.power(10, self.log10_m_for_noise[None])), dtype='float64')
            theoretical_error = np.power(m_original, -(self.data['f__r'] + 1))
            reference_line = -np.log10(theoretical_error)
            axs[subplot_nr].plot(sorted(self.log10_m_for_noise[None]), sorted(reference_line), linestyle='--',
                                 linewidth=1, color='grey')

            axs[subplot_nr].legend(numpoints=1)
            axs[subplot_nr].grid(linewidth=0.5, linestyle=':')
            axs[subplot_nr].xaxis.set_tick_params(width=0.5, color='grey')
            axs[subplot_nr].yaxis.set_tick_params(width=0.5, color='grey')
            subplot_nr += 1

        # description and general plot styles
        fig.text(0.5, 0.02, u'log\u2081\u2080m', ha='center')
        fig.text(0.01, 0.5, u'-log\u2081\u2080err', va='center', rotation='vertical')
        plt.suptitle("{} for {}(r={}, p={})\nbased on {} sample functions".format(
            self.data['algorithm_name'], self.data['example_fun_name'],
            self.data['f__r'],
            self.data['p'],
            self.data['executions_number']))
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.9,
                            wspace=0.07, hspace=0.1)
        plt.xlim([x_min - 0.2, x_max + 0.2])
        plt.xticks(np.arange(np.floor(x_min), np.ceil(x_max), 0.5))

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


def create_example(example_name, delta=None, f__r=4):
    if example_name == 'example1':
        return Example1(delta, f__r)
    if example_name == 'example2':
        return Example2(delta, f__r)
    if example_name == 'example3':
        return Example3(delta, f__r)
    raise Exception("incorrect example function name")


def create_algorithm(algorithm_name, example_function, knots_number):
    if algorithm_name == 'alg2014':
        return Alg2014(example_function, knots_number)
    if algorithm_name == 'alg2015':
        return Alg2015(example_function, knots_number)
    raise Exception("incorrect algorithm name")


def calculate(repeat_count, knots_counts, deltas, algorithm_name, example_fun_name, p, parallel=False, f__r=4):
    """
    calculates approximation error for specified algorithm and function with respect to norm L_p (1 <= p <= infinity)
    calculations are repeated 'n_times' for each knots count from 'array'
    """
    extra_data = {
        'algorithm_name': algorithm_name,
        'example_fun_name': example_fun_name,
        'executions_number': repeat_count,
        'f__r': f__r,
        'p': p
    }
    results_collector = ResultsCollector(len(knots_counts) * len(deltas), extra_data)

    if parallel:
        with mp.Pool(processes=mp.cpu_count() - 2) as pool:
            apply_results = []
            for knots_number in reversed(knots_counts):
                for noise in deltas:
                    print("starting processing algorithm({} times) for m={} and delta={}..."
                          .format(repeat_count, knots_number, noise))

                    function = create_example(example_fun_name, noise, f__r=f__r)
                    alg = create_algorithm(algorithm_name, function, knots_number)

                    apply_result = pool.apply_async(
                        func=worst_case_error_n,
                        args=(alg, 1 if noise is None else repeat_count, p),
                        callback=results_collector.callback_handler)

                    apply_results.append(apply_result)

            results_collector.print_status()
            for r in apply_results:
                r.wait()

    else:
        for knots_number in reversed(knots_counts):
            for noise in deltas:
                print("running algorithm({} times) for m={}, noise={}".format(repeat_count, knots_number, noise))

                function = create_example(example_fun_name, noise, f__r=f__r)
                alg = create_algorithm(algorithm_name, function, knots_number)

                result_tuple = worst_case_error_n(
                    alg=alg,
                    repeat_count=1 if noise is None else repeat_count,
                    lp_norm=p
                )
                results_collector.callback_handler(result_tuple)

    return results_collector


def main():
    log10_m_array = np.linspace(1.3, 4.4, num=20)  # 10 ** 4.7 =~ 50118

    m_array = [int(10 ** log10_m) for log10_m in log10_m_array]
    noises = [None, 1e-12, 1e-8, 1e-4]

    n_runs = 1
    alg, example = 'alg2015', 'example3'

    create_example(example).plot()

    results = calculate(n_runs, m_array, noises, alg, example, p='infinity', parallel=True, f__r=3)

    # alg = Alg2015(example=Example2(None), n_knots=8966)
    # results = alg.run()

    print("FINISHED")
    return results


if __name__ == '__main__':
    plt.rcParams['axes.linewidth'] = 0.1  # set the value globally
    plt.rcParams['font.size'] = 7
    plt.rcParams['font.family'] = 'sans-serif'
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

    main_callback.plot_results(save=True)

    # Example2(None)
    # alg = Alg2015(example=example_function, n_knots=8966)
    # worst_case_error_n(
    #     alg=alg,
    #     num=3
    # )
    # results = alg.run()
