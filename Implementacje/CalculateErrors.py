import logging
import multiprocessing as mp
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from Examples import Example1, Example2, Example3, Example4, Example5, Example6, Example7
from Utilis import worst_case_error_n
from alg2014.Alg2014_implementation import Alg2014
from alg2015.Alg2015_implementation import Alg2015

markers = ['1', '2', '1', '2']
colors = ['orange', 'grey', 'green', 'b']
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁼⁽⁾")


def apply_global_plot_styles():
    plt.rcParams['axes.linewidth'] = 0.1
    plt.rcParams['font.size'] = 7
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 175
    plt.rcParams['savefig.dpi'] = 175


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

        if self.finished_tasks / self.tasks_number >= 0.99:
            self.plot_results_2()

    def plot_results_2(self, save=False):
        fig, ax = plt.subplots()

        ref_noise = None if None in self.log10_m_for_noise.keys() else list(self.log10_m_for_noise.keys())[0]

        x_min, x_max = min(self.log10_m_for_noise[ref_noise]), max(self.log10_m_for_noise[ref_noise])

        m_original = np.array(np.floor(np.power(10, self.log10_m_for_noise[ref_noise])), dtype='float64')
        theoretical_error = np.power(m_original, -(self.data['f__r'] + 1))
        reference_line = -np.log10(theoretical_error) - 2.0

        subplot_nr = 0
        for key_noise in sorted(self.log10_m_for_noise.keys(), key=lambda x: (x is not None, x), reverse=False):
            plt.plot(
                sorted(self.log10_m_for_noise[key_noise]), sorted(self.log10_errors_for_noise[key_noise]),
                c=colors[subplot_nr],
                marker=markers[subplot_nr],
                linewidth=1,
                label=r'$\delta$=' + ("{:.0e}".format(key_noise) if key_noise is not None else '0')
                # alpha=0.5
            )
            subplot_nr += 1

        plt.plot(
            sorted(self.log10_m_for_noise[ref_noise]), sorted(reference_line),
            color='grey',
            linestyle='--',
            linewidth=1,
            label=("m{}".format(-(self.data['f__r'] + 1))).translate(SUP),
        )

        plt.grid(linewidth=0.5, linestyle=':')
        plt.legend(numpoints=1)
        ax.xaxis.set_tick_params(width=0.5, color='grey')
        ax.yaxis.set_tick_params(width=0.5, color='grey')

        # description and general plot styles
        fig.text(0.5, 0.02, r'$\log_{10}m$', ha='center')
        fig.text(0.01, 0.5, r'$-\log_{10}err$', va='center', rotation='vertical')
        plt.suptitle("{} for {}(r={}, p={})\nbased on {} sample functions ({})".format(
            self.data['algorithm_name'], self.data['example_fun_name'],
            self.data['f__r'],
            self.data['p'],
            self.data['executions_number'],
            datetime.now()))
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
    name = example_name.capitalize()
    if name[:7] == 'Example':
        return globals()[name](delta, f__r)

    raise Exception("incorrect example function name")


def create_algorithm(algorithm_name, example_function, knots_number, p=2):
    if algorithm_name == 'alg2014':
        return Alg2014(example_function, knots_number)
    if algorithm_name == 'alg2015':
        return Alg2015(example_function, knots_number, p)
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
    tasks_number = len(knots_counts) * len(deltas)
    results_collector = ResultsCollector(tasks_number, extra_data)

    if parallel:
        with mp.Pool(processes=mp.cpu_count() - 2) as pool:
            apply_results = []
            for knots_number in reversed(knots_counts):
                for noise in deltas:
                    print("starting processing algorithm({} times) for m={} and delta={}..."
                          .format(repeat_count, knots_number, noise))

                    function = create_example(example_fun_name, noise, f__r=f__r)
                    alg = create_algorithm(algorithm_name, function, knots_number, p)

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
    log10_m_array = np.linspace(1.3, 3.5, num=15)  # 10 ** 4.7 =~ 50118

    m_array = [int(10 ** log10_m) for log10_m in log10_m_array]
    noises = [None]  # [None, 1e-12, 1e-8, 1e-4]
    n_runs = 1
    alg = 'alg2014'
    example = 'Example2'
    p_norm = 'infinity'
    r = 4

    # create_example(example).plot()

    results = calculate(n_runs, m_array, noises, alg, example, p=p_norm, parallel=True, f__r=r)
    # alg = Alg2015(example=Example2(None), n_knots=35, p=p_norm)
    # results = alg.run()

    print("FINISHED")
    return results


if __name__ == '__main__':
    apply_global_plot_styles()
    logging.basicConfig(level=logging.INFO, filename='myapp.log', format="%(asctime)s:%(message)s")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    start_datetime = datetime.now()
    logging.info('Started at %s', start_datetime.strftime("%d/%m/%Y %H:%M:%S"))

    main_callback = main()

    end_datetime = datetime.now()
    processing_time = end_datetime - start_datetime
    logging.info('Finished at %s (execution time: %s)',
                 end_datetime.strftime("%d/%m/%Y %H:%M:%S"),
                 str(processing_time))

    # %%

    main_callback.plot_results_2(save=True)

    # Example2(None)
    # alg = Alg2015(example=example_function, n_knots=8966)
    # worst_case_error_n(
    #     alg=alg,
    #     num=3
    # )
    # results = alg.run()
