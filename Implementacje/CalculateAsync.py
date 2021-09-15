import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from Examples import Example2
from Alg2015_implementation import Alg2015, worst_case_error_n


class MyCallback:
    def __init__(self, max_count):
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


def calculate(n_times, array, deltas):
    errors = {}
    for elem in array:
        for delta in deltas:
            print("running algorithm({} times) for m={}, noise={}".format(n_times, elem, delta))

            error = worst_case_error_n(
                alg=Alg2015(func=example_fun, n_knots=elem, noise=delta),
                num=n_times
            )
            errors[(elem, delta)] = error

    return errors


def calculate_async(num, array, deltas):
    my_callback = MyCallback(len(array) * len(deltas))

    with mp.Pool(processes=mp.cpu_count() - 1) as pool:

        errors = {}
        for elem in reversed(array):

            for delta in deltas:
                print("starting processing algorithm({} times) for m={} and delta={}...".format(num, elem, delta))
                alg = Alg2015(func=example_fun, n_knots=elem, noise=delta)
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


if __name__ == '__main__':
    example_fun = Example2()
    example_fun.f__r = 4

    # be careful with parameters bellow, e.g. too small m can break an algorithm
    log10_m_array = np.linspace(1.8, 4.5, num=15)  # 10 ** 4.7 =~ 50118

    n_runs = 30
    noises = [None, 10e-6, 10e-4, 10e-2]  # [None, 10e-12, 10e-8, 10e-4]
    m_array = list(np.array(np.power(10, log10_m_array), dtype='int'))

    # results = calculate(n_runs, m_array, noises)
    results = calculate_async(n_runs, m_array, noises)

    # %%

    # learn1 = np.linspace(0.01, 0.00001)
    # learn2 = np.log10(learn1)
    # learn3 = -learn2
