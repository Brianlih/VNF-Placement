import matplotlib.pyplot as plt
import numpy as np
import time, datetime, random

def main(arr):
    result_mean_cplex_time_cost = [1.0497397826268122, 14.036933477108295, 19.648495967571552, 470.3441597865178, 854.0287587826068]
    result_mean_hGreedy_time_cost = [0.0008602325732891376, 0.0015211288745586688, 0.0019505207355205829, 0.0032067665686974158, 0.0037832076732928935]
    result_mean_improved_greedy_time_cost = [0.008280644050011268, 0.011481908651498647, 0.01495511715228741, 0.017488406254695013, 0.019311189651489258]
    result_mean_sa_time_cost = [0.7261153918046218, 1.0363994561708891, 1.3671568723825307, 1.70669018305265, 2.0702769389519324]
    number_of_requests = [4,6,8,10,12]

    if False:
        results = [result_mean_cplex_acc_rate, result_mean_sa_acc_rate, result_mean_improved_greedy_acc_rate, result_mean_hGreedy_acc_rate]
        ylim = 0
        for i in range(len(results)):
            for j in range(len(results[i])):
                if results[i][j] > ylim:
                    ylim = results[i][j]


        colors = ['mediumseagreen', 'bisque', 'paleturquoise', 'mediumturquoise']
        patterns = ['//', '', '--', '||']
        labels = ['CPLEX', 'VISA', 'Improved-greedy', 'HGreedy']

        results = {
            'result_mean_cplex_res_value': tuple(result_mean_cplex_acc_rate),
            'result_mean_sa_acc_rate': tuple(result_mean_sa_acc_rate),
            'result_mean_improved_greedy_res_value': tuple(result_mean_improved_greedy_acc_rate),
            'result_mean_hGreedy_res_value': tuple(result_mean_hGreedy_acc_rate),
        }

        x = np.arange(len(number_of_requests))  # the label locations
        width = 0.2  # the width of the bars
        multiplier = -0.5
        l = 0

        fig, ax = plt.subplots(layout='constrained')

        for method, result in results.items():
            offset = width * multiplier
            if method == 'result_mean_sa_acc_rate':
                ax.bar(x + offset, result, width, color=colors[l], label=labels[l], align='center')
            else:
                ax.bar(x + offset, result, width, color='none', edgecolor=colors[l], label=labels[l], hatch=patterns[l], align='center')
            multiplier += 1
            l += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Number of requests')
        ax.set_ylabel('Acceptance Rate')
        # ax.set_title('number_of_iteration=' + str(number_of_iteration))
        ax.set_xticks(x + width, number_of_requests)
        ax.legend(loc='upper left')
        ax.set_ylim(0, ylim + 0.3)
        # plt.show()
        current_date = datetime.datetime.now()
        plt.savefig("../result/" + str(current_date.month) + str(current_date.day) + "-bar.png")
    else:
        # # line 1 points
        # x1 = number_of_requests
        # y1 = result_mean_cplex_time_cost
        # plt.plot(x1, y1, 's-', color='mediumseagreen', label="CPLEX", markersize=8, linewidth=2.5)

        # # line 2 points
        # x2 = number_of_requests
        # y2 = result_mean_ga_res_value
        # plt.plot(x2, y2, 'o-', color='g', label="GA", markersize=8, linewidth=2.5)

        # # line 3 points
        # x3 = number_of_requests
        # y3 = result_mean_random_time_cost
        # plt.plot(x3, y3, 'D-', color='b', label="Random", markersize=8, linewidth=2.5)

        # # line 4 points
        # x4 = number_of_requests
        # y4 = result_mean_greedy_time_cost
        # plt.plot(x4, y4, '*-', color='yellowgreen', label="Greedy", markersize=8, linewidth=2.5)

        # # line 5 points
        # x5 = number_of_requests
        # y5 = result_mean_hGreedy_time_cost
        # plt.plot(x5, y5, 'x-', color='paleturquoise', label="HGreedy", markersize=8, linewidth=2.5)

        # # line 6 points
        # x6 = number_of_requests
        # y6 = result_mean_improved_greedy_time_cost
        # plt.plot(x6, y6, 'D-', color='mediumturquoise', label="Improved-greedy", markersize=8, linewidth=2.5)

        # # line 7 points
        # x7 = number_of_requests
        # y7 = result_mean_sa_time_cost
        # plt.plot(x7, y7, 'o-', color='darkorange', label="VISA", markersize=8, linewidth=2.5)
        x = [i for i in range(len(arr))]
        y = arr
        plt.plot(x, y, color='darkorange', label="VISA", markersize=8, linewidth=2.5)

        # plt.xticks(number_of_requests, [str(number_of_requests[i]) for i in range(len(number_of_requests))])
        x_ticks_to_display = [x[i] for i in range(0, len(x), 1000)]
        plt.xticks(x_ticks_to_display)
        plt.xlabel('Iterations')
        plt.ylabel('Profit')
        plt.legend()
        # plt.show()
        current_date = datetime.datetime.now()
        plt.savefig("../result/" + str(current_date.month) + str(current_date.day) + "-line.png")
