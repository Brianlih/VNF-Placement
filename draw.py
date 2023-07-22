import matplotlib.pyplot as plt
import numpy as np
import time, datetime, random

result_mean_cplex_time_cost = [1.7213527134486608, 6.4950636965887885, 6.895397254398891, 13.763271485056196, 166.698226247515]
result_mean_hGreedy_time_cost = [0.0009699719292776925, 0.001394033432006836, 0.0020048448017665316, 0.002517427716936384, 0.003148674964904785]
result_mean_improved_greedy_time_cost = [0.007462569645472935, 0.009599379130772181, 0.011977825845990862, 0.013792548860822405, 0.015609843390328544]
result_mean_sa_time_cost = [0.4544812100274222, 0.6251762424196515, 0.7863494498389108, 0.9829794679369245, 1.1851140430995397]
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

    # line 5 points
    x5 = number_of_requests
    y5 = result_mean_hGreedy_time_cost
    plt.plot(x5, y5, 'x-', color='paleturquoise', label="HGreedy", markersize=8, linewidth=2.5)

    # line 6 points
    x6 = number_of_requests
    y6 = result_mean_improved_greedy_time_cost
    plt.plot(x6, y6, 'D-', color='mediumturquoise', label="Improved-greedy", markersize=8, linewidth=2.5)

    # line 7 points
    x7 = number_of_requests
    y7 = result_mean_sa_time_cost
    plt.plot(x7, y7, 'D-', color='bisque', label="VISA", markersize=8, linewidth=2.5)

    plt.xticks(number_of_requests, [str(number_of_requests[i]) for i in range(len(number_of_requests))])
    plt.xlabel('Number of requests')
    plt.ylabel('Time Cost')
    plt.legend()
    # plt.show()
    current_date = datetime.datetime.now()
    plt.savefig("../result/" + str(current_date.month) + str(current_date.day) + "-line.png")
