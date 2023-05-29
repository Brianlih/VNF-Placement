import matplotlib.pyplot as plt
import time, datetime, random

result_mean_cplex_time_cost = [1.7206578254699707, 27.4901282787323, 85.80319738388062, 242.11509919166565, 1090.8190262317657]
result_mean_greedy_time_cost = [0.0009717941284179688, 0.0012617111206054688, 0.0015358924865722656, 0.0016703605651855469, 0.0019230842590332031]
result_mean_hGreedy_time_cost = [0.0011599063873291016, 0.0017011165618896484, 0.0016741752624511719, 0.002090930938720703, 0.0024220943450927734]
result_mean_improved_greedy_time_cost = [0.004582881927490234, 0.005814552307128906, 0.007433414459228516, 0.007916688919067383, 0.008903741836547852]
result_mean_sa_time_cost = [0.34139013290405273, 0.4293196201324463, 0.5094871520996094, 0.6249442100524902, 0.8489704132080078]
number_of_requests = [6,8,10,12,14]

if False:
    results = [result_mean_cplex_res_value, result_mean_improved_greedy_res_value, result_mean_greedy_res_value, result_mean_random_res_value, result_mean_hGreedy_res_value]
    ylim = 0
    for i in range(len(results)):
        for j in range(len(results[i])):
            if results[i][j] > ylim:
                ylim = results[i][j]


    # results = [result_mean_cplex_res_value, result_mean_ga_res_value, result_mean_greedy_res_value, result_mean_random_res_value]
    colors = ['red', 'orange', 'yellow', 'black', 'blue']
    labels = ['CPLEX', 'VPIG', 'Greedy', 'HGreedy', 'Random']

    results = {
        'result_mean_cplex_res_value': tuple(result_mean_cplex_res_value),
        # 'result_mean_ga_res_value': tuple(result_mean_ga_res_value),
        'result_mean_improved_greedy_res_value': tuple(result_mean_improved_greedy_res_value),
        'result_mean_greedy_res_value': tuple(result_mean_greedy_res_value),
        'result_mean_hGreedy_res_value': tuple(result_mean_hGreedy_res_value),
        'result_mean_random_res_value': tuple(result_mean_random_res_value),
    }

    x = np.arange(len(number_of_requests))  # the label locations
    width = 0.16  # the width of the bars
    multiplier = -0.5
    l = 0

    fig, ax = plt.subplots(layout='constrained')

    for method, result in results.items():
        offset = width * multiplier
        ax.bar(x + offset, result, width, color=colors[l], label=labels[l], align='center')
        multiplier += 1
        l += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Number of requests')
    ax.set_ylabel('Profit')
    # ax.set_title('number_of_iteration=' + str(number_of_iteration))
    ax.set_xticks(x + width, number_of_requests)
    ax.legend(loc='upper left')
    ax.set_ylim(0, ylim + 10)
    # plt.show()
    current_date = datetime.datetime.now()
    plt.savefig("../result/" + str(current_date.month) + str(current_date.day) + "-bar.png")
else:
    # # line 1 points
    # x1 = number_of_requests
    # y1 = result_mean_cplex_time_cost
    # plt.plot(x1, y1, 's-', color='lightcoral', label="CPLEX", markersize=8, linewidth=2.5)

    # # line 2 points
    # x2 = number_of_requests
    # y2 = result_mean_ga_res_value
    # plt.plot(x2, y2, 'o-', color='g', label="GA", markersize=8, linewidth=2.5)

    # # line 3 points
    # x3 = number_of_requests
    # y3 = result_mean_random_time_cost
    # plt.plot(x3, y3, 'D-', color='b', label="Random", markersize=8, linewidth=2.5)

    # line 4 points
    x4 = number_of_requests
    y4 = result_mean_greedy_time_cost
    plt.plot(x4, y4, '*-', color='yellowgreen', label="Greedy", markersize=8, linewidth=2.5)

    # line 5 points
    x5 = number_of_requests
    y5 = result_mean_hGreedy_time_cost
    plt.plot(x5, y5, 'x-', color='powderblue', label="HGreedy", markersize=8, linewidth=2.5)

    # line 6 points
    x6 = number_of_requests
    y6 = result_mean_improved_greedy_time_cost
    plt.plot(x6, y6, 'D-', color='peru', label="Improved Greedy", markersize=8, linewidth=2.5)

    # line 7 points
    x7 = number_of_requests
    y7 = result_mean_sa_time_cost
    plt.plot(x7, y7, 'D-', color='orange', label="SA", markersize=8, linewidth=2.5)

    plt.xticks(number_of_requests, [str(number_of_requests[i]) for i in range(len(number_of_requests))])
    plt.xlabel('Number of requests')
    plt.ylabel('Time Cost')
    plt.legend()
    # plt.show()
    current_date = datetime.datetime.now()
    plt.savefig("../result/" + str(current_date.month) + str(current_date.day) + "-line.png")
