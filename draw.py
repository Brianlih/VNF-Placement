import matplotlib.pyplot as plt
import time, datetime, random

result_mean_cplex_time_cost = [0.12113022804260254, 0.9974167346954346, 1.8883697986602783, 10906.952996253967, 1654.819324016571]
result_mean_random_time_cost = [0.0010898113250732422, 0.0024728775024414062, 0.002023458480834961, 0.0028481483459472656, 0.0027701854705810547]
result_mean_greedy_time_cost = [0.0008394718170166016, 0.0016977787017822266, 0.0015995502471923828, 0.0016782283782958984, 0.002077817916870117]
result_mean_hGreedy_time_cost = [0.0010271072387695312, 0.001672983169555664, 0.002091646194458008, 0.002351999282836914, 0.002458333969116211]
result_mean_improved_greedy_time_cost = [0.0049402713775634766, 0.010271549224853516, 0.012263298034667969, 0.013041973114013672, 0.01700735092163086]
number_of_requests = [4,6,8,10,12]

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
    # line 1 points
    x1 = number_of_requests
    y1 = result_mean_cplex_time_cost
    plt.plot(x1, y1, 's-', color='r', label="CPLEX", markersize=8, linewidth=2.5)

    # # line 2 points
    # x2 = number_of_requests
    # y2 = result_mean_ga_res_value
    # plt.plot(x2, y2, 'o-', color='g', label="GA", markersize=8, linewidth=2.5)

    # line 3 points
    x3 = number_of_requests
    y3 = result_mean_random_time_cost
    plt.plot(x3, y3, 'D-', color='b', label="Random", markersize=8, linewidth=2.5)

    # line 4 points
    x4 = number_of_requests
    y4 = result_mean_greedy_time_cost
    plt.plot(x4, y4, '*-', color='y', label="Greedy", markersize=8, linewidth=2.5)

    # line 5 points
    x5 = number_of_requests
    y5 = result_mean_hGreedy_time_cost
    plt.plot(x5, y5, 'x-', color='black', label="HGreedy", markersize=8, linewidth=2.5)

    # line 6 points
    x6 = number_of_requests
    y6 = result_mean_improved_greedy_time_cost
    plt.plot(x6, y6, 'D-', color='orange', label="Improved Greedy", markersize=8, linewidth=2.5)

    plt.xticks(number_of_requests, [str(number_of_requests[i]) for i in range(len(number_of_requests))])
    plt.xlabel('Number of requests')
    plt.ylabel('Profit')
    plt.legend()
    # plt.show()
    current_date = datetime.datetime.now()
    plt.savefig("../result/" + str(current_date.month) + str(current_date.day) + "-line.png")
