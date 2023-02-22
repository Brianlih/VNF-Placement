import random
import networkx as nx
import matplotlib.pyplot as plt

def cal_result(G, number_of_iteration, number_of_node):
    result = 0
    for it in range(number_of_iteration):
        path_cost = 0
        current_node = 0
        while current_node != number_of_node - 1:
            candidate_nodes = []
            next_node = -1
            for i in range(number_of_node):
                if G.has_edge(current_node, i):
                    candidate_nodes.append(i)
            min_weight = 100
            for i in candidate_nodes:
                if G[current_node][i]["weight"] < min_weight:
                    next_node = i
                    min_weight = G[current_node][i]["weight"]
            if next_node != -1:
                path_cost += G[current_node][next_node]["weight"]
                current_node = next_node
            print("current_node: ", current_node)
        result += path_cost
    result /= number_of_iteration
    return result

number_of_iteration = 100
number_of_nodes = [5, 8, 11, 14, 17,20]
lower_bound_of_pi_wv = 2
upper_bound_of_pi_wv = 3
res = []
for n in number_of_nodes:
    print("n: ", n)
    G = nx.DiGraph()
    G.add_nodes_from([0, n - 1])
    if n == 5:
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)])
    elif n == 8:
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
                          (2, 5), (3, 6), (4, 7), (5, 7), (6, 7)])
    elif n == 11:
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (2, 6),
                          (3, 5), (3, 6), (4, 8), (5, 7), (6, 7), (6, 9),
                          (7, 10), (8, 10), (9, 10)])
    elif n == 14:
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 5), (2, 4), (3, 4),
                          (3, 6), (4, 8), (5, 8), (5, 9), (6, 7), (7, 10),
                          (7, 11), (8, 10), (8, 11), (9, 11), (9, 12),
                          (10, 13), (11, 13), (12, 13)])
    elif n == 17:
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 5), (1, 6), (2, 4),
                          (3, 6), (4, 7), (4, 8), (5, 7), (5, 8), (6, 9),
                          (7, 11), (8, 10), (8, 12), (9, 10), (10, 15),
                          (11, 14), (12, 13), (13, 16), (14, 16), (15, 16)])
    elif n == 20:
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (1,5), (2, 4),
                        (2, 5), (2, 6), (3, 6), (4, 7), (4, 8), (5, 7),
                        (6, 8), (6, 9), (7,10), (7, 11), (8, 12), (9, 11),
                        (10, 13), (10, 15), (11, 14), (12, 13), (13, 17),
                        (13, 18), (14, 17), (15, 16), (16, 19), (17, 19),
                        (18, 19)])
    for i in range(n):
        for j in range(n):
            if G.has_edge(i, j):
                G[i][j]["weight"] = random.randint(lower_bound_of_pi_wv, upper_bound_of_pi_wv)
    nx.draw_networkx(G)
    plt.show()

    res.append(cal_result(G, number_of_iteration, n))

    x2 = number_of_nodes
    y2 = res
    plt.plot(x2, y2, '*-', color='y', label="Greedy", markersize=8, linewidth=2.5)

    plt.xlabel('Number of nodes')
    plt.ylabel('Latency')
    plt.title('number_of_iteration=' + str(number_of_iteration))
    plt.legend()
    plt.show()

