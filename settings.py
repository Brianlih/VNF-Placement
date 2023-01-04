import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

def construct_topo(filename_topo):
    nodes_firstColumn = np.genfromtxt(filename_topo, dtype="int", usecols=(1))
    nodes_secondColumn = np.genfromtxt(filename_topo, dtype="int", usecols=(2))
    quantity_nodes = max(np.amax(nodes_firstColumn), np.amax(nodes_secondColumn)) + 1
    edge_weights = [random.randint(lower_bound_of_pi_wv, upper_bound_of_pi_wv)
        for i in range(quantity_nodes)]
    print("edge_weights = ", edge_weights)
    Graph = nx.Graph()
    for i in range(len(nodes_firstColumn)):
        Graph.add_edge(nodes_firstColumn[i], nodes_secondColumn[i], weight=edge_weights[i])
    for i in range(quantity_nodes):
        Graph.add_edge(i, i, weight=0)
    return Graph

def v2v_shortest_path_length(graph, v1, v2):
    shortest_path_length = nx.shortest_path_length(graph, v1, v2, "weight", method="dijkstra")
    return shortest_path_length

def check_are_neighbors(m, f, arr):
    for i in range(len(arr)):
        if arr[i] == m and i < len(arr) - 1:
            if arr[i + 1] == f:
                return 1
            break
    return 0

def check_is_first_vnf(f, arr):
    if f == arr[0]:
        return 1
    return 0

def check_is_last_vnf(f, arr):
    if f == arr[len(arr) - 1]:
        return 1
    return 0

def init():
    global M, number_of_VNF_types, number_of_requests, number_of_nodes, lower_bound_of_eta_f
    global upper_bound_of_eta_f, lower_bound_of_cpu_f, upper_bound_of_cpu_f, lower_bound_of_cpu_v
    global upper_bound_of_cpu_v, lower_bound_of_mem_v, upper_bound_of_mem_v, lower_bound_of_rho_v
    global upper_bound_of_rho_v, lower_bound_of_F_i, upper_bound_of_F_i, lower_bound_of_pi_wv
    global upper_bound_of_pi_wv, lower_bound_of_r_i, upper_bound_of_r_i
    global F, G, nodes, cpu_v, mem_v, eta_f, cpu_f, F_i, psi_f, profit_i, r_i, s_i, e_i
    global number_of_individual, number_of_gene_in_an_individual, elitism_rate, iteration_for_one_ga
    global maximum_of_iteration_for_one_ga_crossover, maximum_of_iteration_for_one_ga_mutation
    global number_of_individual_chose_from_population_for_tournament, crossover_rate, mutation_rate

    M = 1000000
    number_of_VNF_types = 5
    number_of_requests = 15
    number_of_nodes = 4

    lower_bound_of_eta_f = 1
    upper_bound_of_eta_f = 3
    lower_bound_of_cpu_f = 2
    upper_bound_of_cpu_f = 5
    lower_bound_of_cpu_v = 7
    upper_bound_of_cpu_v = 10
    lower_bound_of_mem_v = 3
    upper_bound_of_mem_v = 5
    lower_bound_of_rho_v = 40
    upper_bound_of_rho_v = 50
    lower_bound_of_F_i = 1
    upper_bound_of_F_i = 5
    lower_bound_of_pi_wv = 1
    upper_bound_of_pi_wv = 10
    lower_bound_of_r_i = 50
    upper_bound_of_r_i = 80

    number_of_individual = 50 # population size
    number_of_gene_in_an_individual = number_of_VNF_types * number_of_requests
    elitism_rate = 0.1
    iteration_for_one_ga = 50
    maximum_of_iteration_for_one_ga_crossover = 20
    maximum_of_iteration_for_one_ga_mutation = 20
    number_of_individual_chose_from_population_for_tournament = 5
    crossover_rate = 0.5
    mutation_rate = 0.015

    F = [i for i in range(number_of_VNF_types)]
    print("F = ", F)
    number_of_topo = 1
    # G = construct_topo("D:/python_CPLEX_projects/VNF_placement/topo/topos/"
    #     + str(number_of_nodes) + "-"+ str(number_of_topo)+ ".txt")
    G = construct_topo("D:/python_CPLEX_projects/VNF_placement/topo/small_topo/"
        + str(number_of_nodes) + "-"+ str(number_of_topo)+ ".txt")
    # nx.draw_networkx(G)
    # plt.show()
    print("number_of_nodes = ", number_of_nodes)
    print("edges = ", G.edges())

    nodes = []
    for i in range(number_of_nodes):
        nodes.append(i)
    print("nodes = ", nodes)

    cpu_v = []
    for i in range(number_of_nodes):
        cpu_v.append(random.randint(lower_bound_of_cpu_v, upper_bound_of_cpu_v))
    print("cpu_v = ", cpu_v)

    mem_v = []
    for i in range(number_of_nodes):
        mem_v.append(random.randint(lower_bound_of_mem_v, upper_bound_of_mem_v))
    print("mem_v = ", mem_v)

    eta_f =[]
    for i in range(number_of_VNF_types):
        eta_f.append(random.randint(lower_bound_of_eta_f, upper_bound_of_eta_f))
    print("eta_f = ", eta_f)

    cpu_f = []
    for i in range(number_of_VNF_types):
        cpu_f.append(random.randint(lower_bound_of_cpu_f, upper_bound_of_cpu_f))
    print("cpu_f = ", cpu_f)

    F_i = []
    for i in range(number_of_requests):
        tmp = []
        number_of_needed_VNF_by_request_i = random.randint(lower_bound_of_F_i, upper_bound_of_F_i)
        tmp = random.sample(F, number_of_needed_VNF_by_request_i)
        F_i.append(tmp)
    print("F_i = ", F_i)

    psi_f = []
    count = 0
    for i in range(number_of_VNF_types):
        for j in range(len(F_i)):
            for k in range(len(F_i[j])):
                if F_i[j][k] == F[i]:
                    count += 1
                    break
        psi_f.append(count / number_of_requests)
        count = 0
    print("psi_f = ", psi_f)

    profit_i = []
    for i in range(number_of_requests):
        profit = 0
        for j in range(len(F_i[i])):
            profit += eta_f[F_i[i][j]] * (1 + psi_f[F_i[i][j]]) * cpu_f[F_i[i][j]]
        profit_i.append(profit)
    print("profit_i = ", profit_i)


    r_i = []
    for i in range(number_of_requests):
        r_i.append(random.randint(lower_bound_of_r_i, upper_bound_of_r_i))
    print("r_i = ", r_i)

    s_i = []
    for i in range(number_of_requests):
        s_i.append(random.randint(0, number_of_nodes - 1))
    print("s_i = ", s_i)

    e_i = []
    buffer = random.randint(0, number_of_nodes - 1)
    for i in range(number_of_requests):
        while(buffer == s_i[i]):
            buffer = random.randint(0, number_of_nodes - 1)
        e_i.append(buffer)
    print("e_i = ", e_i)
