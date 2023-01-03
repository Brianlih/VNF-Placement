import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

def construct_topo(filename_topo):
    nodes_firstColumn = np.genfromtxt(filename_topo, dtype="int", usecols=(1))
    nodes_secondColumn = np.genfromtxt(filename_topo, dtype="int", usecols=(2))
    quantity_nodes = max(np.amax(nodes_firstColumn), np.amax(nodes_secondColumn)) + 1
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
    global F, G, nodes, cpu_v, mem_v, eta_f, cpu_f, F_i, psi_f, profit_i, r_i, s_i, e_i
    global number_of_individual, number_of_gene_in_an_individual, elitism_rate, maximum_of_iteration_for_ga
    global maximum_of_iteration_for_one_ga_crossover, maximum_of_iteration_for_one_ga_mutation
    global number_of_individual_chose_from_population_for_tournament, crossover_rate, mutation_rate
    global edge_weights

    M = 1000000
    number_of_VNF_types = 5
    number_of_requests = 15
    number_of_nodes = 4

    F =  [0, 1, 2, 3, 4]
    edge_weights =  [2, 3, 6, 3]
    number_of_nodes =  4
    edges =  [(0, 1), (0, 2), (0, 0), (1, 3), (1, 1), (2, 3), (2, 2), (3, 3)]
    nodes =  [0, 1, 2, 3]
    cpu_v =  [9, 8, 9, 10]
    mem_v =  [3, 5, 5, 5]
    eta_f =  [1, 2, 3, 3, 3]
    cpu_f =  [2, 2, 5, 3, 3]
    F_i =  [[2, 4, 0, 1], [4, 3, 0], [1, 4, 2, 3], [4, 1, 2, 3, 0], [3, 1, 0, 2, 4], [3, 2, 4, 1, 0], [0, 2, 1, 4, 3], [3, 4, 0, 1], [1, 2, 3, 0], [2, 3], [2, 4, 3, 0], [0, 1], [1], [4, 3, 1], [3, 1, 4, 2]]
    psi_f =  [0.6666666666666666, 0.8, 0.6666666666666666, 0.8, 0.7333333333333333]
    profit_i =  [51.13333333333334, 35.13333333333334, 64.0, 67.33333333333333, 67.33333333333334, 67.33333333333333, 67.33333333333334, 42.33333333333334, 51.73333333333334, 41.2, 60.13333333333334, 10.533333333333333, 7.2, 39.00000000000001, 64.0]
    r_i =  [52, 68, 60, 74, 51, 64, 77, 76, 62, 54, 69, 61, 65, 68, 76]
    s_i =  [3, 2, 3, 0, 2, 2, 0, 1, 3, 3, 0, 1, 1, 2, 1]
    e_i =  [0, 0, 0, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3]

    number_of_individual = 50 # population size
    number_of_gene_in_an_individual = number_of_VNF_types * number_of_requests
    elitism_rate = 0.1
    maximum_of_iteration_for_ga = 1000
    maximum_of_iteration_for_one_ga_crossover = 20
    maximum_of_iteration_for_one_ga_mutation = 20
    number_of_individual_chose_from_population_for_tournament = 5
    crossover_rate = 0.5
    mutation_rate = 0.015

    number_of_topo = 1
    # G = construct_topo("D:/python_CPLEX_projects/VNF_placement/topo/topos/"
    #     + str(number_of_nodes) + "-"+ str(number_of_topo)+ ".txt")
    G = construct_topo("D:/python_CPLEX_projects/VNF_placement/topo/small_topo/"
        + str(number_of_nodes) + "-"+ str(number_of_topo)+ ".txt")
