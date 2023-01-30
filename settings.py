import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

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
    if f == arr[-1]:
        return 1
    return 0

def init(number_of_requests, number_of_VNF_types, seed):
    global M, F, G, nodes, cpu_v, mem_v, eta_f, cpu_f
    global number_of_individual, number_of_gene_in_an_individual, elitism_rate, iteration_for_ga
    global max_iter_cro_mut, number_of_individual_chose_from_population_for_tournament, crossover_rate, mutation_rate

    M = 1000000

    lower_bound_of_eta_f = 1
    upper_bound_of_eta_f = 3
    lower_bound_of_cpu_f = 2
    upper_bound_of_cpu_f = 5

    number_of_individual = 50 # population size
    elitism_rate = 0.1
    iteration_for_ga = 50
    max_iter_cro_mut = 15
    number_of_gene_in_an_individual = number_of_VNF_types * number_of_requests
    number_of_individual_chose_from_population_for_tournament = 5
    crossover_rate = 0.5
    mutation_rate = 0.015

    F = [i for i in range(number_of_VNF_types)]
    print("F = ", F)

    eta_f =[]
    s = seed
    for i in range(number_of_VNF_types):
        random.seed(s)
        eta_f.append(random.randint(lower_bound_of_eta_f, upper_bound_of_eta_f))
        s += 1
    print("eta_f = ", eta_f)

    cpu_f = []
    s = seed
    for i in range(number_of_VNF_types):
        random.seed(s)
        cpu_f.append(random.randint(lower_bound_of_cpu_f, upper_bound_of_cpu_f))
        s += 1
    print("cpu_f = ", cpu_f)
