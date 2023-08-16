import random
from datetime import datetime
import networkx as nx

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

def init(number_of_requests, number_of_VNF_types):
    global M, F, nodes, eta_f, cpu_f

    M = 1000000

    lower_bound_of_eta_f = 1
    upper_bound_of_eta_f = 2
    lower_bound_of_cpu_f = 1
    upper_bound_of_cpu_f = 7

    F = [i for i in range(number_of_VNF_types)]
    # print("F = ", F)

    eta_f =[]
    for i in range(number_of_VNF_types):
        eta_f.append(random.randint(lower_bound_of_eta_f, upper_bound_of_eta_f))
    # print("eta_f = ", eta_f)

    cpu_f = []
    for i in range(number_of_VNF_types):
        cpu_f.append(random.randint(lower_bound_of_cpu_f, upper_bound_of_cpu_f))
    # print("cpu_f = ", cpu_f)
