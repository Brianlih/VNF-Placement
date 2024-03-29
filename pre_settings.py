import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

def construct_topo(filename_topo, lower_bound_of_pi_wv, upper_bound_of_pi_wv):
    nodes_firstColumn = np.genfromtxt(filename_topo, dtype="int", usecols=(1))
    nodes_secondColumn = np.genfromtxt(filename_topo, dtype="int", usecols=(2))
    quantity_nodes = max(np.amax(nodes_firstColumn), np.amax(nodes_secondColumn)) + 1
    edge_weights = [random.randint(lower_bound_of_pi_wv, upper_bound_of_pi_wv)
        for i in range(len(nodes_firstColumn))]
    # print("edge_weights = ", edge_weights)
    Graph = nx.Graph()
    for i in range(len(nodes_firstColumn)):
        Graph.add_edge(nodes_firstColumn[i], nodes_secondColumn[i], weight=edge_weights[i])
    for i in range(quantity_nodes):
        Graph.add_edge(i, i, weight=0)
    return Graph

def construct_test_topo(filename_topo):
    nodes_firstColumn = np.genfromtxt(filename_topo, dtype="int", usecols=(1))
    nodes_secondColumn = np.genfromtxt(filename_topo, dtype="int", usecols=(2))
    nodes_thirdColumn = np.genfromtxt(filename_topo, dtype="int", usecols=(3))
    quantity_nodes = max(np.amax(nodes_firstColumn), np.amax(nodes_secondColumn)) + 1
    Graph = nx.Graph()
    for i in range(len(nodes_firstColumn)):
        Graph.add_edge(nodes_firstColumn[i], nodes_secondColumn[i], weight=nodes_thirdColumn[i])
    for i in range(quantity_nodes):
        Graph.add_edge(i, i, weight=0)
    return Graph

def init(run_time, number_of_nodes, number_of_VNF_types):
    global G, nodes, cpu_v, mem_v, lower_bound_of_pi_wv, upper_bound_of_pi_wv, cost_f
    if run_time != 0:
        lower_bound_of_cpu_v = 8
        upper_bound_of_cpu_v = 16
        lower_bound_of_mem_v = 2
        upper_bound_of_mem_v = 4
        lower_bound_of_pi_wv = 1
        upper_bound_of_pi_wv = 10

        number_of_topo = 100
        G = construct_topo("topo/ftopo/" + str(number_of_nodes) + "-"+ str(number_of_topo)+".txt", lower_bound_of_pi_wv, upper_bound_of_pi_wv)
        # print("edges = ", G.edges())

        nodes = []
        for i in range(number_of_nodes):
            nodes.append(i)
        # print("nodes = ", nodes)

        cpu_v = []
        for i in range(number_of_nodes):
            cpu_v.append(random.randint(lower_bound_of_cpu_v, upper_bound_of_cpu_v))
        # print("cpu_v = ", cpu_v)

        mem_v = []
        for i in range(number_of_nodes):
            mem_v.append(random.randint(lower_bound_of_mem_v, upper_bound_of_mem_v))
        # print("mem_v = ", mem_v)
