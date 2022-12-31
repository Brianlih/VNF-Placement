import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from docplex.mp.model import Model

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

if __name__ == "__main__":
    #--------------------------------------------------------------------------------------------
    # Creating input data
    #--------------------------------------------------------------------------------------------

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
    print("profit_i ", profit_i)


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

    #------------------------------------------------------------------------------------------
    # Creating the model
    #------------------------------------------------------------------------------------------

    VNF_placement_model = Model("VNF_placement")

    #------------------------------------------------------------------------------------------
    # Creating decsision variables
    #------------------------------------------------------------------------------------------

    z = VNF_placement_model.binary_var_dict(number_of_requests, name="z")
    x = VNF_placement_model.binary_var_dict((
        (i, f, v)
        for i in range(number_of_requests)
        for f in F_i[i]
        for v in nodes),
        name="x"
    )
    y = VNF_placement_model.binary_var_dict((
        (f, v)
        for f in F
        for v in nodes),
        name="y"
    )

    #-------------------------------------------------------------------------------------
    # Adding the constraints
    #-------------------------------------------------------------------------------------

    # Delay requirement constraint
    tau_vnf_i = []
    for i in range(number_of_requests):
        vnf_delay = 0
        for w in nodes:
            for v in nodes:
                if v != w:
                    for m in F_i[i]:
                        for f in F_i[i]:
                            vnf_delay += x[i, m, w] * x[i, f, v] * check_are_neighbors(m, f, F_i[i]) * v2v_shortest_path_length(G, w, v)
        tau_vnf_i.append(vnf_delay)

    tau_starting_i = []
    for i in range(number_of_requests):
        start_delay = 0
        for v in nodes:
            for f in F_i[i]:
                start_delay += x[i, f, v] * check_is_first_vnf(f, F_i[i]) * v2v_shortest_path_length(G, s_i[i], v)
        tau_starting_i.append(start_delay)

    tau_ending_i = []
    for i in range(number_of_requests):
        end_delay = 0
        for v in nodes:
            for f in F_i[i]:
                end_delay += x[i, f, v] * check_is_last_vnf(f, F_i[i]) * v2v_shortest_path_length(G, e_i[i], v)
        tau_ending_i.append(end_delay)

    tau_i = []
    for i in range(len(tau_ending_i)):
        tau_i.append(tau_vnf_i[i] + tau_starting_i[i] + tau_ending_i[i])

    sequence = set()
    removed_set = set()
    for i in range(number_of_requests):
        sequence.add(i)
    for i in range(number_of_requests):
        if len(F_i[i]) <= 1:
            VNF_placement_model.add_constraint(tau_i[i] <= M * (1-z[i]) + r_i[i])
            removed_set.add(i)
    sequence -= removed_set

    VNF_placement_model.add_quadratic_constraints(
        tau_i[i] <= M * (1-z[i]) + r_i[i] for i in sequence
    )

    # # Number of same type VNF in a request constraint
    # for i in range(number_of_requests):
    #     for f in F:
    #         count = 0
    #         for l in range(len(F_i[i])):
    #             if F_i[i][l] == f:
    #                 count += 1
    #         VNF_placement_model.add_constraint(count <= 1, ctname="numbre_of_same_type_VNF_in_a_request")

    # Relation between z and x constraint
    VNF_placement_model.add_constraints((
        sum(x[i, f, v] for v in range(number_of_nodes)) == z[i]
        for i in range(number_of_requests)
        for f in F_i[i]),
        names="relation_between_z_and_x"
    )

    # Relation between y and x constraint
    VNF_placement_model.add_constraints((
        y[f, v] - x[i, f, v] >= 0
        for i in range(number_of_requests)
        for f in F_i[i]
        for v in nodes),
        names="relation_between_y_and_x"
    )

    # CPU capacity constraint
    for v in range(number_of_nodes):
        occupied_cpu_resources = 0
        for i in range(number_of_requests):
            for f in F_i[i]:
                occupied_cpu_resources += x[i, f, v] * cpu_f[f]
        VNF_placement_model.add_constraint(occupied_cpu_resources <= cpu_v[v])
    
    # Memory capacity constraint
    for v in range(number_of_nodes):
        occupied_mem_resources = 0
        for f in F:
            occupied_mem_resources += y[f, v]
        VNF_placement_model.add_constraint(occupied_mem_resources <= mem_v[v])

    #-------------------------------------------------------------------------------------
    # Defineing the objective function
    #-------------------------------------------------------------------------------------

    obj_fn = sum(z[i] * profit_i[i] for i in range(number_of_requests))

    print(VNF_placement_model.print_information())

    VNF_placement_model.set_objective('max', obj_fn)

    # Solve the model and output the solution
    sol = VNF_placement_model.solve()
    if sol:
        print(sol)
    else:
        print("No solution found")
