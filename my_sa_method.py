import random, time
from copy import deepcopy
import settings

def find_new_solution(improved_greedy_sol, data):
    new_sol = deepcopy(improved_greedy_sol)
    r_index = random.randint(0, data.number_of_requests - 1)
    vnf_type = random.sample(data.F_i[r_index], k=1)
    start = data.number_of_VNF_types * r_index
    mut_loc = start + vnf_type
    while True:
        rn = random.randint(-1, data.number_of_nodes - 1)
        if rn != new_sol[mut_loc] and rn != data.s_i[r_index] and rn != data.e_i[r_index]:
            new_sol[mut_loc] = rn
            break
    return new_sol

def check_if_meet_delay_requirement(request, i, data):
    tau_vnf_i = 0
    tau_i = 0
    first_vnf = -1
    last_vnf = -1
    for vnf_1 in data.F_i[i]:
        for vnf_2 in data.F_i[i]:
            if settings.check_are_neighbors(vnf_1, vnf_2, data.F_i[i]):
                tau_vnf_i += settings.v2v_shortest_path_length(data.G, request[vnf_1], request[vnf_2])
            if settings.check_is_first_vnf(vnf_1, data.F_i[i]):
                first_vnf = vnf_1
            if settings.check_is_last_vnf(vnf_1, data.F_i[i]):
                last_vnf = vnf_1
    tau_i += tau_vnf_i
    tau_i += settings.v2v_shortest_path_length(data.G, data.s_i[i], request[first_vnf])
    tau_i += settings.v2v_shortest_path_length(data.G, data.e_i[i], request[last_vnf])
        
    if tau_i <= data.r_i[i]:
        return True
    return False

def check_capacity(new_sol, data):
    new_sol_overload_nodes = []
    new_sol_available_nodes = []

    # CPU constraint
    for i in data.nodes:
        new_sol_cpu_count = 0
        for j in range(len(new_sol)):
            if new_sol[j] == i:
                vnf_type = j % data.number_of_VNF_types
                new_sol_cpu_count += data.cpu_f[vnf_type]
        if new_sol_cpu_count > data.cpu_v[i] and i not in new_sol_overload_nodes:
            new_sol_overload_nodes.append(i)
        elif new_sol_cpu_count < data.cpu_v[i] and i not in new_sol_available_nodes:
            new_sol_available_nodes.append(i)
    
    # Memory constraint
    for i in data.nodes:
        new_sol_vnf_types = [0] * data.number_of_VNF_types
        new_sol_mem_count = 0
        for j in range(len(new_sol)):
            if new_sol[j] == i:
                tmp = j % data.number_of_VNF_types
                new_sol_vnf_types[tmp] = 1
        for j in range(len(new_sol_vnf_types)):
            if new_sol_vnf_types[j] == 1:
                new_sol_mem_count += 1
        if new_sol_mem_count > data.mem_v[i] and i not in new_sol_overload_nodes:
            new_sol_overload_nodes.append(i)
        elif new_sol_mem_count < data.mem_v[i] and i not in new_sol_available_nodes:
            new_sol_available_nodes.append(i)
        
    return new_sol_available_nodes, new_sol_overload_nodes

def check_acception(new_sol, data):
    acception = []
    for i in range(data.number_of_requests):
        start = data.number_of_VNF_types * i
        end = start + data.number_of_VNF_types
        acception.append(check_if_meet_delay_requirement(new_sol[start:end], i, data))
    return acception

def main(data_from_cplex, improved_greedy_sol, improved_greedy_res):
    data = data_from_cplex
    current_temperature = 1000
    final_temperature = 0.01
    cooling_rate = 0.99

    while current_temperature > final_temperature:
        cap = check_capacity(new_sol, data)
        new_sol = find_new_solution(improved_greedy_sol, cap, data)
        acception = check_acception(new_sol, data)
