import random, time, math
from copy import deepcopy
import settings

def find_new_solution(improved_greedy_sol, data, seed):
    available_nodes = []
    overload_nodes = []
    new_sol = deepcopy(improved_greedy_sol)
    random.seed(seed)
    r_index = random.randint(0, data.number_of_requests - 1)
    random.seed(seed)
    tmp = random.sample(data.F_i[r_index], k=1)
    vnf_type = tmp[0]
    start = data.number_of_VNF_types * r_index
    mut_loc = start + vnf_type
    muted_node = new_sol[mut_loc]
    while True:
        random.seed(seed)
        rn = random.randint(-1, data.number_of_nodes - 1)
        if rn != new_sol[mut_loc] and rn != data.s_i[r_index] and rn != data.e_i[r_index]:
            new_sol[mut_loc] = rn
            break
        seed += 1
    overload_nodes = check_capacity(new_sol, data)
    flag = True
    loop_count = 0
    while overload_nodes != []:
        for node in overload_nodes:
            candidates = []
            for i in range(len(new_sol)):
                if new_sol[i] == node:
                    candidates.append(i)
            random.seed(seed)
            tmp = random.sample(candidates, k=1)
            loc = tmp[0]
            vnf_type = loc % data.number_of_VNF_types
            available_nodes = find_available_nodes(new_sol, vnf_type, muted_node, data)
            if len(available_nodes) > 0:
                random.seed(seed)
                buffer = random.sample(available_nodes, k=1)
                new_sol[loc] = buffer[0]
            else:
                random.seed(seed)
                buffer = random.sample(data.nodes, k=1)
                new_sol[loc] = buffer[0]
            overload_nodes = check_capacity(new_sol, data)
            seed += 1
        loop_count += 1
        if loop_count >= 1000:
            flag = False
            break
    return new_sol, flag

def find_available_nodes(new_sol, v_type, muted_node, data):
    available_nodes = []
    for i in data.nodes:
        vnf_types = [0] * data.number_of_VNF_types
        mem_count = 0
        for j in range(len(new_sol)):
            if new_sol[j] == i:
                tmp = j % data.number_of_VNF_types
                vnf_types[tmp] = 1
        for j in range(len(vnf_types)):
            if vnf_types[j] == 1:
                mem_count += 1
        if mem_count < data.mem_v[i]: # Memory constraint
            cpu_count = 0
            for j in range(len(new_sol)):
                if new_sol[j] == i:
                    vnf_type = j % data.number_of_VNF_types
                    cpu_count += data.cpu_f[vnf_type]
            if (data.cpu_v[i] - cpu_count) > data.cpu_f[v_type]: # CPU constraint
                if i != muted_node:
                    available_nodes.append(i)
    return available_nodes

def check_if_meet_delay_requirement(request, i, data):
    if -1 in request:
        return False
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
    overload_nodes = []
    
    for i in data.nodes:
        vnf_types = [0] * data.number_of_VNF_types
        mem_count = 0
        for j in range(len(new_sol)):
            if new_sol[j] == i:
                tmp = j % data.number_of_VNF_types
                vnf_types[tmp] = 1
        for j in range(len(vnf_types)):
            if vnf_types[j] == 1:
                mem_count += 1
        if mem_count > data.mem_v[i]: # Memory constraint
            overload_nodes.append(i)
        else:
            cpu_count = 0
            for j in range(len(new_sol)):
                if new_sol[j] == i:
                    vnf_type = j % data.number_of_VNF_types
                    cpu_count += data.cpu_f[vnf_type]
            if cpu_count > data.cpu_v[i]: # CPU constraint
                overload_nodes.append(i)
        
    return overload_nodes

def check_acception(new_sol, data):
    start = -1
    last = -1
    acception = []
    for i in range(data.number_of_requests):
        start = data.number_of_VNF_types * i
        end = start + data.number_of_VNF_types
        count = 0

        j = start
        while j < end:
            if new_sol[j] != -2 and new_sol[j] != -1:
                count += 1
            j += 1
        if count < len(data.F_i[i]):
            acception.append(False)
        elif check_if_meet_delay_requirement(new_sol[start:end], i, data):
            acception.append(True)
        else:
            # j = start
            # while j < end:
            #     if new_sol[j] != -2:
            #         new_sol[j] = -1
            #     j += 1
            acception.append(False)
    return acception

def main(data_from_cplex, improved_greedy_sol, improved_greedy_res):
    data = data_from_cplex
    seed = 1
    # print("seed:", seed)
    start_time = time.time()

    current_sol = deepcopy(improved_greedy_sol)
    new_sol = []
    current_temperature = 1
    final_temperature = 0.0000001
    cooling_rate = 0.99
    diff = 0
    prob = 0
    current_best_res = improved_greedy_res

    while current_temperature > final_temperature:
        # print("current_temperature:",  current_temperature)
        new_sol, flag = find_new_solution(current_sol, data, seed)
        if flag:
            acception = check_acception(new_sol, data)
            profit = 0
            for i in range(data.number_of_requests):
                if acception[i]:
                    profit += data.profit_i[i]
            if profit >= current_best_res:
                current_best_res = profit
                current_sol = new_sol
            else:
                diff = profit - current_best_res
                prob = math.exp(diff / current_temperature)
                random.seed(seed)
                if random.uniform(0, 1) < prob:
                    current_best_res = profit
                    current_sol = new_sol
            current_temperature *= cooling_rate
        seed += 1

    end_time = time.time()
    time_cost = end_time - start_time

    total_profit = 0
    if current_best_res > improved_greedy_res:
        total_profit = current_best_res
    else:
        total_profit = improved_greedy_res

    acception = check_acception(new_sol, data)
    acc_count = 0
    for i in range(len(acception)):
        if acception[i]:
            acc_count += 1
    acc_rate = acc_count / data.number_of_requests

    res = {
        "total_profit": total_profit,
        "time_cost": time_cost,
        # "solution": greedy_solution,
        "acc_rate": acc_rate,
        # "average_delay": average_delay
    }
    return res
