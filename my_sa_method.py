import random, time, math
from copy import deepcopy
import settings

def find_new_solution(improved_greedy_sol, data, seed):
    available_nodes = []
    overload_nodes = []
    new_sol = deepcopy(improved_greedy_sol)
    while True:
        random.seed(seed)
        loc = random.randint(0, len(new_sol) - 1)
        if new_sol[loc] != -2:
            r_index = loc // data.num_of_VNF_types
            vnf_type = loc % data.num_of_VNF_types
            break
        seed += 1
    while True:
        random.seed(seed)
        rn = random.randint(-1, data.num_of_nodes - 1)
        if rn != new_sol[loc] and rn != data.s_i[r_index] and rn != data.e_i[r_index]:
            new_sol[loc] = rn
            break
        seed += 1
    overload_nodes = check_capacity(new_sol, data)
    flag = True
    loop_count = 0
    while overload_nodes != []:
        for n in overload_nodes:
            candidates = []
            for i in range(len(new_sol)):
                if new_sol[i] == n[0]:
                    candidates.append(i)
            random.seed(seed)
            buffer = random.sample(candidates, k=1)
            cand = buffer[0]
            vnf_type = cand % data.num_of_VNF_types
            available_nodes = find_available_nodes(new_sol, vnf_type, data)
            if len(available_nodes) > 0:
                random.seed(seed)
                buffer = random.sample(available_nodes, k=1)
                new_sol[cand] = buffer[0]
            else:
                random.seed(seed)
                buffer = random.sample(data.nodes, k=1)
                new_sol[cand] = buffer[0]
        overload_nodes = check_capacity(new_sol, data)
        seed += 1
        loop_count += 1
        if loop_count >= 50:
            flag = False
            break
    return new_sol, flag

def find_available_nodes(new_sol, v_type, data):
    available_nodes = []
    for i in data.nodes:
        vnf_types = [0] * data.num_of_VNF_types
        mem_count = 0
        for j in range(len(new_sol)):
            if new_sol[j] == i:
                tmp = j % data.num_of_VNF_types
                vnf_types[tmp] = 1
        for j in range(len(vnf_types)):
            if vnf_types[j] == 1:
                mem_count += 1
        cpu_count = 0
        for j in range(len(new_sol)):
            if new_sol[j] == i:
                tmp = j % data.num_of_VNF_types
                cpu_count += data.cpu_f[tmp]
        diff_cpu = data.cpu_v[i] - cpu_count
        if vnf_types[v_type] == 1:
            if diff_cpu > data.cpu_f[v_type]: # CPU constraint
                available_nodes.append(i)
        else:
            if mem_count < data.mem_v[i] and diff_cpu > data.cpu_f[v_type]: # Mem and CPU constraint
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
        vnf_types = [0] * data.num_of_VNF_types
        mem_count = 0
        for j in range(len(new_sol)):
            if new_sol[j] == i:
                tmp = j % data.num_of_VNF_types
                vnf_types[tmp] = 1
        for j in range(len(vnf_types)):
            if vnf_types[j] == 1:
                mem_count += 1
        cpu_count = 0
        for j in range(len(new_sol)):
            if new_sol[j] == i:
                vnf_type = j % data.num_of_VNF_types
                cpu_count += data.cpu_f[vnf_type]
        if mem_count > data.mem_v[i]:
            if cpu_count > data.cpu_v[i]:
                overload_nodes.append((i, mem_count - data.mem_v[i], cpu_count - data.cpu_v[i]))
            else:
                overload_nodes.append((i, mem_count - data.mem_v[i], 0))
        elif cpu_count > data.cpu_v[i]:
            overload_nodes.append((i, 0, cpu_count - data.cpu_v[i]))
        
    return overload_nodes

def check_acception(new_sol, data):
    start = -1
    last = -1
    acception = []
    for i in range(data.num_of_requests):
        start = data.num_of_VNF_types * i
        end = start + data.num_of_VNF_types
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

def find_delay_time(sol, data):
    delay_times = []
    for i in range(data.num_of_requests):
        start = i * data.num_of_VNF_types
        end = start + data.num_of_VNF_types
        request = sol[start:end]
        if -1 in request:
            delay_times.append(False)
        else:
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
            delay_times.append(tau_i)
    return delay_times

def main(data_from_cplex, improved_greedy_sol, improved_greedy_res):
    data = data_from_cplex
    seed = 1
    # print("seed:", seed)
    start_time = time.time()

    current_sol = deepcopy(improved_greedy_sol)
    new_sol = []
    current_temperature = 1000
    final_temperature = 0.0001
    cooling_rate = 0.99
    diff = 0
    prob = 0
    current_res = improved_greedy_res

    while current_temperature > final_temperature:
        # print("current_temperature:",  current_temperature)
        new_sol, flag = find_new_solution(current_sol, data, seed)
        if flag:
            acception = check_acception(new_sol, data)
            profit = 0
            for i in range(data.num_of_requests):
                if acception[i]:
                    profit += data.profit_i[i]
            if profit >= current_res:
                current_res = profit
                current_sol = new_sol
            else:
                diff = profit - current_res
                prob = math.exp(diff / current_temperature)
                random.seed(seed)
                if random.uniform(0, 1) < prob:
                    current_res = profit
                    current_sol = new_sol
            current_temperature *= cooling_rate
        seed += 1

    end_time = time.time()
    time_cost = end_time - start_time

    total_profit = 0
    if current_res > improved_greedy_res:
        total_profit = current_res
    else:
        total_profit = improved_greedy_res
        current_sol = improved_greedy_sol

    vnf_on_node = [[] for i in range(data.num_of_nodes)]
    for i in data.nodes:
        for j in range(len(current_sol)):
            if current_sol[j] == i:
                vnf_type = j % data.num_of_VNF_types
                vnf_on_node[i].append(vnf_type)
    shared_count = 0
    vnf_count = 0
    for i in data.nodes:
        for j in vnf_on_node[i]:
            vnf_count += 1
            count = 0
            for k in range(data.num_of_requests):
                if j in data.F_i[k]:
                    loc = data.num_of_VNF_types * k + j
                    if current_sol[loc] == i:
                        count += 1
                        if count > 1:
                            shared_count += 1
                            break
    ratio_of_vnf_shared = shared_count / vnf_count

    acception = check_acception(current_sol, data)
    acc_count = 0
    for i in range(len(acception)):
        if acception[i]:
            acc_count += 1
    acc_rate = acc_count / data.num_of_requests

    delay_times = find_delay_time(current_sol, data)
    print("delay_times: ", delay_times)

    res = {
        "total_profit": total_profit,
        "time_cost": time_cost,
        # "solution": current_sol,
        "acc_rate": acc_rate,
        # "average_delay": average_delay,
        "ratio_of_vnf_shared": ratio_of_vnf_shared
    }
    return res
