import random, time, math
from copy import deepcopy
import settings, pre_settings

def find_new_solution(current_sol, data, seed):
    available_nodes = []
    have_been_considered = []
    overload_node = -1
    rn = -1
    new_sol = deepcopy(current_sol)
    vnf_type = -1
    r_index = -1
    loc = -1
    while True:
        random.seed(seed)
        loc = random.randint(0, len(new_sol) - 1)
        if new_sol[loc] != -2:
            r_index = loc // data.num_of_VNF_types
            vnf_type = loc % data.num_of_VNF_types
            break
        seed += 1
    have_been_considered.append(new_sol[loc])
    while True:
        random.seed(seed)
        rn = random.randint(-1, data.num_of_nodes - 1)
        if rn != new_sol[loc] and rn != data.s_i[r_index] and rn != data.e_i[r_index]:
            new_sol[loc] = rn
            break
        seed += 1
    have_been_considered.append(rn)
    overload_node = check_capacity(new_sol, rn, data)
    flag = True
    loop_count = 0
    if overload_node != -1:
        set1 = set(have_been_considered)
        available_nodes = find_available_nodes(new_sol, vnf_type, data)
        set2 = set(available_nodes)
        ava_nodes = list(set2 - set1)
        if len(ava_nodes) > 0:
            random.seed(seed)
            buffer = random.sample(ava_nodes, k=1)
            new_sol[loc] = buffer[0]
            # overload_node = check_capacity(new_sol, new_sol[loc], data)
        else:
            flag = False
            # print("Infisible")
        # else:
        #     random.seed(seed)
        #     buffer = random.sample(data.nodes, k=1)
        #     new_sol[loc] = buffer[0]
        #     have_been_considered.append(buffer[0])
        seed += 1
        # loop_count += 1
        # if loop_count >= 50:
        #     # print("Infisible")
        #     flag = False
        #     break
    return new_sol, r_index, flag

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
    first_vnf = data.F_i[i][0]
    last_vnf = data.F_i[i][-1]
    if len(data.F_i[i]) == 1:
        tau_vnf_i = 0
    else:
        for j in range(len(data.F_i[i]) - 1):
            k = j + 1
            tau_vnf_i += settings.v2v_shortest_path_length(
                data.G,
                request[data.F_i[i][j]],
                request[data.F_i[i][k]])
    tau_i += tau_vnf_i
    tau_i += settings.v2v_shortest_path_length(data.G, data.s_i[i], request[first_vnf])
    tau_i += settings.v2v_shortest_path_length(data.G, data.e_i[i], request[last_vnf])
        
    if tau_i <= data.r_i[i]:
        return True
    return False

def check_capacity(new_sol, node, data):
    overload_node = -1
    vnf_types = [0] * data.num_of_VNF_types
    mem_count = 0
    for j in range(len(new_sol)):
        if new_sol[j] == node:
            tmp = j % data.num_of_VNF_types
            vnf_types[tmp] = 1
    for j in range(len(vnf_types)):
        if vnf_types[j] == 1:
            mem_count += 1
    cpu_count = 0
    for j in range(len(new_sol)):
        if new_sol[j] == node:
            vnf_type = j % data.num_of_VNF_types
            cpu_count += data.cpu_f[vnf_type]

    if mem_count > data.mem_v[node] or cpu_count > data.cpu_v[node]:
        overload_node = node
        
    return overload_node

def check_acception(new_sol, ir, acception, data):
    acc = deepcopy(acception)
    if None in acc:
        for i in range(data.num_of_requests):
            start = data.num_of_VNF_types * i
            end = start + data.num_of_VNF_types
            if -1 in new_sol[start:end]:
                acc[i] = False
            elif check_if_meet_delay_requirement(new_sol[start:end], i, data):
                acc[i] = True
            else:
                acc[i] = False
    else:
        start = data.num_of_VNF_types * ir
        end = start + data.num_of_VNF_types
        if -1 in new_sol[start:end]:
            acc[ir] = False
        elif check_if_meet_delay_requirement(new_sol[start:end], ir, data):
            acc[ir] = True
        else:
            acc[ir] = False
    return acc

# def find_delay_time(sol, data):
#     delay_times = []
#     for i in range(data.num_of_requests):
#         start = i * data.num_of_VNF_types
#         end = start + data.num_of_VNF_types
#         request = sol[start:end]
#         if -1 in request:
#             delay_times.append(False)
#         else:
#             tau_vnf_i = 0
#             tau_i = 0
#             first_vnf = -1
#             last_vnf = -1
#             for vnf_1 in data.F_i[i]:
#                 for vnf_2 in data.F_i[i]:
#                     if settings.check_are_neighbors(vnf_1, vnf_2, data.F_i[i]):
#                         tau_vnf_i += settings.v2v_shortest_path_length(data.G, request[vnf_1], request[vnf_2])
#                     if settings.check_is_first_vnf(vnf_1, data.F_i[i]):
#                         first_vnf = vnf_1
#                     if settings.check_is_last_vnf(vnf_1, data.F_i[i]):
#                         last_vnf = vnf_1
#             tau_i += tau_vnf_i
#             tau_i += settings.v2v_shortest_path_length(data.G, data.s_i[i], request[first_vnf])
#             tau_i += settings.v2v_shortest_path_length(data.G, data.e_i[i], request[last_vnf])
#             delay_times.append(tau_i)
#     return delay_times

def main(data_from_cplex, improved_greedy_sol, improved_greedy_res, s):
    it_count = 0
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
    current_acception = [None for i in range(data.num_of_requests)]
    new_acception = [None for i in range(data.num_of_requests)]
    best_res = 0
    same_res_count = 0
    res_arr = []
    best_arr = []

    while same_res_count <= 10000:
        # it_count += 1
        # print("current_temperature:",  current_temperature)
        new_sol, ir, flag = find_new_solution(current_sol, data, seed)
        if flag:
            new_acception = check_acception(new_sol, ir, current_acception, data)
            profit = 0
            for i in range(data.num_of_requests):
                if new_acception[i]:
                    profit += data.profit_i[i]
            if profit >= current_res:
                if profit > best_res:
                    best_res = profit
                current_res = profit
                current_sol = new_sol
                current_acception = new_acception
                same_res_count = 0
            else:
                diff = profit - current_res
                prob = math.exp(diff / current_temperature)
                random.seed(seed)
                if random.uniform(0, 1) < prob:
                    current_res = profit
                    current_sol = new_sol
                    current_acception = new_acception
                same_res_count += 1
            res_arr.append(current_res)
            best_arr.append(best_res)
            current_temperature *= cooling_rate
        seed += 1

    end_time = time.time()
    time_cost = end_time - start_time

    # total_profit = 0
    # if current_res > improved_greedy_res:
    #     total_profit = current_res
    # else:
    #     total_profit = improved_greedy_res
    #     current_sol = improved_greedy_sol

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
    if vnf_count > 0:
        ratio_of_vnf_shared = shared_count / vnf_count
    else:
        ratio_of_vnf_shared = 0

    # current_res -= vnf_count * pre_settings.cost_f
    # acception = check_acception(current_sol, data)
    acc_count = 0
    for i in range(len(current_acception)):
        if current_acception[i]:
            acc_count += 1
    acc_rate = acc_count / data.num_of_requests

    # delay_times = find_delay_time(current_sol, data)
    # print("delay_times: ", delay_times)

    res = {
        "total_profit": current_res,
        "time_cost": time_cost,
        # "solution": current_sol,
        "acc_rate": acc_rate,
        # "average_delay": average_delay,
        "ratio_of_vnf_shared": ratio_of_vnf_shared,
        "res_arr": res_arr,
        "best_arr": best_arr
    }
    return res
