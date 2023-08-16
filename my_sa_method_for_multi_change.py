import random, time, math
from copy import deepcopy
import settings, pre_settings

def find_new_solution(current_sol, data):
    changed_ratio = [0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.001]
    change_locs = []
    available_nodes = []
    overload_nodes = []
    impacted_nodes = []
    impacted_requests = []
    rn = -1
    new_sol = deepcopy(current_sol)
    vnf_type = -1
    r_index = -1
    loc = -1

    buffer = random.sample(changed_ratio, k=1)
    cr = buffer[0]
    required_length = data.num_of_requests * data.num_of_VNF_types * cr
    while len(change_locs) < required_length:
        while True:
            loc = random.randint(0, len(new_sol) - 1)
            if new_sol[loc] != -2:
                change_locs.append(loc)
                r_index = loc // data.num_of_VNF_types
                vnf_type = loc % data.num_of_VNF_types
                impacted_requests.append(r_index)
                break
        while True:
            rn = random.randint(-1, data.num_of_nodes - 1)
            if rn != new_sol[loc] and rn != data.s_i[r_index] and rn != data.e_i[r_index]:
                new_sol[loc] = rn
                if rn != -1:
                    impacted_nodes.append(rn)
                break
    impacted_nodes = list(set(impacted_nodes))
    impacted_requests = set(impacted_requests)
    overload_nodes = check_capacity(new_sol, impacted_nodes, data)
    flag = True
    loop_count = 0
    while overload_nodes != []:
        for n in overload_nodes:
            candidates = []
            for i in range(len(new_sol)):
                if new_sol[i] == n:
                    candidates.append(i)
            buffer = random.sample(candidates, k=1)
            cand = buffer[0]
            r_index = cand // data.num_of_VNF_types
            vnf_type = cand % data.num_of_VNF_types
            impacted_requests.add(r_index)
            available_nodes = find_available_nodes(new_sol, vnf_type, data)
            if len(available_nodes) > 0:
                buffer = random.sample(available_nodes, k=1)
                new_sol[cand] = buffer[0]
            else:
                rn = random.randint(-1, data.num_of_nodes - 1)
                new_sol[cand] = rn
                if rn != -1:
                    impacted_nodes.append(rn)
        impacted_nodes = list(set(impacted_nodes))
        overload_nodes = check_capacity(new_sol, impacted_nodes, data)
        loop_count += 1
        if loop_count >= 50:
            print("Infisible")
            flag = False
            break
    impacted_requests = list(impacted_requests)
    return new_sol, impacted_requests, flag

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

def check_capacity(new_sol, impacted_nodes, data):
    overload_nodes = []
    for i in impacted_nodes:
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
        
        if mem_count > data.mem_v[i] or cpu_count > data.cpu_v[i]:
            overload_nodes.append(i)
        
    return overload_nodes

def check_acception(new_sol, impacted_requests, acception, data):
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
        for i in impacted_requests:
            start = data.num_of_VNF_types * i
            end = start + data.num_of_VNF_types
            if -1 in new_sol[start:end]:
                acc[i] = False
            elif check_if_meet_delay_requirement(new_sol[start:end], i, data):
                acc[i] = True
            else:
                acc[i] = False
    return acc

def main(data_from_cplex, improved_greedy_sol, improved_greedy_res):
    it_count = 0
    fisible_count = 0
    infisible_count = 0
    data = data_from_cplex
    start_time = time.time()

    current_sol = deepcopy(improved_greedy_sol)
    new_sol = []
    current_temperature = 100
    final_temperature = 0.0001
    cooling_rate = 0.99
    cooling_rate_for_worse_sol = 0.99999
    diff = 0
    prob = 0
    current_res = improved_greedy_res
    current_acception = [None for i in range(data.num_of_requests)]
    new_acception = [None for i in range(data.num_of_requests)]
    best_res = improved_greedy_res
    same_res_count = 0
    res_arr = []
    best_arr = []

    while it_count <= 200000:
        # print("current_temperature:",  current_temperature)
        new_sol, impacted_requests, flag = find_new_solution(current_sol, data)
        if flag:
            it_count += 1
            fisible_count += 1
            new_acception = check_acception(new_sol, impacted_requests, current_acception, data)
            profit = 0
            for i in range(data.num_of_requests):
                if new_acception[i]:
                    profit += data.profit_i[i]
            if profit >= current_res:
                if profit > best_res:
                    best_res = profit
                if profit > current_res:
                    same_res_count = 0
                else:
                    same_res_count += 1
                current_res = profit
                current_sol = new_sol
                current_acception = new_acception
                current_temperature *= cooling_rate
            else:
                diff = profit - current_res
                prob = math.exp(diff / current_temperature)
                if random.uniform(0, 1) < prob:
                    current_res = profit
                    current_sol = new_sol
                    current_acception = new_acception
                same_res_count += 1
                current_temperature *= cooling_rate
            res_arr.append(current_res)
            best_arr.append(best_res)

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

    res = {
        "total_profit": current_res,
        "time_cost": time_cost,
        # "solution": current_sol,
        "acc_rate": acc_rate,
        # "average_delay": average_delay,
        "ratio_of_vnf_shared": ratio_of_vnf_shared,
        "res_arr": res_arr,
        "best_arr": best_arr,
        "fisible_count": fisible_count
    }
    return res
