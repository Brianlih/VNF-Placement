import time
from copy import deepcopy
import random
import settings

def check_if_meet_delay_requirement(request_assign_node, i, data):
    tau_vnf_i = 0
    tau_i = 0
    first_vnf = -1
    last_vnf = -1
    for vnf_1 in data.F_i[i]:
        for vnf_2 in data.F_i[i]:
            if settings.check_are_neighbors(vnf_1, vnf_2, data.F_i[i]):
                tau_vnf_i += settings.v2v_shortest_path_length(data.G, request_assign_node[i][vnf_1], request_assign_node[i][vnf_2])
            if settings.check_is_first_vnf(vnf_1, data.F_i[i]):
                first_vnf = vnf_1
            if settings.check_is_last_vnf(vnf_1, data.F_i[i]):
                last_vnf = vnf_1
    tau_i += tau_vnf_i
    tau_i += settings.v2v_shortest_path_length(data.G, data.s_i[i], request_assign_node[i][first_vnf])
    tau_i += settings.v2v_shortest_path_length(data.G, data.e_i[i], request_assign_node[i][last_vnf])
        
    if tau_i <= data.r_i[i]:
        return True, tau_i
    return False, tau_i

def calculate_two_phase_length_of_nodes(pre_node, r_index, data):
    two_phases_len = []
    for i in range(len(data.nodes)):
        length = settings.v2v_shortest_path_length(data.G, pre_node, i)
        length += settings.v2v_shortest_path_length(data.G, i, data.e_i[r_index])
        two_phases_len.append(length)
    return two_phases_len

def main(data_from_cplex):
    data = data_from_cplex
    if_considered_to_placed = [False for i in range(data.number_of_requests)]
    total_delay = 0
    start_time = time.time()

    # Initialize decision variables
    buffer_z = [0] * data.number_of_requests
    request_assign_node = [[] for i in range(data.number_of_requests)]
    for i in range(data.number_of_requests):
        for j in range(data.number_of_VNF_types):
            request_assign_node[i].append(-2)

    vnf_on_node = [[] for i in range(data.number_of_nodes)]
    rest_cpu_v = deepcopy(data.cpu_v)
    rest_mem_v = deepcopy(data.mem_v)
    random_request_sequence = random.sample(data.F_i, k=len(data.F_i))

    r_index = -1
    rrs_index = 0
    while rrs_index < data.number_of_requests:
        request = random_request_sequence[rrs_index]
        for i in range(data.number_of_requests):
            if request == data.F_i[i] and if_considered_to_placed[i] == False:
                r_index = i
                if_considered_to_placed[i] = True
                break

        # resources befor placing request
        buffer_cpu = deepcopy(rest_cpu_v)
        buffer_mem = deepcopy(rest_mem_v)
        buffer_vnf_on_node = deepcopy(vnf_on_node)
        buffer_request_assign_node = deepcopy(request_assign_node)
        flag = False
        for vnf_type in request:
            node_sequence = random.sample(data.nodes, k=data.number_of_nodes)
            node_sequence.remove(data.s_i[r_index])
            node_sequence.remove(data.e_i[r_index])
            for node in node_sequence:
                if vnf_type not in buffer_vnf_on_node[node]:
                    if buffer_mem[node] >= 1 and data.cpu_f[vnf_type] <= buffer_cpu[node]:
                        buffer_vnf_on_node[node].append(vnf_type)
                        buffer_mem[node] -= 1
                        buffer_request_assign_node[r_index][vnf_type] = node
                        buffer_cpu[node] -= data.cpu_f[vnf_type]
                        break
                elif data.cpu_f[vnf_type] <= buffer_cpu[node]:
                    if data.cpu_f[vnf_type] <= buffer_cpu[node]:
                        buffer_request_assign_node[r_index][vnf_type] = node
                        buffer_cpu[node] -= data.cpu_f[vnf_type]
                        break
                if node == node_sequence[-1]:
                    flag = True
            if flag:
                # Return to the state before placing request
                buffer_cpu = rest_cpu_v
                buffer_mem = rest_mem_v
                buffer_vnf_on_node = vnf_on_node
                buffer_request_assign_node = request_assign_node
                break
        if not flag:
            meet, delay = check_if_meet_delay_requirement(buffer_request_assign_node, r_index, data)
            if meet:
                rest_cpu_v = buffer_cpu
                rest_mem_v = buffer_mem
                vnf_on_node = buffer_vnf_on_node
                request_assign_node = buffer_request_assign_node
                buffer_z[r_index] = 1
                total_delay += delay
            else:
                # return to the state before placing request
                buffer_cpu = rest_cpu_v
                buffer_mem = rest_mem_v
                buffer_vnf_on_node = vnf_on_node
                buffer_request_assign_node = request_assign_node
        rrs_index += 1

    # print("Random solution: ", request_assign_node)
    end_time = time.time()
    time_cost = end_time - start_time
    total_profit = 0
    acc_count = 0
    for i in range(data.number_of_requests):
        if buffer_z[i] == 1:
            total_profit += data.profit_i[i]
            acc_count += 1
    average_delay = 0
    if acc_count > 0:
        average_delay = total_delay / acc_count
    else:
        average_delay = 0
    acc_rate = acc_count / data.number_of_requests
    res = {
        "total_profit": total_profit,
        "time_cost": time_cost,
        "acc_rate": acc_rate,
        "average_delay": average_delay
    }
    return res
