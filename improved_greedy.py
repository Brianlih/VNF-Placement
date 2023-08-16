import time
from copy import deepcopy
import settings, pre_settings

def check_if_meet_delay_requirement(request_assign_node, i, data):
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
                request_assign_node[i][data.F_i[i][j]],
                request_assign_node[i][data.F_i[i][k]])

    tau_i += tau_vnf_i
    tau_i += settings.v2v_shortest_path_length(
        data.G,
        data.s_i[i],
        request_assign_node[i][first_vnf])
    tau_i += settings.v2v_shortest_path_length(
        data.G,
        data.e_i[i],
        request_assign_node[i][last_vnf])
        
    if tau_i <= data.r_i[i]:
        return True, tau_i
    return False, tau_i

def calculate_requests_needed_cpu(data):
    request_needed_cpu = []
    for i in range(len(data.F_i)):
        cpu_count = 0
        for j in range(len(data.F_i[i])):
            cpu_count += data.cpu_f[data.F_i[i][j]]
        request_needed_cpu.append(cpu_count)
    return request_needed_cpu

def sort_requests(request_needed_cpu, data):
    request_value = []
    max_c = max(request_needed_cpu)
    min_c = min(request_needed_cpu)
    max_p = max(data.profit_i)
    min_p = min(data.profit_i)
    for i in range(len(data.F_i)):
        request_value.append(((data.profit_i[i] - min_p + 1) / (max_p - min_p + 1))
            / ((request_needed_cpu[i] - min_c + 1) / (max_c - min_c + 1))
        )
            
    sorted_requests = sorted(
        data.F_i,
        key= lambda request : request_value[data.F_i.index(request)],
        reverse=True
    )
    return sorted_requests

def calculate_two_phase_length_of_nodes(pre_node, r_index, data):
    two_phases_len = []
    for i in range(len(data.nodes)):
        length = settings.v2v_shortest_path_length(data.G, pre_node, i)
        length += settings.v2v_shortest_path_length(data.G, i, data.e_i[r_index])
        two_phases_len.append(length)
    return two_phases_len

def sort_nodes(rest_cpu_v, r_index, vnf_type, buffer_request_assign_node, data):
    node_value = []
    is_first_vnf = settings.check_is_first_vnf(vnf_type, data.F_i[r_index])
    if is_first_vnf:
        pre_node = data.s_i[r_index]
    else:
        pre_vnf_index = data.F_i[r_index].index(vnf_type) - 1
        pre_vnf = data.F_i[r_index][pre_vnf_index]
        pre_node = buffer_request_assign_node[r_index][pre_vnf]
    two_phases_len = calculate_two_phase_length_of_nodes(pre_node, r_index, data)
    max_tpl = max(two_phases_len)
    min_tpl = min(two_phases_len)
    max_rc = max(rest_cpu_v)
    min_rc = min(rest_cpu_v)
    for i in range(len(data.nodes)):
        node_value.append(
            ((rest_cpu_v[i] - min_rc + 1)/ (max_rc - min_rc + 1))
            / ((two_phases_len[i] - min_tpl + 1)/ (max_tpl - min_tpl + 1))
        )
    sorted_nodes = sorted(data.nodes, key= lambda node : node_value[node], reverse=True)
    return sorted_nodes

def main(data_from_cplex):
    data = data_from_cplex
    if_considered_to_placed = [False for i in range(data.num_of_requests)]
    total_delay = 0
    start_time = time.time()

    # Initialize decision variables
    buffer_z = [0] * data.num_of_requests # z
    vnf_on_node = [[] for i in range(data.num_of_nodes)] # y
    request_assign_node = [[] for i in range(data.num_of_requests)] # x
    for i in range(data.num_of_requests):
        for j in range(data.num_of_VNF_types):
            if j in data.F_i[i]:
                request_assign_node[i].append(-1)
            else:
                request_assign_node[i].append(-2)
    
    request_needed_cpu = calculate_requests_needed_cpu(data)
    sorted_requests = sort_requests(request_needed_cpu, data)
    
    rest_cpu_v = deepcopy(data.cpu_v)
    rest_mem_v = deepcopy(data.mem_v)
    r_index = -1
    sr_index = 0
    while sr_index < len(sorted_requests):
        request = sorted_requests[sr_index]
        for i in range(len(data.F_i)):
            if request == data.F_i[i] and if_considered_to_placed[i] == False:
                r_index = i
                if_considered_to_placed[i] = True
                break

        # Resources befor placing request
        buffer_cpu = deepcopy(rest_cpu_v)
        buffer_mem = deepcopy(rest_mem_v)
        buffer_vnf_on_node = deepcopy(vnf_on_node)
        buffer_request_assign_node = deepcopy(request_assign_node)

        # Greedy
        flag = False
        for vnf_type in request:
            sorted_nodes = sort_nodes(buffer_cpu, r_index, vnf_type, buffer_request_assign_node, data)
            sorted_nodes.remove(data.s_i[r_index])
            sorted_nodes.remove(data.e_i[r_index])
            for node in sorted_nodes:
                if vnf_type not in buffer_vnf_on_node[node]:
                    if buffer_mem[node] >= 1 and data.cpu_f[vnf_type] <= buffer_cpu[node]:
                        buffer_vnf_on_node[node].append(vnf_type)
                        buffer_mem[node] -= 1
                        buffer_request_assign_node[r_index][vnf_type] = node
                        buffer_cpu[node] -= data.cpu_f[vnf_type]
                        break
                elif data.cpu_f[vnf_type] <= buffer_cpu[node]:
                    buffer_request_assign_node[r_index][vnf_type] = node
                    buffer_cpu[node] -= data.cpu_f[vnf_type]
                    break
                if node == sorted_nodes[-1]:
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
                # Return to the state before placing request
                buffer_cpu = rest_cpu_v
                buffer_mem = rest_mem_v
                buffer_vnf_on_node = vnf_on_node
                buffer_request_assign_node = request_assign_node
        sr_index += 1

    end_time = time.time()
    time_cost = end_time - start_time

    greedy_solution = []
    for i in range(data.num_of_requests):
        greedy_solution.extend(request_assign_node[i])

    total_profit = 0
    acc_count = 0
    for i in range(data.num_of_requests):
        if buffer_z[i] == 1:
            total_profit += data.profit_i[i]
            acc_count += 1
    acc_rate = acc_count / data.num_of_requests

    res = {
        "total_profit": total_profit,
        "time_cost": time_cost,
        "solution": greedy_solution,
        "acc_rate": acc_rate,
    }
    return res