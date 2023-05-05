import time
from copy import deepcopy
import settings

def check_if_meet_delay_requirement(request_assign_node, i, data):
    tau_vnf_i = 0
    tau_i = 0
    first_vnf = -1
    last_vnf = -1
    for vnf_1 in data.F_i[i]:
        for vnf_2 in data.F_i[i]:
            if settings.check_are_neighbors(vnf_1, vnf_2, data.F_i[i]):
                tau_vnf_i += settings.v2v_shortest_path_length(
                    data.G,
                    request_assign_node[i][vnf_1],
                    request_assign_node[i][vnf_2])
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

def sort_requests(data):
    request_value = []
    for i in range(len(data.F_i)):
        request_value.append(data.profit_i[i])
            
    sorted_requests = sorted(
        data.F_i,
        key= lambda request : request_value[data.F_i.index(request)],
        reverse=True
    )
    
    return sorted_requests

def sort_nodes(pre_node, data):
    node_value = []
    for i in range(len(data.nodes)):
        node_value.append(settings.v2v_shortest_path_length(data.G, pre_node, i))
    sorted_nodes = sorted(data.nodes, key= lambda node : node_value[node])
    
    return sorted_nodes

def main(data_from_cplex):
    data = data_from_cplex
    if_considered_to_placed = [False for i in range(data.number_of_requests)]
    total_delay = 0
    start_time = time.time()

    # Initialize decision variables
    buffer_z = [0] * data.number_of_requests # z
    vnf_on_node = [[] for i in range(data.number_of_nodes)] # y
    request_assign_node = [[] for i in range(data.number_of_requests)] # x
    for i in range(data.number_of_requests):
        for j in range(data.number_of_VNF_types):
            request_assign_node[i].append(-2)
    
    sorted_requests = sort_requests(data)
    
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
            if request.index(vnf_type) == 0:
                pre_node = data.s_i[r_index]
            else:
                pre_node = buffer_request_assign_node[r_index][request[request.index(vnf_type) - 1]]
            sorted_nodes = sort_nodes(pre_node, data)
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

    # print("greedy solution: ", request_assign_node)
    end_time = time.time()
    time_cost = end_time - start_time

    greedy_solution = []
    for i in range(data.number_of_requests):
        greedy_solution.extend(request_assign_node[i])

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
        "solution": greedy_solution,
        "acc_rate": acc_rate,
        "average_delay": average_delay
    }
    # print("acc_count: ", acc_count)
    return res