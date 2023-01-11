import time
from copy import deepcopy
import networkx as nx
import random
import settings

def check_if_meet_delay_requirement(request_assign_node, i, data):
    if -2 in request_assign_node[i]:
        return False
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
    if request_assign_node[i][first_vnf] != data.s_i[i]:
        tau_i += settings.v2v_shortest_path_length(data.G, data.s_i[i], request_assign_node[i][first_vnf])
    if request_assign_node[i][last_vnf] != data.e_i[i]:
        tau_i += settings.v2v_shortest_path_length(data.G, data.e_i[i], request_assign_node[i][last_vnf])
        
    if tau_i <= data.r_i[i]:
        return True
    return False

def main(data_from_cplex):
    data = data_from_cplex
    start_time = time.time()
    buffer_z = [0] * data.number_of_requests
    request_assign_node = [[] for i in range(data.number_of_requests)]
    for i in range(data.number_of_requests):
        for j in range(data.number_of_VNF_types):
            request_assign_node[i].append(-2)

    vnf_on_node = [[] for i in range(data.number_of_nodes)]
    rest_cpu_v = data.cpu_v
    rest_mem_v = data.mem_v
    buffer_F_i = deepcopy(data.F_i)
    while len(buffer_F_i) > 0:
        # randomly select a request
        request = random.choice(buffer_F_i)
        buffer_F_i.remove(request)
        r_index = data.F_i.index(request)

        # find all path
        all_paths = nx.all_simple_paths(data.G, source=data.s_i[r_index], target=data.e_i[r_index])
        all_paths_list = list(all_paths)

        # resources befor placing request
        buffer_cpu = deepcopy(rest_cpu_v)
        buffer_mem = deepcopy(rest_mem_v)
        buffer_vnf_on_node = deepcopy(vnf_on_node)
        buffer_request_assign_node = deepcopy(request_assign_node)
        while True:
            assigned_count = 0
            
            # randomly select a path
            path = random.choice(all_paths_list)
            all_paths_list.remove(path)

            for vnf_type in range(data.number_of_VNF_types):
                if vnf_type not in request:
                    buffer_request_assign_node[r_index][vnf_type] = -1
                else:
                    for node in path:
                        if vnf_type not in buffer_vnf_on_node[node]:
                            if buffer_mem[node] >= 1:
                                buffer_vnf_on_node[node].append(vnf_type)
                                buffer_mem[node] -= 1
                                buffer_request_assign_node[r_index][vnf_type] = node
                                buffer_cpu[node] -= data.cpu_f[vnf_type]
                                assigned_count += 1
                                break
                        else:
                            if data.cpu_f[vnf_type] <= buffer_cpu[node]:
                                buffer_request_assign_node[r_index][vnf_type] = node
                                buffer_cpu[node] -= data.cpu_f[vnf_type]
                                assigned_count += 1
                                break
            if (check_if_meet_delay_requirement(buffer_request_assign_node, r_index, data) and
                assigned_count == len(request)):
                rest_cpu_v = buffer_cpu
                rest_mem_v = buffer_mem
                vnf_on_node = buffer_vnf_on_node
                request_assign_node = buffer_request_assign_node
                buffer_z[r_index] = 1
                break
            else:
                # return to the state before placing request
                buffer_cpu = rest_cpu_v
                buffer_mem = rest_mem_v
                buffer_vnf_on_node = vnf_on_node
                buffer_request_assign_node = request_assign_node
                if len(all_paths_list) <= 0:
                    # request can not be placed on the network completely
                    # so reject it
                    break
    # print("Random solution: ", request_assign_node)
    end_time = time.time()
    time_cost = end_time - start_time
    total_profit = 0
    for i in range(data.number_of_requests):
        if buffer_z[i] == 1:
            total_profit += data.profit_i[i]
    res = {
        "total_profit": total_profit,
        "time_cost": time_cost
    }
    return res
