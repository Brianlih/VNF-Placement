import time
from copy import deepcopy
import networkx as nx
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

def sort_candidates(candidates, pre_node, graph):
    distance_to_prenode = []
    
    for node in candidates:
        distance_to_prenode.append(settings.v2v_shortest_path_length(graph, pre_node, node))

    sorted_candidates = sorted(candidates, key= lambda node : distance_to_prenode[candidates.index(node)])
    return sorted_candidates

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
            request_assign_node[i].append(-2)
    
    sorted_requests = sorted(
        data.F_i,
        key= lambda request : len(request),
        reverse=True
    )

    rest_cpu_v = deepcopy(data.cpu_v)
    rest_mem_v = deepcopy(data.mem_v)
    r_index = -1
    pre_node = -1
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
        
        for vnf_type in request:
            mapped = False
            candidates = []
            if request.index(vnf_type) == 0:
                pre_node = data.s_i[r_index]
            else:
                pre_node = buffer_request_assign_node[r_index][request[request.index(vnf_type) - 1]]
            
            for i in range(len(buffer_vnf_on_node)):
                if vnf_type in buffer_vnf_on_node[i]:
                    candidates.append(i)
            
            sorted_cnadidates = sort_candidates(candidates, pre_node, data.G)
            for node in sorted_cnadidates:
                if data.cpu_f[vnf_type] <= buffer_cpu[node]:
                    buffer_request_assign_node[r_index][vnf_type] = node
                    buffer_cpu[node] -= data.cpu_f[vnf_type]
                    pre_node = node
                    mapped = True
                    break
            if mapped == False:
                edges = list(nx.bfs_edges(data.G, pre_node))
                nodes = [pre_node] + [v for u, v in edges]
                nodes_for_new_vnf = [node for node in nodes if node not in sorted_cnadidates]
                for node in nodes_for_new_vnf:
                    if buffer_mem[node] >= 1 and data.cpu_f[vnf_type] <= buffer_cpu[node]:
                        buffer_vnf_on_node[node].append(vnf_type)
                        buffer_mem[node] -= 1
                        buffer_request_assign_node[r_index][vnf_type] = node
                        buffer_cpu[node] -= data.cpu_f[vnf_type]
                        pre_node = node
                        mapped = True
                        break
            if mapped == False:
                # Return to the state before placing request
                buffer_cpu = rest_cpu_v
                buffer_mem = rest_mem_v
                buffer_vnf_on_node = vnf_on_node
                buffer_request_assign_node = request_assign_node
                break
        if mapped == True:
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

    hgreedy_solution = []
    for i in range(data.num_of_requests):
        hgreedy_solution.extend(request_assign_node[i])

    vnf_on_node = [[] for i in range(data.num_of_nodes)]
    for i in data.nodes:
        for j in range(len(hgreedy_solution)):
            if hgreedy_solution[j] == i:
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
                    if hgreedy_solution[loc] == i:
                        count += 1
                        if count > 1:
                            shared_count += 1
                            break
    if vnf_count > 0:
        ratio_of_vnf_shared = shared_count / vnf_count
    else:
        ratio_of_vnf_shared = 0

    total_profit = 0
    acc_count = 0
    for i in range(data.num_of_requests):
        if buffer_z[i] == 1:
            total_profit += data.profit_i[i]
            acc_count += 1
    # total_profit -= vnf_count * pre_settings.cost_f
    average_delay = 0
    if acc_count > 0:
        average_delay = total_delay / acc_count
    else:
        average_delay = 0
    acc_rate = acc_count / data.num_of_requests
    res = {
        "total_profit": total_profit,
        "time_cost": time_cost,
        "acc_rate": acc_rate,
        "average_delay": average_delay,
        "ratio_of_vnf_shared": ratio_of_vnf_shared
    }
    # print("acc_count: ", acc_count)
    return res
                    
