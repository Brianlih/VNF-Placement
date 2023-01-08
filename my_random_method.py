import time
from copy import deepcopy
import networkx as nx
import random
import settings

def check_if_meet_delay_requirement(request_assign_node, i):
    if -2 in request_assign_node[i]:
        return False
    tau_vnf_i = 0
    tau_i = 0
    first_vnf = -1
    last_vnf = -1
    for vnf_1 in settings.F_i[i]:
        for vnf_2 in settings.F_i[i]:
            if settings.check_are_neighbors(vnf_1, vnf_2, settings.F_i[i]):
                tau_vnf_i += settings.v2v_shortest_path_length(settings.G, request_assign_node[i][vnf_1], request_assign_node[i][vnf_2])
            if settings.check_is_first_vnf(vnf_1, settings.F_i[i]):
                first_vnf = vnf_1
            if settings.check_is_last_vnf(vnf_1, settings.F_i[i]):
                last_vnf = vnf_1
    tau_i += tau_vnf_i
    if request_assign_node[i][first_vnf] != settings.s_i[i]:
        tau_i += settings.v2v_shortest_path_length(settings.G, settings.s_i[i], request_assign_node[i][first_vnf])
    if request_assign_node[i][last_vnf] != settings.e_i[i]:
        tau_i += settings.v2v_shortest_path_length(settings.G, settings.e_i[i], request_assign_node[i][last_vnf])
        
    if tau_i <= settings.r_i[i]:
        return True
    return False

if __name__ == "__main__":
    # number_of_requests = [15]
    number_of_requests = [15, 20, 25, 30, 35, 40, 45, 50, 55]
    number_of_VNF_types = [5]
    # number_of_VNF_types = [5, 10, 15, 20, 25, 30, 35, 40, 45]

    settings.init(number_of_requests[0], number_of_VNF_types[0])
    start_time = time.time()
    buffer_z = [1] * number_of_requests[0]
    request_assign_node = [[] for i in range(number_of_requests[0])]
    for i in range(len(request_assign_node)):
        for j in range(number_of_VNF_types[0]):
            request_assign_node[i].append(-2)

    vnf_on_node = [[] for i in range(settings.number_of_nodes)]
    rest_cpu_v = deepcopy(settings.cpu_v)
    rest_mem_v = deepcopy(settings.mem_v)
    buffer_F_i = deepcopy(settings.F_i)
    while len(buffer_F_i) > 0:
        # randomly select a request
        request = random.choice(buffer_F_i)
        buffer_F_i.remove(request)
        r_index = settings.F_i.index(request)

        # find all path
        all_paths = nx.all_simple_paths(settings.G, source=settings.s_i[r_index], target=settings.e_i[r_index])
        all_paths_list = list(all_paths)

        # resources befor placing request
        buffer_cpu = rest_cpu_v
        buffer_mem = rest_mem_v
        buffer_vnf_on_node = vnf_on_node

        while True:
            assigned_count = 0
            
            # randomly select a path
            path = random.choice(all_paths_list)
            all_paths_list.remove(path)

            for vnf_type in range(number_of_VNF_types[0]):
                if vnf_type not in request:
                    request_assign_node[r_index][vnf_type] = -1
                else:
                    for node in path:
                        if settings.cpu_f[vnf_type] <= rest_cpu_v[node]:
                            if vnf_type not in vnf_on_node[node]:
                                if rest_mem_v[node] > 1:
                                    vnf_on_node[node].append(vnf_type)
                                    rest_mem_v[node] -= 1
                            request_assign_node[r_index][vnf_type] = node
                            rest_cpu_v[node] -= settings.cpu_f[vnf_type]
                            assigned_count += 1
                            break
            if (check_if_meet_delay_requirement(request_assign_node, r_index) and
                assigned_count == len(request)):
                break
            else:
                # return to the state before placing request
                rest_cpu_v = buffer_cpu
                rest_mem_v = buffer_mem
                vnf_on_node = buffer_vnf_on_node
                if len(all_paths_list) <= 0:
                    # request can not be placed on the network completely
                    # so reject it
                    buffer_z[r_index] = 0
                    break
