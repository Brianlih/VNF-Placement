import numpy as np
import random
import networkx as nx
import settings
import time
from copy import deepcopy

def check_if_meet_cpu_capacity_constraint(chromosome):
    print("check cpu")
    flag = True
    for i in settings.nodes:
        print(i)
        cpu_count = 0
        for j in range(len(chromosome)):
            if chromosome[j] == i:
                tmp = (j + 1) % settings.number_of_VNF_types
                if tmp == 0:
                    cpu_count += settings.cpu_f[settings.number_of_VNF_types - 1]
                else:
                    cpu_count += settings.cpu_f[tmp - 1]
        if cpu_count > settings.cpu_v[i]:
            flag = False
            break
    return flag
def check_if_meet_mem_capacity_constraint(chromosome):
    print("check mem")
    flag = True
    for i in settings.nodes:
        print(i)
        vnf_types = [0] * settings.number_of_VNF_types
        mem_count = 0
        for j in range(len(chromosome)):
            if chromosome[j] == i:
                tmp = (j + 1) % settings.number_of_VNF_types
                if tmp == 0:
                    vnf_types[settings.number_of_VNF_types - 1] = 1
                else:
                    vnf_types[tmp - 1] = 1
        for j in range(len(vnf_types)):
            if vnf_types[j] == 1:
                mem_count += 1
        if mem_count > settings.mem_v:
            flag = False
            break
    return flag

def check_if_meet_delay_requirement(request, i):
    tau_vnf_i = 0
    tau_i = 0
    for vnf_1 in settings.F_i[i]:
        for vnf_2 in settings.F_i[i]:
            if settings.check_are_neighbors(vnf_1, vnf_2, settings.F_i[i]):
                tau_vnf_i += settings.v2v_shortest_path_length(settings.G, request[vnf_1], request[vnf_2])
    tau_i += tau_vnf_i + settings.v2v_shortest_path_length(settings.G, settings.s_i[i], request[vnf_1]) + settings.v2v_shortest_path_length(settings.G, settings.e_i[i], request[vnf_2])
    if tau_i <= settings.r_i[i]:
        return True
    return False

#------------------------------------------------------------------------------------------
# Initialize the input data
#------------------------------------------------------------------------------------------

start_time = time.time()
settings.init()

#------------------------------------------------------------------------------------------
# Initialize population randomly
#------------------------------------------------------------------------------------------

population = []
request_list = [r for r in range(settings.number_of_requests)]
for p in range(settings.number_of_individual):
    chromosome = [-3] * settings.number_of_gene_in_an_individual
    placement_sequence = random.sample(request_list, k=settings.number_of_requests)
    vnf_on_node = [[] for i in range(settings.number_of_nodes)]
    rest_cpu_v = deepcopy(settings.cpu_v)
    rest_mem_v = deepcopy(settings.mem_v)
    for i in placement_sequence:
        all_paths = nx.all_simple_paths(settings.G, source=settings.s_i[i], target=settings.e_i[i])
        all_paths_list = list(all_paths)
        j = i * settings.number_of_VNF_types
        start = j
        last = j + settings.number_of_VNF_types - 1
        buffer_cpu = rest_cpu_v # cpu_v befor placing F_i[i]
        buffer_mem = rest_mem_v
        buffer_vnf_on_ndoe = vnf_on_node
        while True:
            assigned_count = 0
            path = all_paths_list[random.randint(0, len(all_paths_list) - 1)]
            all_paths_list.remove(path)
            while(j <= last):
                vnf_type = j % settings.number_of_VNF_types
                if vnf_type not in settings.F_i[i]:
                    chromosome[j] = -2
                else:
                    for node in path:
                        if settings.cpu_f[vnf_type] <= rest_cpu_v[node]:
                            if vnf_type not in vnf_on_node[node]:
                                if rest_mem_v[node] < 1:
                                    continue
                                else:
                                    vnf_on_node[node].append(vnf_type)
                                    rest_mem_v[node] -= 1
                            chromosome[j] = node
                            rest_cpu_v[node] -= settings.cpu_f[vnf_type]
                            assigned_count += 1
                            break
                j += 1
            if assigned_count == len(settings.F_i[i]):
                break
            else:
                if len(all_paths_list) > 0:
                    # return to the state before placing F_i[i]
                    chromosome[start:last + 1] = [-3] * (last + 1 - start)
                    rest_cpu_v = buffer_cpu
                    rest_mem_v = buffer_mem
                    vnf_on_node = buffer_vnf_on_ndoe
                    j = start
                else:
                    # F_i[i] can not be placed on the network completely
                    j = start
                    while j <= last:
                        if chromosome[j] != -2:
                            chromosome[j] = -1
                        j += 1
                    break
    population.append(chromosome)

#------------------------------------------------------------------------------------------
# Calculate the fitness value of each individual and sort them in decresing order
#------------------------------------------------------------------------------------------

fitness_of_chromosomes = [0] * len(population)
for k in range(len(population)):
    j = 0
    while j < len(population[k]):
        if -1 not in population[k][j:j + settings.number_of_VNF_types]:
            i = j // settings.number_of_VNF_types
            if check_if_meet_delay_requirement(population[k][j:j + settings.number_of_VNF_types], i):
                fitness_of_chromosomes[k] += settings.profit_i[i]
        j += settings.number_of_VNF_types
print("fitness_of_chromosomes: ", fitness_of_chromosomes)

                    
end_time = time.time()
time_cost = end_time - start_time
print("time_cost: ",time_cost)
    
# crossover
# mutation
# select the fittest individual as the optimal solution for the current generation
# termination condition
