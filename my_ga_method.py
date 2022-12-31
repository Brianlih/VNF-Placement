import numpy as np
import random
import networkx as nx
import settings
import time

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
    placement_sequence = random.sample(request_list, settings.number_of_requests)
    rest_cpu_v = settings.cpu_v
    rest_mem_v = settings.mem_v
    for i in placement_sequence:
        all_qualified_paths = nx.all_simple_paths(settings.G, source=settings.s_i[i], target=settings.e_i[i])
        all_qualified_paths_list = list(all_qualified_paths)
        j = i * settings.number_of_VNF_types
        start = j
        last = j + settings.number_of_VNF_types - 1
        buffer = rest_cpu_v # cpu_v befor placing F_i[i]
        placed_count = 0
        while True:
            path = all_qualified_paths_list[random.randint(0, len(list(all_qualified_paths)))]
            all_qualified_paths_list.remove(path)
            while(j <= last):
                vnf_type = j % settings.number_of_VNF_types
                if vnf_type not in settings.F_i[i]:
                    chromosome[j] = -2
                else:
                    for node in path:
                        if settings.cpu_f[vnf_type] <= rest_cpu_v[node]:
                            chromosome[j] = node
                            rest_cpu_v[node] -= settings.cpu_f[vnf_type]
                            placed_count += 1
                            break
                j += 1
            if placed_count == len(settings.F_i[i]):
                break
            else:
                if len(all_qualified_paths_list) > 0:
                    chromosome[start:last + 1] = [-3] * (last + 1 - start)
                    rest_cpu_v = buffer # return to the state before placing F_i[i]
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
# print(population[0])

#------------------------------------------------------------------------------------------
# Calculate the fitness value of each individual and sort them in decresing order
#------------------------------------------------------------------------------------------


                    
end_time = time.time()
time_cost = end_time - start_time
print("time_cost: ",time_cost)
    
# crossover
# mutation
# select the fittest individual as the optimal solution for the current generation
# termination condition
