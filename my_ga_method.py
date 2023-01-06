import numpy as np
import random
from random import choice
import networkx as nx
import settings
# from test import settings
import time
from copy import deepcopy

def check_if_meet_cpu_capacity_constraint(chromosome, data):
    for i in data.nodes:
        cpu_count = 0
        for j in range(len(chromosome)):
            if chromosome[j] == i:
                vnf_type = j % data.number_of_VNF_types
                cpu_count += data.cpu_f[vnf_type]
        if cpu_count > data.cpu_v[i]:
            return False
    return True

def check_if_meet_mem_capacity_constraint(chromosome, data):
    for i in data.nodes:
        vnf_types = [0] * data.number_of_VNF_types
        mem_count = 0
        for j in range(len(chromosome)):
            if chromosome[j] == i:
                tmp = j % data.number_of_VNF_types
                vnf_types[tmp] = 1
        for j in range(len(vnf_types)):
            if vnf_types[j] == 1:
                mem_count += 1
        if mem_count > data.mem_v[i]:
            return False
    return True

def check_if_meet_delay_requirement(request, i, data):
    tau_vnf_i = 0
    tau_i = 0
    first_vnf = -1
    last_vnf = -1
    for vnf_1 in data.F_i[i]:
        for vnf_2 in data.F_i[i]:
            if settings.check_are_neighbors(vnf_1, vnf_2, data.F_i[i]):
                tau_vnf_i += settings.v2v_shortest_path_length(data.G, request[vnf_1], request[vnf_2])
            if settings.check_is_first_vnf(vnf_1, data.F_i[i]):
                first_vnf = vnf_1
            if settings.check_is_last_vnf(vnf_1, data.F_i[i]):
                last_vnf = vnf_1
    tau_i += tau_vnf_i
    if request[first_vnf] != data.s_i[i]:
        tau_i += settings.v2v_shortest_path_length(data.G, data.s_i[i], request[first_vnf])
    if request[last_vnf] != data.e_i[i]:
        tau_i += settings.v2v_shortest_path_length(data.G, data.e_i[i], request[last_vnf])
        
    if tau_i <= data.r_i[i]:
        return True
    return False

def calculate_fitness_value(population, data):
    fitness_of_chromosomes = [0] * len(population)
    for k in range(len(population)):
        j = 0
        while j < len(population[k]):
            if -1 not in population[k][j:j + data.number_of_VNF_types]:
                i = j // data.number_of_VNF_types
                if check_if_meet_delay_requirement(population[k][j:j + data.number_of_VNF_types], i, data):
                    fitness_of_chromosomes[k] += data.profit_i[i]
            j += data.number_of_VNF_types
    return fitness_of_chromosomes

def main(data_from_cplex):
    data = data_from_cplex
    start_time = time.time()
    #------------------------------------------------------------------------------------------
    # Initialize population randomly
    #------------------------------------------------------------------------------------------

    population = []
    request_list = [r for r in range(data.number_of_requests)]
    for p in range(data.number_of_individual):
        chromosome = [-3] * data.number_of_gene_in_an_individual
        assign_sequence = random.sample(request_list, k=data.number_of_requests)
        vnf_on_node = [[] for i in range(data.number_of_nodes)]
        rest_cpu_v = deepcopy(data.cpu_v)
        rest_mem_v = deepcopy(data.mem_v)
        for i in assign_sequence:
            all_paths = nx.all_simple_paths(data.G, source=data.s_i[i], target=data.e_i[i])
            all_paths_list = list(all_paths)
            j = i * data.number_of_VNF_types
            start = j
            last = j + data.number_of_VNF_types - 1
            buffer_cpu = rest_cpu_v # cpu_v befor placing F_i[i]
            buffer_mem = rest_mem_v
            buffer_vnf_on_ndoe = vnf_on_node
            while True:
                assigned_count = 0
                path = all_paths_list[random.randint(0, len(all_paths_list) - 1)]
                all_paths_list.remove(path)
                while(j <= last):
                    vnf_type = j % data.number_of_VNF_types
                    if vnf_type not in data.F_i[i]:
                        chromosome[j] = -2
                    else:
                        for node in path:
                            if data.cpu_f[vnf_type] <= rest_cpu_v[node]:
                                if vnf_type not in vnf_on_node[node]:
                                    if rest_mem_v[node] < 1:
                                        continue
                                    else:
                                        vnf_on_node[node].append(vnf_type)
                                        rest_mem_v[node] -= 1
                                chromosome[j] = node
                                rest_cpu_v[node] -= data.cpu_f[vnf_type]
                                assigned_count += 1
                                break
                    j += 1
                if assigned_count == len(data.F_i[i]):
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
    # Calculate the fitness value of each individual, sort them in decresing order
    #------------------------------------------------------------------------------------------

    fitness_of_chromosomes = calculate_fitness_value(population, data)

    sorted_population_index = sorted(
        range(len(fitness_of_chromosomes)),
        key= lambda k: fitness_of_chromosomes[k],
        reverse=True
    )

    fittest = []
    it = 1
    while it <= data.iteration_for_one_ga:
        # print("iteration ", it)
        #------------------------------------------------------------------------------------------
        # Selection
        #------------------------------------------------------------------------------------------

        elitisms = []
        for i in range(int(data.number_of_individual * data.elitism_rate)):
            elitisms.append(population[sorted_population_index[i]])
        population.extend(elitisms)

        #------------------------------------------------------------------------------------------
        # Crossover & Mutation
        #------------------------------------------------------------------------------------------

        while len(population) < 2 * data.number_of_individual + int(data.number_of_individual * data.elitism_rate):
            tournament_set = random.sample(
                sorted_population_index,
                k=data.number_of_individual_chose_from_population_for_tournament
            )
            p1_index = data.number_of_individual
            for i in range(len(tournament_set)):
                if sorted_population_index.index(tournament_set[i]) < p1_index:
                    p1_index = sorted_population_index.index(tournament_set[i])
            tournament_set = random.sample(
                sorted_population_index,
                k=data.number_of_individual_chose_from_population_for_tournament
            )
            p2_index = data.number_of_individual
            for i in range(len(tournament_set)):
                if sorted_population_index.index(tournament_set[i]) < p2_index:
                    p2_index = sorted_population_index.index(tournament_set[i])

            it_crossover = 1
            while it_crossover <= data.maximum_of_iteration_for_one_ga_crossover:
                p1 = deepcopy(population[p1_index])
                p2 = deepcopy(population[p2_index])
                for i in range(len(p1)):
                    if p1[i] != -2 and p2[i] != -2:
                        crossover_R = random.uniform(0, 1)
                        if crossover_R > data.crossover_rate:
                            # crossover
                            buffer = p1[i]
                            p1[i] = p2[i]
                            p2[i] = buffer
                if (check_if_meet_cpu_capacity_constraint(p1, data) and
                    check_if_meet_cpu_capacity_constraint(p2, data) and
                    check_if_meet_mem_capacity_constraint(p1, data) and
                    check_if_meet_mem_capacity_constraint(p2, data)):
                    break
                it_crossover += 1

            if it_crossover > data.maximum_of_iteration_for_one_ga_crossover:
                p1 = population[p1_index]
                p2 = population[p2_index]

            it_mutation = 1
            while it_mutation <= data.maximum_of_iteration_for_one_ga_mutation:
                p11 = deepcopy(p1)
                p22 = deepcopy(p2)
                for i in range(len(p11)):
                    if p11[i] != -2 and p22[i] != -2:
                        mutation_R_11 = random.uniform(0, 1)
                        mutation_R_22 = random.uniform(0, 1)
                        if mutation_R_11 > data.mutation_rate:
                            # mutation
                            while True:
                                rn = random.randint(0, data.number_of_nodes - 1)
                                if rn != p11[i]:
                                    p11[i] = rn
                                    break
                        if mutation_R_22 > data.mutation_rate:
                            # mutation
                            while True:
                                rn = random.randint(0, data.number_of_nodes - 1)
                                if rn != p22[i]:
                                    p22[i] = rn
                                    break
                if (check_if_meet_cpu_capacity_constraint(p11, data) and
                    check_if_meet_cpu_capacity_constraint(p22, data) and
                    check_if_meet_mem_capacity_constraint(p11, data) and
                    check_if_meet_mem_capacity_constraint(p22, data)):
                    population.append(p11)
                    population.append(p22)
                    break
                it_mutation += 1

            if it_mutation > data.maximum_of_iteration_for_one_ga_mutation:
                population.append(p1)
                population.append(p2)

        del population[0:data.number_of_individual]
        while len(population) > data.number_of_individual:
            population.pop()

        #------------------------------------------------------------------------------------------
        # Calculate the fitness value of each individual, and sort them in decresing order
        #------------------------------------------------------------------------------------------

        fitness_of_chromosomes = calculate_fitness_value(population, data)

        sorted_population_index = sorted(
            range(len(fitness_of_chromosomes)),
            key= lambda k: fitness_of_chromosomes[k],
            reverse=True
        )

        #------------------------------------------------------------------------------------------
        # Select the fittest individual as the optimal solution for the current generation
        #------------------------------------------------------------------------------------------

        fittest.append(fitness_of_chromosomes[0])
        it += 1
    
    # solution = population[fitness_of_chromosomes.index(fittest[-1])]
    fittest_value = fittest[-1]
    end_time = time.time()
    time_cost = end_time - start_time

    res = {
        "fittest_value": fittest_value,
        "time_cost": time_cost,
    }

    return res
