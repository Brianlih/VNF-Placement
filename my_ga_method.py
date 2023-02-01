import numpy as np
import random
from random import choice
import networkx as nx
import settings
import time
from copy import deepcopy

def check_cap_after_cro_mut(p1, p2, data):
    nodes_p1_cpu = []
    nodes_p2_cpu = []
    nodes_p1_mem = []
    nodes_p2_mem = []

    # CPU constraint
    for i in data.nodes:
        p1_cpu_count = 0
        p2_cpu_count = 0
        for j in range(len(p1)):
            if p1[j] == i:
                vnf_type = j % data.number_of_VNF_types
                p1_cpu_count += data.cpu_f[vnf_type]
            if p2[j] == i:
                vnf_type = j % data.number_of_VNF_types
                p2_cpu_count += data.cpu_f[vnf_type]
        if p1_cpu_count > data.cpu_v[i]:
            nodes_p1_cpu.append(i)
        if p2_cpu_count > data.cpu_v[i]:
            nodes_p2_cpu.append(i)
    
    # Memory constraint
    for i in data.nodes:
        p1_vnf_types = [0] * data.number_of_VNF_types
        p2_vnf_types = [0] * data.number_of_VNF_types
        p1_mem_count = 0
        p2_mem_count = 0
        for j in range(len(p1)):
            if p1[j] == i:
                tmp = j % data.number_of_VNF_types
                p1_vnf_types[tmp] = 1
            if p2[j] == i:
                tmp = j % data.number_of_VNF_types
                p2_vnf_types[tmp] = 1
        for j in range(len(p1_vnf_types)):
            if p1_vnf_types[j] == 1:
                p1_mem_count += 1
        for j in range(len(p2_vnf_types)):
            if p2_vnf_types[j] == 1:
                p2_mem_count += 1
        if p1_mem_count > data.mem_v[i]:
            nodes_p1_mem.append(i)
        if p2_mem_count > data.mem_v[i]:
            nodes_p2_mem.append(i)
        
    return nodes_p1_cpu, nodes_p1_mem, nodes_p2_cpu, nodes_p2_mem

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

    # Initialize population randomly
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

            # resources befor placing F_i[i]
            buffer_cpu = deepcopy(rest_cpu_v)
            buffer_mem = deepcopy(rest_mem_v)
            buffer_vnf_on_node = deepcopy(vnf_on_node)

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
                            if vnf_type not in buffer_vnf_on_node[node]:
                                if buffer_mem[node] >= 1 and data.cpu_f[vnf_type] <= buffer_cpu[node]:
                                    buffer_vnf_on_node[node].append(vnf_type)
                                    buffer_mem[node] -= 1
                                    chromosome[j] = node
                                    buffer_cpu[node] -= data.cpu_f[vnf_type]
                                    assigned_count += 1
                                    break
                            else:
                                if data.cpu_f[vnf_type] <= buffer_cpu[node]:
                                    chromosome[j] = node
                                    buffer_cpu[node] -= data.cpu_f[vnf_type]
                                    assigned_count += 1
                                    break
                    j += 1
                if assigned_count == len(data.F_i[i]):
                    # Update resource state
                    rest_cpu_v = buffer_cpu
                    rest_mem_v = buffer_mem
                    vnf_on_node = buffer_vnf_on_node
                    break
                else:
                    # Return to the state before placing F_i[i]
                    buffer_cpu = rest_cpu_v
                    buffer_mem = rest_mem_v
                    buffer_vnf_on_node = vnf_on_node
                    if len(all_paths_list) > 0:
                        chromosome[start:last + 1] = [-3] * (last + 1 - start)
                        j = start
                    else:
                        # Request(F_i[i]) can not be placed on the network
                        # completely so reject it
                        j = start
                        while j <= last:
                            if chromosome[j] != -2:
                                chromosome[j] = -1
                            j += 1
                        break
        population.append(chromosome)

    # Calculate the fitness value of each individual and
    # sort them in decreasing order
    fitness_of_chromosomes = calculate_fitness_value(population, data)

    sorted_population_index = sorted(
        range(len(fitness_of_chromosomes)),
        key= lambda k: fitness_of_chromosomes[k],
        reverse=True
    )

    fittest = []
    it = 1
    while it <= data.iteration_for_ga:
        # Selection
        elitisms = []
        for i in range(int(data.number_of_individual * data.elitism_rate)):
            elitisms.append(population[sorted_population_index[i]])
        population.extend(elitisms)

        # Crossover & Mutation
        while len(population) < (
            2 * data.number_of_individual
            + int(data.number_of_individual * data.elitism_rate)):
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

            nodes_p1_cpu = []
            nodes_p2_cpu = []
            nodes_p1_mem = []
            nodes_p2_mem = []
            it_cm = 1
            while it_cm <= data.max_iter_cro_mut:
                p1 = deepcopy(population[p1_index])
                p2 = deepcopy(population[p2_index])
                # Crossover
                for i in range(len(p1)):
                    if p1[i] != -2 and p2[i] != -2:
                        if (p1[i] in nodes_p1_cpu or
                            p2[i] in nodes_p2_cpu or
                            p1[i] in nodes_p1_mem or
                            p2[i] in nodes_p2_mem):
                            buffer = p1[i]
                            p1[i] = p2[i]
                            p2[i] = buffer
                            # Check constraints
                            (nodes_p1_cpu,
                            nodes_p2_cpu,
                            nodes_p1_mem,
                            nodes_p2_mem) = check_cap_after_cro_mut(p1, p2, data)
                        else:
                            cr = random.uniform(0, 1)
                            if cr > data.crossover_rate:
                                buffer = p1[i]
                                p1[i] = p2[i]
                                p2[i] = buffer
                                # Check constraints
                                (nodes_p1_cpu,
                                nodes_p2_cpu,
                                nodes_p1_mem,
                                nodes_p2_mem) = check_cap_after_cro_mut(p1, p2, data)

                # Mutation
                for i in range(len(p1)):
                    if p1[i] != -2 and p2[i] != -2:
                        if (p1[i] in nodes_p1_cpu or
                            p2[i] in nodes_p2_cpu or
                            p1[i] in nodes_p1_mem or
                            p2[i] in nodes_p2_mem):
                            while True:
                                rn = random.randint(0, data.number_of_nodes - 1)
                                if rn != p1[i]:
                                    p1[i] = rn
                                    break
                            while True:
                                rn = random.randint(0, data.number_of_nodes - 1)
                                if rn != p2[i]:
                                    p2[i] = rn
                                    break
                            # Check constraints
                            (nodes_p1_cpu,
                            nodes_p2_cpu,
                            nodes_p1_mem,
                            nodes_p2_mem) = check_cap_after_cro_mut(p1, p2, data)
                        else:
                            mutation_R_1 = random.uniform(0, 1)
                            mutation_R_2 = random.uniform(0, 1)
                            if mutation_R_1 > data.mutation_rate:
                                while True:
                                    rn = random.randint(0, data.number_of_nodes - 1)
                                    if rn != p1[i]:
                                        p1[i] = rn
                                        break
                                # Check constraints
                                (nodes_p1_cpu,
                                nodes_p2_cpu,
                                nodes_p1_mem,
                                nodes_p2_mem) = check_cap_after_cro_mut(p1, p2, data)
                            if mutation_R_2 > data.mutation_rate:
                                while True:
                                    rn = random.randint(0, data.number_of_nodes - 1)
                                    if rn != p2[i]:
                                        p2[i] = rn
                                        break
                                # Check constraints
                                (nodes_p1_cpu,
                                nodes_p2_cpu,
                                nodes_p1_mem,
                                nodes_p2_mem) = check_cap_after_cro_mut(p1, p2, data)

                if (len(nodes_p1_cpu) == 0 and
                    len(nodes_p2_cpu) == 0 and
                    len(nodes_p1_mem) == 0 and
                    len(nodes_p2_mem) == 0):
                    population.append(p1)
                    population.append(p2)
                    break
                else:
                    it_cm += 1
            if it_cm > data.max_iter_cro_mut:
                population.append(p1)
                population.append(p2)

        del population[0:data.number_of_individual]
        while len(population) > data.number_of_individual:
            population.pop()

        # Calculate the fitness value of each individual, and sort them in decresing order
        fitness_of_chromosomes = calculate_fitness_value(population, data)

        sorted_population_index = sorted(
            range(len(fitness_of_chromosomes)),
            key= lambda k: fitness_of_chromosomes[k],
            reverse=True
        )

        # Select the fittest individual as the optimal solution for the current generation
        fittest.append(fitness_of_chromosomes[0])
        it += 1
    
    # solution = population[fitness_of_chromosomes.index(fittest[-1])]
    # print("GA solution: ", solution)
    fittest_value = fittest[-1]
    end_time = time.time()
    time_cost = end_time - start_time

    res = {
        "fittest_value": fittest_value,
        "time_cost": time_cost,
    }

    return res
