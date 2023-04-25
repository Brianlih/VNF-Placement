import random, time
from copy import deepcopy
import settings

def check_cap_after_cro_mut(p1, p2, data):
    p1_overload_nodes = []
    p2_overload_nodes = []
    p1_available_nodes = []
    p2_available_nodes = []

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
        if p1_cpu_count > data.cpu_v[i] and i not in p1_overload_nodes:
            p1_overload_nodes.append(i)
        elif p1_cpu_count < data.cpu_v[i] and i not in p1_available_nodes:
            p1_available_nodes.append(i)
        if p2_cpu_count > data.cpu_v[i] and i not in p2_overload_nodes:
            p2_overload_nodes.append(i)
        elif p2_cpu_count < data.cpu_v[i] and i not in p2_available_nodes:
            p2_available_nodes.append(i)
    
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
        if p1_mem_count > data.mem_v[i] and i not in p1_overload_nodes:
            p1_overload_nodes.append(i)
        elif p1_mem_count < data.mem_v[i] and i not in p1_available_nodes:
            p1_available_nodes.append(i)
        if p2_mem_count > data.mem_v[i] and i not in p2_overload_nodes:
            p2_overload_nodes.append(i)
        elif p2_mem_count < data.mem_v[i] and i not in p2_available_nodes:
            p2_available_nodes.append(i)
        
    return p1_available_nodes, p2_available_nodes, p1_overload_nodes, p2_overload_nodes

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
    tau_i += settings.v2v_shortest_path_length(data.G, data.s_i[i], request[first_vnf])
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

def adjust_occ(p, data):
    start = -1
    last = -1
    i = 0
    while i < len(p):
        if p[i] == -1:
            start = (i // data.number_of_VNF_types) * data.number_of_VNF_types
            last = start + data.number_of_VNF_types - 1
            j = start
            while j <= last:
                if p[j] != -2 and p[j] != -1:
                    p[j] = -1
                j += 1
            i = last
        i += 1
    return p

def main(data_from_cplex):
    data = data_from_cplex
    start_time = time.time()

    # Initialize population randomly
    population = []
    request_list = [r for r in range(data.number_of_requests)]
    sFlag = False
    eFlag = False
    for p in range(data.number_of_individual):
        chromosome = [-3] * data.number_of_gene_in_an_individual
        assign_sequence = random.sample(request_list, k=data.number_of_requests)
        vnf_on_node = [[] for i in range(data.number_of_nodes)]
        rest_cpu_v = deepcopy(data.cpu_v)
        rest_mem_v = deepcopy(data.mem_v)
        node_set = deepcopy(data.nodes)
        for i in assign_sequence:
            j = i * data.number_of_VNF_types
            start = j
            last = j + data.number_of_VNF_types - 1
            # resources befor placing F_i[i]
            buffer_cpu = deepcopy(rest_cpu_v)
            buffer_mem = deepcopy(rest_mem_v)
            buffer_vnf_on_node = deepcopy(vnf_on_node)

            if data.s_i[i] in node_set:
                node_set.remove(data.s_i[i])
                sFlag = True
            if data.e_i[i] in node_set:
                node_set.remove(data.e_i[i])
                eFlag = True

            removed_node_list = []
            assigned_count = 0
            while(j <= last):
                vnf_type = j % data.number_of_VNF_types
                if vnf_type not in data.F_i[i]:
                    chromosome[j] = -2
                else:
                    choiced_nodes = []
                    while(len(node_set) > 0):
                        node = random.choice(node_set)
                        if vnf_type not in buffer_vnf_on_node[node]:
                            if buffer_mem[node] >= 1 and data.cpu_f[vnf_type] <= buffer_cpu[node]:
                                buffer_vnf_on_node[node].append(vnf_type)
                                buffer_mem[node] -= 1
                                chromosome[j] = node
                                buffer_cpu[node] -= data.cpu_f[vnf_type]
                                assigned_count += 1
                                if buffer_mem[node] < 1 or buffer_cpu[node] < min(data.cpu_f):
                                    node_set.remove(node)
                                    removed_node_list.append(node)
                                break
                            else:
                                choiced_nodes.append(node)
                                node_set.remove(node)
                        else:
                            if data.cpu_f[vnf_type] <= buffer_cpu[node]:
                                chromosome[j] = node
                                buffer_cpu[node] -= data.cpu_f[vnf_type]
                                assigned_count += 1
                                if buffer_mem[node] < 1 or buffer_cpu[node] < min(data.cpu_f):
                                    node_set.remove(node)
                                    removed_node_list.append(node)
                                break
                            else:
                                choiced_nodes.append(node)
                                node_set.remove(node)
                    node_set.extend(choiced_nodes)
                j += 1
            if sFlag:
                node_set.append(data.s_i[i])
                sFlag = False
            if eFlag:
                node_set.append(data.e_i[i])
                eFlag = False
            if assigned_count == len(data.F_i[i]):
                # Update resource state
                rest_cpu_v = buffer_cpu
                rest_mem_v = buffer_mem
                vnf_on_node = buffer_vnf_on_node
            else:
                # Return to the state before placing F_i[i]
                buffer_cpu = rest_cpu_v
                buffer_mem = rest_mem_v
                buffer_vnf_on_node = vnf_on_node
                node_set.extend(removed_node_list)
                # Request(F_i[i]) can not be placed on the network
                # completely so reject it
                j = start
                while j <= last:
                    if chromosome[j] != -2:
                        chromosome[j] = -1
                    j += 1
        population.append(chromosome)

    # Calculate the fitness value of each individual and
    # sort them in decreasing order
    fitness_of_chromosomes = calculate_fitness_value(population, data)

    sorted_population = sorted(
        population,
        key= lambda p: fitness_of_chromosomes[population.index(p)],
        reverse=True
    )
    population = sorted_population

    fittest = [-100]
    current_fittest = -1
    same_res_count = 0
    it = 1
    while True:
        count = 0
        # Selection
        # elitisms
        for i in range(int(data.number_of_individual * data.elitism_rate)):
            population.append(population[i])

        # Crossover & Mutation
        while len(population) < (2 * data.number_of_individual):
            tournament_set = random.sample(
                population[:settings.number_of_individual],
                k=data.number_of_individual_chose_from_population_for_tournament
            )

            p1_index = data.number_of_individual
            for i in range(len(tournament_set)):
                if population.index(tournament_set[i]) < p1_index:
                    p1_index = population.index(tournament_set[i])
            
            tournament_set = random.sample(
                population[:settings.number_of_individual],
                k=data.number_of_individual_chose_from_population_for_tournament
            )

            p2_index = data.number_of_individual
            for i in range(len(tournament_set)):
                if population.index(tournament_set[i]) < p2_index:
                    p2_index = population.index(tournament_set[i])

            p1_available_nodes = []
            p2_available_nodes = []
            p1_overload_nodes = []
            p2_overload_nodes = []
            it_cm = 1
            while it_cm <= data.max_repeat_time:
                p1 = deepcopy(population[p1_index])
                p2 = deepcopy(population[p2_index])
                # Crossover
                for i in range(len(p1)):
                    if p1[i] != -2 and p2[i] != -2:
                        if p1[i] in p1_overload_nodes or p2[i] in p2_overload_nodes:
                            buffer = p1[i]
                            p1[i] = p2[i]
                            p2[i] = buffer
                            # Check occupied situation
                            p1 = adjust_occ(p1, data)
                            p2 = adjust_occ(p2, data)
                            # Check constraints
                            (p1_available_nodes,
                            p2_available_nodes,
                            p1_overload_nodes,
                            p2_overload_nodes) = check_cap_after_cro_mut(p1, p2, data)
                        else:
                            cr = random.uniform(0, 1)
                            if cr < data.crossover_rate:
                                buffer = p1[i]
                                p1[i] = p2[i]
                                p2[i] = buffer
                                # Check occupied situation
                                p1 = adjust_occ(p1, data)
                                p2 = adjust_occ(p2, data)
                                # Check constraints
                                (p1_available_nodes,
                                p2_available_nodes,
                                p1_overload_nodes,
                                p2_overload_nodes) = check_cap_after_cro_mut(p1, p2, data)

                # Mutation
                for i in range(len(p1)):
                    if p1[i] != -2 and p2[i] != -2:
                        r_index = i // data.number_of_VNF_types
                        if p1[i] in p1_overload_nodes:
                            if data.s_i[r_index] in p1_available_nodes:
                                p1_available_nodes.remove(data.s_i[r_index])
                            if data.e_i[r_index] in p1_available_nodes:
                                p1_available_nodes.remove(data.e_i[r_index])
                            selected_nodes = []
                            flag = False
                            while True:
                                for n in selected_nodes:
                                    if n in p1_available_nodes:
                                        p1_available_nodes.remove(n)
                                if len(p1_available_nodes) > 0:
                                    p1[i] = random.choice(p1_available_nodes)
                                    selected_nodes.append(p1[i])
                                else:
                                    flag = True
                                    while True:
                                        rn = random.randint(-1, data.number_of_nodes - 1)
                                        if rn != p1[i] and rn != data.s_i[r_index] and rn != data.e_i[r_index]:
                                            p1[i] = rn
                                            break
                                # Check occupied situation
                                p1 = adjust_occ(p1, data)
                                # Check constraints
                                (p1_available_nodes,
                                p2_available_nodes,
                                p1_overload_nodes,
                                p2_overload_nodes) = check_cap_after_cro_mut(p1, p2, data)
                                if flag:
                                    break
                                elif p1[i] not in p1_overload_nodes:
                                    break
                        else:
                            mutation_R_1 = random.uniform(0, 1)
                            if mutation_R_1 < data.mutation_rate:
                                while True:
                                    rn = random.randint(-1, data.number_of_nodes - 1)
                                    if rn != p1[i] and rn != data.s_i[r_index] and rn != data.e_i[r_index]:
                                        p1[i] = rn
                                        break
                                # Check occupied situation
                                p1 = adjust_occ(p1, data)
                                # Check constraints
                                (p1_available_nodes,
                                p2_available_nodes,
                                p1_overload_nodes,
                                p2_overload_nodes) = check_cap_after_cro_mut(p1, p2, data)

                        if p2[i] in p2_overload_nodes:
                            if data.s_i[r_index] in p2_available_nodes:
                                p2_available_nodes.remove(data.s_i[r_index])
                            if data.e_i[r_index] in p2_available_nodes:
                                p2_available_nodes.remove(data.e_i[r_index])
                            selected_nodes = []
                            flag = False
                            while True:
                                for n in selected_nodes:
                                    if n in p2_available_nodes:
                                        p2_available_nodes.remove(n)
                                if len(p2_available_nodes) > 0:
                                    p2[i] = random.choice(p2_available_nodes)
                                    selected_nodes.append(p2[i])
                                else:
                                    flag = True
                                    while True:
                                        rn = random.randint(-1, data.number_of_nodes - 1)
                                        if rn != p2[i] and rn != data.s_i[r_index] and rn != data.e_i[r_index]:
                                            p2[i] = rn
                                            break
                                # Check occupied situation
                                p2 = adjust_occ(p2, data)
                                # Check constraints
                                (p1_available_nodes,
                                p2_available_nodes,
                                p1_overload_nodes,
                                p2_overload_nodes) = check_cap_after_cro_mut(p1, p2, data)
                                if flag:
                                    break
                                elif p2[i] not in p2_overload_nodes:
                                    break
                        else:
                            mutation_R_2 = random.uniform(0, 1)
                            if mutation_R_2 < data.mutation_rate:
                                while True:
                                    rn = random.randint(-1, data.number_of_nodes - 1)
                                    if rn != p2[i] and rn != data.s_i[r_index] and rn != data.e_i[r_index]:
                                        p2[i] = rn
                                        break
                                # Check occupied situation
                                p2 = adjust_occ(p2, data)
                                # Check constraints
                                (p1_available_nodes,
                                p2_available_nodes,
                                p1_overload_nodes,
                                p2_overload_nodes) = check_cap_after_cro_mut(p1, p2, data)
                flag = False
                if len(p1_overload_nodes) == 0:
                    population.append(p1)
                    flag = True
                if len(p2_overload_nodes) == 0:
                    population.append(p2)
                    flag = True
                if flag:
                    break
                it_cm += 1
            # print("it_cm:", it_cm)
            if it_cm > data.max_repeat_time:
                count += 1
                population.append(population[p1_index])
                population.append(population[p2_index])

        # print("count: ", count)
        del population[:data.number_of_individual]

        # Calculate the fitness value of each individual, and sort them in decresing order
        fitness_of_chromosomes = calculate_fitness_value(population, data)
        sorted_population = sorted(
            population,
            key= lambda p: fitness_of_chromosomes[population.index(p)],
            reverse=True
        )
        population = sorted_population

        fitness_of_chromosomes = calculate_fitness_value(population, data)

        # Select the fittest individual as the optimal solution for the current generation
        fittest.append(fitness_of_chromosomes[0])
        if fitness_of_chromosomes[0] == current_fittest:
            same_res_count += 1
        else:
            same_res_count = 0
            current_fittest = fitness_of_chromosomes[0]
        if same_res_count >= 50:
            break
        # print("CPLEX res: ", data.cplex_res)
        # print("fittest_value: ", fitness_of_chromosomes[0])
        # print("it: ", it)
        it += 1
    
    # solution = population[fitness_of_chromosomes.index(fittest[-1])]
    # print("GA solution: ", solution)
    fittest_value = max(fittest)
    end_time = time.time()
    time_cost = end_time - start_time

    res = {
        "fittest_value": fittest_value,
        "time_cost": time_cost,
    }

    return res
