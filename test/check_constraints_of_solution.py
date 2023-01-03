import settings

def check_if_meet_cpu_capacity_constraint(chromosome):
    flag = True
    for i in settings.nodes:
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
    flag = True
    for i in settings.nodes:
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
        if mem_count > settings.mem_v[i]:
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
    tau_i += (
        tau_vnf_i
        + settings.v2v_shortest_path_length(settings.G, settings.s_i[i], request[vnf_1])
        + settings.v2v_shortest_path_length(settings.G, settings.e_i[i], request[vnf_2])
    )
    if tau_i <= settings.r_i[i]:
        return True
    return False

if __name__ == "__main__":
    settings.init()

    chromosome = [-1, -1, -1, -2, -1, -1, -2, -2, -1, -1, -2, 3, 3, 3, 2, 0, 0, 0, 2, 2, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -2, -2,
    -2, 1, 1, -2, -1, -2, -1, -1, -1, -1, -1, -2, -2, -2, -2, -1, -2, -2, -2, -2, -1, -2, -1, -1, -2,
    -1, -1, -1, -1]


    cpu_capacity = check_if_meet_cpu_capacity_constraint(chromosome)
    mem_capacity = check_if_meet_mem_capacity_constraint(chromosome)
    delay_requirement = []
    for i in range(settings.number_of_requests):
        j = i * settings.number_of_VNF_types
        if -1 not in chromosome[j:j + settings.number_of_VNF_types]:
            if check_if_meet_delay_requirement(chromosome[j:j + settings.number_of_VNF_types], i):
                delay_requirement.append(True)
            else:
                delay_requirement.append(False)
    
    print("cpu_capacity: ", cpu_capacity)
    print("mem_capacity: ", mem_capacity)
    print("delay_requirement: ", delay_requirement)
