from docplex.mp.model import Model
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import numpy as np
import time, datetime, random, multiprocessing
import settings, pre_settings, settings_per_iteration
import my_ga_method, my_random_method, my_greedy_method, hGreedy, my_sa_method, improved_greedy

if __name__ == "__main__":
# def worker(seeds, seed_lower_bound, seed_upper_bound):
    # itt = 1
    # # seeds = []
    # while itt <= 1:
    # number_of_requests = [8]
    number_of_requests = [4,6,8,10,12]
    number_of_VNF_types = [5]
    # number_of_VNF_types = [5,6,7,8,9]
    # number_of_nodes = [12,13,14,15,16]
    number_of_nodes = [16]
    number_of_iteration = 1
    # seed = datetime.datetime.now().timestamp()
    # seed = itt
    # print("seed: ", seed)

    result_mean_cplex_res_value = []
    result_mean_ga_res_value = []
    result_mean_random_res_value = []
    result_mean_greedy_res_value = []
    result_mean_hGreedy_res_value = []
    result_mean_improved_greedy_res_value = []
    result_mean_sa_res_value = []

    result_mean_cplex_time_cost = []
    result_mean_ga_time_cost = []
    result_mean_random_time_cost = []
    result_mean_greedy_time_cost = []
    result_mean_hGreedy_time_cost = []
    result_mean_improved_greedy_time_cost = []
    result_mean_sa_time_cost = []

    result_mean_cplex_acc_rate = []
    result_mean_random_acc_rate = []
    result_mean_greedy_acc_rate = []
    result_mean_hGreedy_acc_rate = []
    result_mean_improved_greedy_acc_rate = []
    result_mean_sa_acc_rate = []

    result_mean_cplex_average_delay = []
    result_mean_random_average_delay = []
    result_mean_greedy_average_delay = []
    result_mean_hGreedy_average_delay = []
    result_mean_improved_greedy_average_delay = []
    result_mean_sa_average_delay = []

    result_mean_cplex_average_ratio_of_vnf_shared = []
    result_mean_random_average_ratio_of_vnf_shared = []
    result_mean_greedy_average_ratio_of_vnf_shared = []
    result_mean_hGreedy_average_ratio_of_vnf_shared = []
    result_mean_improved_greedy_average_ratio_of_vnf_shared = []
    result_mean_sa_average_ratio_of_vnf_shared = []

    run_time = 0

    for nr in range(len(number_of_requests)):
        # print("number of VNF types: ", number_of_VNF_types[0])
        # print("number of request: ", number_of_requests[nr])
        # print("number of nodes: ", number_of_nodes[0])
        run_time += 1
        
        cplex_res = 0
        cplex_time_cost = []
        ga_time_cost = []
        random_time_cost = []
        greedy_time_cost = []
        improved_greedy_time_cost = []
        sa_time_cost = []

        mean_cplex_res_value = 0
        mean_ga_res_value = 0
        mean_random_res_value = 0
        mean_greedy_res_value = 0
        mean_improved_greedy_res_value = 0
        mean_hGreedy_res_value = 0
        mean_sa_res_value = 0

        mean_cplex_time_cost = 0
        mean_ga_time_cost = 0
        mean_random_time_cost = 0
        mean_greedy_time_cost = 0
        mean_improved_greedy_time_cost = 0
        mean_hGreedy_time_cost = 0
        mean_sa_time_cost = 0

        mean_cplex_acc_rate = 0
        mean_ga_acc_rate = 0
        mean_random_acc_rate = 0
        mean_greedy_acc_rate = 0
        mean_improved_greedy_acc_rate = 0
        mean_hGreedy_acc_rate = 0
        mean_sa_acc_rate = 0

        mean_cplex_average_delay = 0
        mean_ga_average_delay = 0
        mean_random_average_delay = 0
        mean_greedy_average_delay = 0
        mean_improved_greedy_average_delay = 0
        mean_hGreedy_average_delay = 0
        mean_sa_average_delay = 0

        mean_cplex_average_ratio_of_vnf_shared = 0
        mean_ga_average_ratio_of_vnf_shared = 0
        mean_random_average_ratio_of_vnf_shared = 0
        mean_greedy_average_ratio_of_vnf_shared = 0
        mean_improved_greedy_average_ratio_of_vnf_shared = 0
        mean_hGreedy_average_ratio_of_vnf_shared = 0
        mean_sa_average_ratio_of_vnf_shared = 0

        # average_request_values = []
        seeds = [1382, 1260, 3642, 34, 1414, 1805, 811, 2934, 66, 4702, 459, 103, 3608, 2240, 2496]
        for seed in seeds:
            print("number of request: ", number_of_requests[nr])
            print("seed: ", seed)
            # Initialize the input data
            pre_settings.init(seed, run_time, number_of_nodes[0], number_of_VNF_types[0])
            # Initialize the input data
            settings.init(number_of_requests[nr], number_of_VNF_types[0], seed)
            # Initialize the input data for each iteration
            settings_per_iteration.init(number_of_requests[nr], number_of_VNF_types[0], number_of_nodes[0], seed)
            
            start_time = time.time()
            # print("CPLEX started!")

            # Creating the model
            VNF_placement_model = Model("VNF_placement")

            # Creating decsision variables
            z = VNF_placement_model.binary_var_dict(
                number_of_requests[nr], name="z")
            x = VNF_placement_model.binary_var_dict((
                (i, f, v)
                for i in range(number_of_requests[nr])
                for f in settings_per_iteration.F_i[i]
                for v in pre_settings.nodes),
                name="x"
            )
            y = VNF_placement_model.binary_var_dict((
                (f, v)
                for f in settings.F
                for v in pre_settings.nodes),
                name="y"
            )

            # Adding the constraints
            
            # Delay requirement constraint
            tau_vnf_i = []
            for i in range(number_of_requests[nr]):
                vnf_delay = 0
                for w in pre_settings.nodes:
                    for v in pre_settings.nodes:
                        if v != w:
                            for m in settings_per_iteration.F_i[i]:
                                for f in settings_per_iteration.F_i[i]:
                                    vnf_delay += (
                                        x[i, m, w]
                                        * x[i, f, v]
                                        * settings.check_are_neighbors(m, f, settings_per_iteration.F_i[i])
                                        * settings.v2v_shortest_path_length(pre_settings.G, w, v)
                                    )
                tau_vnf_i.append(vnf_delay)

            tau_starting_i = []
            for i in range(number_of_requests[nr]):
                start_delay = 0
                for v in pre_settings.nodes:
                    for f in settings_per_iteration.F_i[i]:
                        start_delay += (
                            x[i, f, v]
                            * settings.check_is_first_vnf(f, settings_per_iteration.F_i[i])
                            * settings.v2v_shortest_path_length(pre_settings.G, settings_per_iteration.s_i[i], v)
                        )
                tau_starting_i.append(start_delay)

            tau_ending_i = []
            for i in range(number_of_requests[nr]):
                end_delay = 0
                for v in pre_settings.nodes:
                    for f in settings_per_iteration.F_i[i]:
                        end_delay += (
                            x[i, f, v]
                            * settings.check_is_last_vnf(f, settings_per_iteration.F_i[i])
                            * settings.v2v_shortest_path_length(pre_settings.G, settings_per_iteration.e_i[i], v)
                        )
                tau_ending_i.append(end_delay)

            tau_i = []
            for i in range(len(tau_ending_i)):
                tau_i.append(tau_vnf_i[i] + tau_starting_i[i] + tau_ending_i[i])

            sequence = set()
            removed_set = set()
            for i in range(number_of_requests[nr]):
                sequence.add(i)
            for i in range(number_of_requests[nr]):
                if len(settings_per_iteration.F_i[i]) <= 1:
                    VNF_placement_model.add_constraint(
                        tau_i[i] <= settings.M * (1-z[i]) + settings_per_iteration.r_i[i])
                    removed_set.add(i)
            sequence -= removed_set

            VNF_placement_model.add_quadratic_constraints(
                tau_i[i] <= settings.M * (1-z[i]) + settings_per_iteration.r_i[i] for i in sequence
            )

            # Relation between z and x constraint
            VNF_placement_model.add_constraints((
                sum(x[i, f, v] for v in range(number_of_nodes[0])) == z[i]
                for i in range(number_of_requests[nr])
                for f in settings_per_iteration.F_i[i]),
                names="relation_between_z_and_x"
            )

            # Relation between y and x constraint
            VNF_placement_model.add_constraints((
                y[f, v] - x[i, f, v] >= 0
                for i in range(number_of_requests[nr])
                for f in settings_per_iteration.F_i[i]
                for v in pre_settings.nodes),
                names="relation_between_y_and_x"
            )

            # CPU capacity constraint
            for v in range(number_of_nodes[0]):
                occupied_cpu_resources = 0
                for i in range(number_of_requests[nr]):
                    for f in settings_per_iteration.F_i[i]:
                        occupied_cpu_resources += x[i, f, v] * settings.cpu_f[f]
                VNF_placement_model.add_constraint(occupied_cpu_resources <= pre_settings.cpu_v[v])

            # Memory capacity constraint
            for v in range(number_of_nodes[0]):
                occupied_mem_resources = 0
                for f in settings.F:
                    occupied_mem_resources += y[f, v]
                VNF_placement_model.add_constraint(occupied_mem_resources <= pre_settings.mem_v[v])

            # Defineing the objective function
            deployment_cost = 0
            for v in range(number_of_nodes[0]):
                for f in range(number_of_VNF_types[0]):
                    deployment_cost += y[f, v] * pre_settings.cost_f[f]
            obj_fn = sum(z[i] * settings_per_iteration.profit_i[i] for i in range(number_of_requests[nr])) - deployment_cost
            print(VNF_placement_model.print_information())
            VNF_placement_model.set_objective('max', obj_fn)

            # Solve the model
            sol = VNF_placement_model.solve()

            end_time = time.time()
            cplex_time_cost.append(end_time - start_time)

            acc_rate = 0
            vnf_count = 0
            shared_vnf_count = 0
            
            if sol:
                cplex_res = sol.get_value(obj_fn)
                x_res = VNF_placement_model.solution.get_values(list(x.values()))
                z_res = VNF_placement_model.solution.get_values(list(z.values()))
                print(sol)

                # Caculate acceptance rate
                acc_count = 0
                for i in range(len(z_res)):
                    if z_res[i] == 1.0:
                        acc_count += 1
                acc_rate = acc_count / number_of_requests[nr]

                # Caculate ratio of shared VNF
                ratio_of_vnf_shared = 0
                used_count_of_vnfs_on_nodes = [[0 for i in range(number_of_VNF_types[0])] for i in range(number_of_nodes[0])]
                for loc in range(len(x_res)):
                    if x_res[loc] == 1.00:
                        i = 0
                        f_count = 0
                        v = 0
                        count_1 = len(settings_per_iteration.F_i[i]) * number_of_nodes[0]
                        while loc >= count_1:
                            i += 1
                            count_1 += len(settings_per_iteration.F_i[i]) * number_of_nodes[0]
                        count_1 -= len(settings_per_iteration.F_i[i]) * number_of_nodes[0]
                        loc -= count_1
                        count_2 = number_of_nodes[0]
                        while loc >= count_2:
                            f_count += 1
                            count_2 += number_of_nodes[0]
                        count_2 -= number_of_nodes[0]
                        loc -= count_2
                        v = loc % number_of_nodes[0]
                        # print("i,f,v: ", i,settings_per_iteration.F_i[i][f_count],v)
                        used_count_of_vnfs_on_nodes[v][settings_per_iteration.F_i[i][f_count]] += 1
                for i in range(number_of_nodes[0]):
                    for j in range(number_of_VNF_types[0]):
                        if used_count_of_vnfs_on_nodes[i][j] > 1:
                            shared_vnf_count += 1
                            vnf_count += 1
                        elif used_count_of_vnfs_on_nodes[i][j] == 1:
                            vnf_count += 1
                ratio_of_vnf_shared = shared_vnf_count / vnf_count
            else:
                cplex_res = 0

            class Data:
                num_of_VNF_types = number_of_VNF_types[0]
                num_of_requests = number_of_requests[nr]
                num_of_nodes = number_of_nodes[0]
                F = settings.F
                G = pre_settings.G
                nodes = pre_settings.nodes
                cpu_v = pre_settings.cpu_v
                mem_v = pre_settings.mem_v
                cpu_f = settings.cpu_f
                F_i = settings_per_iteration.F_i
                profit_i = settings_per_iteration.profit_i
                r_i = settings_per_iteration.r_i
                s_i = settings_per_iteration.s_i
                e_i = settings_per_iteration.e_i
                number_of_individual = settings.number_of_individual
                number_of_gene_in_an_individual = settings.number_of_gene_in_an_individual
                elitism_rate = settings.elitism_rate
                iteration_for_ga = settings.iteration_for_ga
                max_repeat_time = settings.max_repeat_time
                number_of_individual_chose_from_population_for_tournament = settings.number_of_individual_chose_from_population_for_tournament
                crossover_rate = settings.crossover_rate
                mutation_rate = settings.mutation_rate

            # call other methods
            # print("GA started!")
            # ga_res = my_ga_method.main(Data)
            # print("Random started!")
            # random_res = my_random_method.main(Data)
            # print("Greedy started!")
            # greedy_res = my_greedy_method.main(Data)
            # print("hGreedy started!")
            hGreedy_res = hGreedy.main(Data)
            # print("Improved Greedy started!")
            improved_greedy_res = improved_greedy.main(Data)
            # print("SA started!")
            sa_res = my_sa_method.main(Data, improved_greedy_res["solution"], improved_greedy_res["total_profit"])

            # results
            mean_cplex_res_value += cplex_res
            # mean_ga_res_value += ga_res["fittest_value"]
            # mean_random_res_value += random_res["total_profit"]
            # mean_greedy_res_value += greedy_res["total_profit"]
            mean_hGreedy_res_value += hGreedy_res["total_profit"]
            mean_improved_greedy_res_value += improved_greedy_res["total_profit"]
            mean_sa_res_value += sa_res["total_profit"]
            
            mean_cplex_time_cost += end_time - start_time
            # mean_ga_time_cost += ga_res["time_cost"]
            # mean_random_time_cost += random_res["time_cost"]
            # mean_greedy_time_cost += greedy_res["time_cost"]
            mean_hGreedy_time_cost += hGreedy_res["time_cost"]
            mean_improved_greedy_time_cost += improved_greedy_res["time_cost"]
            mean_sa_time_cost += sa_res["time_cost"]
            
            mean_cplex_acc_rate += acc_rate
            # mean_random_acc_rate += random_res["acc_rate"]
            # mean_greedy_acc_rate += greedy_res["acc_rate"]
            mean_hGreedy_acc_rate += hGreedy_res["acc_rate"]
            mean_improved_greedy_acc_rate += improved_greedy_res["acc_rate"]
            mean_sa_acc_rate += sa_res["acc_rate"]

            # mean_random_average_delay += random_res["average_delay"]
            # mean_greedy_average_delay += greedy_res["average_delay"]
            # mean_hGreedy_average_delay += hGreedy_res["average_delay"]
            # mean_improved_greedy_average_delay += improved_greedy_res["average_delay"]
            # mean_sa_average_delay += sa_res["average_delay"]

            mean_cplex_average_ratio_of_vnf_shared += ratio_of_vnf_shared
            # mean_random_average_ratio_of_vnf_shared += random_res["ratio_of_vnf_shared"]
            # mean_greedy_average_ratio_of_vnf_shared += greedy_res["ratio_of_vnf_shared"]
            mean_hGreedy_average_ratio_of_vnf_shared += hGreedy_res["ratio_of_vnf_shared"]
            mean_improved_greedy_average_ratio_of_vnf_shared += improved_greedy_res["ratio_of_vnf_shared"]
            mean_sa_average_ratio_of_vnf_shared += sa_res["ratio_of_vnf_shared"]

        mean_cplex_res_value /= len(seeds)
        # mean_ga_res_value /= number_of_iteration
        # mean_random_res_value /= number_of_iteration
        # mean_greedy_res_value /= number_of_iteration
        mean_hGreedy_res_value /= len(seeds)
        mean_improved_greedy_res_value /= len(seeds)
        mean_sa_res_value /= len(seeds)

        mean_cplex_time_cost /= len(seeds)
        # mean_ga_time_cost /= number_of_iteration
        # mean_random_time_cost /= number_of_iteration
        # mean_greedy_time_cost /= number_of_iteration
        mean_hGreedy_time_cost /= len(seeds)
        mean_improved_greedy_time_cost /= len(seeds)
        mean_sa_time_cost /= len(seeds)

        mean_cplex_acc_rate /= len(seeds)
        # mean_random_acc_rate /= number_of_iteration
        # mean_greedy_acc_rate /= number_of_iteration
        mean_hGreedy_acc_rate /= len(seeds)
        mean_improved_greedy_acc_rate /= len(seeds)
        mean_sa_acc_rate /= len(seeds)

        # mean_random_average_delay /= number_of_iteration
        # mean_greedy_average_delay /= number_of_iteration
        # mean_hGreedy_average_delay /= len(seeds)
        # mean_improved_greedy_average_delay /= len(seeds)
        # mean_sa_average_delay /= len(seeds)

        mean_cplex_average_ratio_of_vnf_shared /= len(seeds)
        # mean_random_average_ratio_of_vnf_shared /= number_of_iteration
        # mean_greedy_average_ratio_of_vnf_shared /= number_of_iteration
        mean_hGreedy_average_ratio_of_vnf_shared /= len(seeds)
        mean_improved_greedy_average_ratio_of_vnf_shared /= len(seeds)
        mean_sa_average_ratio_of_vnf_shared /= len(seeds)


        result_mean_cplex_res_value.append(mean_cplex_res_value)
        # result_mean_ga_res_value.append(mean_ga_res_value)
        # result_mean_random_res_value.append(mean_random_res_value)
        # result_mean_greedy_res_value.append(mean_greedy_res_value)
        result_mean_hGreedy_res_value.append(mean_hGreedy_res_value)
        result_mean_improved_greedy_res_value.append(mean_improved_greedy_res_value)
        result_mean_sa_res_value.append(mean_sa_res_value)
        
        result_mean_cplex_time_cost.append(mean_cplex_time_cost)
        # result_mean_ga_time_cost.append(mean_ga_time_cost)
        # result_mean_random_time_cost.append(mean_random_time_cost)
        # result_mean_greedy_time_cost.append(mean_greedy_time_cost)
        result_mean_hGreedy_time_cost.append(mean_hGreedy_time_cost)
        result_mean_improved_greedy_time_cost.append(mean_improved_greedy_time_cost)
        result_mean_sa_time_cost.append(mean_sa_time_cost)

        result_mean_cplex_acc_rate.append(mean_cplex_acc_rate)
        # result_mean_random_acc_rate.append(mean_random_acc_rate)
        # result_mean_greedy_acc_rate.append(mean_greedy_acc_rate)
        result_mean_hGreedy_acc_rate.append(mean_hGreedy_acc_rate)
        result_mean_improved_greedy_acc_rate.append(mean_improved_greedy_acc_rate)
        result_mean_sa_acc_rate.append(mean_sa_acc_rate)
        
        # result_mean_random_average_delay.append(mean_random_average_delay)
        # result_mean_greedy_average_delay.append(mean_greedy_average_delay)
        # result_mean_hGreedy_average_delay.append(mean_hGreedy_average_delay)
        # result_mean_improved_greedy_average_delay.append(mean_improved_greedy_average_delay)
        # result_mean_sa_average_delay.append(mean_sa_average_delay)

        result_mean_cplex_average_ratio_of_vnf_shared.append(mean_cplex_average_ratio_of_vnf_shared)
        # result_mean_random_average_ratio_of_vnf_shared.append(mean_random_average_ratio_of_vnf_shared)
        # result_mean_greedy_average_ratio_of_vnf_shared.append(mean_greedy_average_ratio_of_vnf_shared)
        result_mean_hGreedy_average_ratio_of_vnf_shared.append(mean_hGreedy_average_ratio_of_vnf_shared)
        result_mean_improved_greedy_average_ratio_of_vnf_shared.append(mean_improved_greedy_average_ratio_of_vnf_shared)
        result_mean_sa_average_ratio_of_vnf_shared.append(mean_sa_average_ratio_of_vnf_shared)

#         flag = True
#         for i in range(len(result_mean_sa_res_value)):
#             if (result_mean_sa_res_value[i] <= result_mean_hGreedy_res_value[i] or
#                 result_mean_sa_res_value[i] <= result_mean_improved_greedy_res_value[i]):
#                     flag = False
#                     break
#         # if flag:
#         #     for i in range(len(result_mean_sa_acc_rate) - 1):
#         #         if result_mean_hGreedy_acc_rate[i] < result_mean_hGreedy_acc_rate[i + 1]:
#         #             flag = False
#         #             break
#         if flag:
#             # seeds.append(seed)
#             seeds.put(seed)
#             # print("Good!")
#             # if len(seeds) == 50:
#             #     break
#         # if itt == seed_upper_bound and len(seeds) != 0:
#         #     final_seeds.extend(seeds)
#         #     break
#         itt += 1
#     # print("seeds: ", seeds)

# if __name__ == "__main__":
#     seed_lower_bound = 1
#     seed_upper_bound = 25

#     seeds = Queue()
#     processes = []
#     num_cpus = multiprocessing.cpu_count()
#     # num_cpus = 39

#     start_time = time.time()

#     for _ in range(num_cpus):
#         p = Process(target=worker, args=(seeds, seed_lower_bound, seed_upper_bound))
#         p.start()
#         processes.append(p)
#         seed_lower_bound += 25
#         seed_upper_bound += 25
    
#     for t in processes:
#         t.join()

#     final_seeds = []
#     while not seeds.empty():
#         result = seeds.get()
#         final_seeds.append(result)
    
#     end_time = time.time()
#     processes_time_cost = end_time - start_time
#     print("processes_time_cost:", processes_time_cost)

#     print("final_seeds: ", final_seeds)
                    
    print("result_mean_cplex_res_value: ", result_mean_cplex_res_value)
    # print("result_mean_ga_res_value:", result_mean_ga_res_value)
    # print("result_mean_random_res_value:", result_mean_random_res_value)
    # print("result_mean_greedy_res_value:", result_mean_greedy_res_value)
    print("result_mean_hGreedy_res_value:", result_mean_hGreedy_res_value)
    print("result_mean_improved_greedy_res_value:", result_mean_improved_greedy_res_value)
    print("result_mean_sa_res_value:", result_mean_sa_res_value)
    print("----------------------------------------------------------------------------------")
    print("result_mean_cplex_time_cost: ", result_mean_cplex_time_cost)
    # print("result_mean_ga_time_cost: ", result_mean_ga_time_cost)
    # print("result_mean_random_time_cost: ", result_mean_random_time_cost)
    # print("result_mean_greedy_time_cost: ", result_mean_greedy_time_cost)
    print("result_mean_hGreedy_time_cost: ", result_mean_hGreedy_time_cost)
    print("result_mean_improved_greedy_time_cost: ", result_mean_improved_greedy_time_cost)
    print("result_mean_sa_time_cost: ", result_mean_sa_time_cost)
    print("----------------------------------------------------------------------------------")
    print("result_mean_cplex_acc_rate: ", result_mean_cplex_acc_rate)
    # print("result_mean_random_acc_rate: ", result_mean_random_acc_rate)
    # print("result_mean_greedy_acc_rate: ", result_mean_greedy_acc_rate)
    print("result_mean_hGreedy_acc_rate: ", result_mean_hGreedy_acc_rate)
    print("result_mean_improved_greedy_acc_rate: ", result_mean_improved_greedy_acc_rate)
    print("result_mean_sa_acc_rate: ", result_mean_sa_acc_rate)
    # print("----------------------------------------------------------------------------------")
    # print("result_mean_random_average_delay: ", result_mean_random_average_delay)
    # print("result_mean_greedy_average_delay: ", result_mean_greedy_average_delay)
    # print("result_mean_hGreedy_average_delay: ", result_mean_hGreedy_average_delay)
    # print("result_mean_improved_greedy_average_delay: ", result_mean_improved_greedy_average_delay)
    # print("result_mean_sa_average_delay: ", result_mean_sa_average_delay)
    print("----------------------------------------------------------------------------------")
    print("result_mean_cplex_average_ratio_of_vnf_shared: ", result_mean_cplex_average_ratio_of_vnf_shared)
    # print("result_mean_random_average_ratio_of_vnf_shared: ", result_mean_random_average_ratio_of_vnf_shared)
    # print("result_mean_greedy_average_ratio_of_vnf_shared: ", result_mean_greedy_average_ratio_of_vnf_shared)
    print("result_mean_hGreedy_average_ratio_of_vnf_shared: ", result_mean_hGreedy_average_ratio_of_vnf_shared)
    print("result_mean_improved_greedy_average_ratio_of_vnf_shared: ", result_mean_improved_greedy_average_ratio_of_vnf_shared)
    print("result_mean_sa_average_ratio_of_vnf_shared: ", result_mean_sa_average_ratio_of_vnf_shared)

    if False:
        results = [result_mean_cplex_time_cost, result_mean_improved_greedy_time_cost, result_mean_greedy_time_cost, result_mean_hGreedy_time_cost]
        ylim = 0
        for i in range(len(results)):
            for j in range(len(results[i])):
                if results[i][j] > ylim:
                    ylim = results[i][j]

        # results = [result_mean_cplex_res_value, result_mean_ga_res_value, result_mean_greedy_res_value, result_mean_hGreedy_res_value, result_mean_random_res_value]
        colors = ['lightcoral', 'peru', 'powderblue', 'yellowgreen']
        labels = ['CPLEX', 'VPIG', 'HGreedy', 'Greedy']

        results = {
            'result_mean_cplex_res_value': tuple(result_mean_cplex_res_value),
            # 'result_mean_ga_res_value': tuple(result_mean_ga_res_value),
            # 'result_mean_sa_res_value': tuple(result_mean_sa_res_value),
            'result_mean_improved_greedy_res_value': tuple(result_mean_improved_greedy_res_value),
            'result_mean_hGreedy_res_value': tuple(result_mean_hGreedy_res_value),
            'result_mean_greedy_res_value': tuple(result_mean_greedy_res_value),
        }
        
        x = np.arange(len(number_of_requests))  # the label locations
        width = 0.2  # the width of the bars
        multiplier = -0.5
        l = 0

        fig, ax = plt.subplots(layout='constrained')

        patterns = ["**", "---", "\\\\\\", "|||"]

        for method, result in results.items():
            offset = width * multiplier
            ax.bar(x + offset, result, width, color=colors[l], label=labels[l], align='center')
            multiplier += 1
            l += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Number of requests')
        ax.set_ylabel('Profit')
        # ax.set_title('number_of_iteration=' + str(number_of_iteration))
        ax.set_xticks(x + width, number_of_requests)
        ax.legend(loc='upper left')
        ax.set_ylim(0, ylim + 10)
        # plt.show()
        current_date = datetime.datetime.now()
        plt.savefig("../result/" + str(current_date.month) + str(current_date.day) + "-bar.png")
    else:
        # line 1 points
        x1 = number_of_requests
        y1 = result_mean_cplex_acc_rate
        plt.plot(x1, y1, 's-', color='mediumseagreen', label="CPLEX", markersize=8, linewidth=2.5)

        # # line 2 points
        # x2 = number_of_requests
        # y2 = result_mean_ga_res_value
        # plt.plot(x2, y2, 'o-', color='g', label="GA", markersize=8, linewidth=2.5)

        # # line 3 points
        # x3 = number_of_requests
        # y3 = result_mean_random_time_cost
        # plt.plot(x3, y3, 'D-', color='b', label="Random", markersize=8, linewidth=2.5)

        # # line 4 points
        # x4 = number_of_VNF_types
        # y4 = result_mean_greedy_res_value
        # plt.plot(x4, y4, '*-', color='yellowgreen', label="Greedy", markersize=8, linewidth=2.5)

        # line 5 points
        x5 = number_of_requests
        y5 = result_mean_hGreedy_acc_rate
        plt.plot(x5, y5, 'x-', color='paleturquoise', label="HGreedy", markersize=8, linewidth=2.5)

        # line 6 points
        x6 = number_of_requests
        y6 = result_mean_improved_greedy_acc_rate
        plt.plot(x6, y6, 'D-', color='mediumturquoise', label="Improved Greedy", markersize=8, linewidth=2.5)

        # line 7 points
        x7 = number_of_requests
        y7 = result_mean_sa_acc_rate
        plt.plot(x7, y7, 'o-', color='bisque', label="SA", markersize=8, linewidth=2.5)

        plt.xticks(number_of_requests, [str(number_of_requests[i]) for i in range(len(number_of_requests))])
        plt.xlabel('Number of requests')
        plt.ylabel('Acceptance rate')
        plt.legend()
        # plt.show()
        current_date = datetime.datetime.now()
        plt.savefig("../result/" + str(current_date.month) + str(current_date.day) + "-line.png")
