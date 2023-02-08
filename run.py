from docplex.mp.model import Model
import matplotlib.pyplot as plt
import time, random
import settings, pre_settings, settings_per_iteration
import my_ga_method, my_random_method, my_greedy_method

def calculate_requests_needed_cpu():
    request_needed_cpu = []
    for i in range(len(settings_per_iteration.F_i)):
        cpu_count = 0
        for j in range(len(settings_per_iteration.F_i[i])):
            cpu_count += settings.cpu_f[settings_per_iteration.F_i[i][j]]
        request_needed_cpu.append(cpu_count)
    return request_needed_cpu

if __name__ == "__main__":
    # number_of_requests = [15]
    number_of_requests = [15, 20, 25, 30, 35, 40, 45, 50, 55]
    number_of_VNF_types = [5]
    # number_of_VNF_types = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    number_of_iteration = 50

    result_mean_cplex_res_value = []
    result_mean_ga_res_value = []
    result_mean_random_res_value = []
    result_mean_greedy_res_value = []
    result_mean_cplex_time_cost = []
    result_mean_ga_time_cost = []
    result_mean_random_time_cost = []
    result_mean_greedy_time_cost = []

    # Initialize the input data
    pre_settings.init()
    seed = random.randint(0, 238467)

    for nr in range(len(number_of_requests)):
        # print("number of VNF types: ", number_of_VNF_types[0])
        print("number of request: ", number_of_requests[nr])
        cplex_res = 0

        cplex_res_value = []
        ga_res_value = []
        random_res_value = []
        greedy_res_value = []

        cplex_time_cost = []
        ga_time_cost = []
        random_time_cost = []
        greedy_time_cost = []

        mean_cplex_res_value = 0
        mean_ga_res_value = 0
        mean_random_res_value = 0
        mean_greedy_res_value = 0

        mean_cplex_time_cost = 0
        mean_ga_time_cost = 0
        mean_random_time_cost = 0
        mean_greedy_time_cost = 0

        # Initialize the input data
        settings.init(number_of_requests[nr], number_of_VNF_types[0], seed)

        for iteration in range(number_of_iteration):
            # Initialize the input data for each iteration
            settings_per_iteration.init(number_of_requests[nr], number_of_VNF_types[0])

            request_needed_cpu = calculate_requests_needed_cpu()
            max_c = max(request_needed_cpu)
            min_c = min(request_needed_cpu)
            request_value = []
            for i in range(number_of_requests[nr]):
                max_p = max(settings_per_iteration.profit_i)
                min_p = min(settings_per_iteration.profit_i)
                if request_needed_cpu[i] == min_c:
                    # Avoiding "devided by zero"
                    request_value.append(
                        (settings_per_iteration.profit_i[i] - min_p)
                        / (max_p - min_p)
                    )
                else:
                    request_value.append(
                        ((settings_per_iteration.profit_i[i] - min_p)
                         / (max_p - min_p))
                        / ((request_needed_cpu[i] - min_c)
                            / (max_c - min_c))
                    )
            print("request_value: ", request_value)
            a_count = 0
            for a in range(number_of_requests[nr]):
                a_count += request_value[a]
            average_request_value = a_count / number_of_requests[nr]
            print("average_request_value: ", average_request_value)

            start_time = time.time()
            # ------------------------------------------------------------------------------------------
            # Creating the model
            # ------------------------------------------------------------------------------------------

            VNF_placement_model = Model("VNF_placement")

            # ------------------------------------------------------------------------------------------
            # Creating decsision variables
            # ------------------------------------------------------------------------------------------

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

            # -------------------------------------------------------------------------------------
            # Adding the constraints
            # -------------------------------------------------------------------------------------

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
                tau_i.append(tau_vnf_i[i] +
                             tau_starting_i[i] + tau_ending_i[i])

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

            # # Number of same type VNF in a request constraint
            # for i in range(number_of_requests[nr]):
            #     for f in settings.F:
            #         count = 0
            #         for l in range(len(settings.F_i[i])):
            #             if settings.F_i[i][l] == f:
            #                 count += 1
            #         VNF_placement_model.add_constraint(count <= 1, ctname="numbre_of_same_type_VNF_in_a_request")

            # Relation between z and x constraint
            VNF_placement_model.add_constraints((
                sum(x[i, f, v]
                    for v in range(pre_settings.number_of_nodes)) == z[i]
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
            for v in range(pre_settings.number_of_nodes):
                occupied_cpu_resources = 0
                for i in range(number_of_requests[nr]):
                    for f in settings_per_iteration.F_i[i]:
                        occupied_cpu_resources += x[i,
                                                    f, v] * settings.cpu_f[f]
                VNF_placement_model.add_constraint(
                    occupied_cpu_resources <= pre_settings.cpu_v[v])

            # Memory capacity constraint
            for v in range(pre_settings.number_of_nodes):
                occupied_mem_resources = 0
                for f in settings.F:
                    occupied_mem_resources += y[f, v]
                VNF_placement_model.add_constraint(
                    occupied_mem_resources <= pre_settings.mem_v[v])

            # -------------------------------------------------------------------------------------
            # Defineing the objective function
            # -------------------------------------------------------------------------------------

            obj_fn = sum(z[i] * settings_per_iteration.profit_i[i]
                         for i in range(number_of_requests[nr]))

            VNF_placement_model.set_objective('max', obj_fn)

            # Solve the model
            sol = VNF_placement_model.solve()
            if sol:
                cplex_res_value.append(sol.get_value(obj_fn))
                cplex_res = sol.get_value(obj_fn)
            else:
                cplex_res_value.append(0)
                cplex_res = 0

            end_time = time.time()
            cplex_time_cost.append(end_time - start_time)

            # define input data for GA method
            class Data:
                number_of_VNF_types = number_of_VNF_types[0]
                number_of_requests = number_of_requests[nr]
                number_of_nodes = pre_settings.number_of_nodes
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
                max_iter_cro_mut = settings.max_iter_cro_mut
                number_of_individual_chose_from_population_for_tournament = settings.number_of_individual_chose_from_population_for_tournament
                crossover_rate = settings.crossover_rate
                mutation_rate = settings.mutation_rate

            # call other methods
            ga_res = my_ga_method.main(Data)
            random_res = my_random_method.main(Data)
            greedy_res = my_greedy_method.main(Data)

            # result
            ga_res_value.append(ga_res["fittest_value"])
            ga_time_cost.append(ga_res["time_cost"])
            random_res_value.append(random_res["total_profit"])
            random_time_cost.append(random_res["time_cost"])
            greedy_res_value.append(greedy_res["total_profit"])
            greedy_time_cost.append(greedy_res["time_cost"])

            mean_cplex_res_value += cplex_res
            mean_ga_res_value += ga_res["fittest_value"]
            mean_random_res_value += random_res["total_profit"]
            mean_greedy_res_value += greedy_res["total_profit"]
            mean_cplex_time_cost += end_time - start_time
            mean_ga_time_cost += ga_res["time_cost"]
            mean_random_time_cost += random_res["time_cost"]
            mean_greedy_time_cost += greedy_res["time_cost"]

        mean_cplex_res_value /= number_of_iteration
        mean_ga_res_value /= number_of_iteration
        mean_random_res_value /= number_of_iteration
        mean_greedy_res_value /= number_of_iteration
        mean_cplex_time_cost /= number_of_iteration
        mean_ga_time_cost /= number_of_iteration
        mean_random_time_cost /= number_of_iteration
        mean_greedy_time_cost /= number_of_iteration

        result_mean_cplex_res_value.append(mean_cplex_res_value)
        result_mean_ga_res_value.append(mean_ga_res_value)
        result_mean_random_res_value.append(mean_random_res_value)
        result_mean_greedy_res_value.append(mean_greedy_res_value)
        result_mean_cplex_time_cost.append(mean_cplex_time_cost)
        result_mean_ga_time_cost.append(mean_ga_time_cost)
        result_mean_random_time_cost.append(mean_random_time_cost)
        result_mean_greedy_time_cost.append(mean_greedy_time_cost)

    print("result_mean_cplex_res_value: ", result_mean_cplex_res_value)
    print("result_mean_ga_res_value:", result_mean_ga_res_value)
    print("result_mean_random_res_value:", result_mean_random_res_value)
    print("result_mean_greedy_res_value:", result_mean_greedy_res_value)
    print("result_mean_cplex_time_cost: ", result_mean_cplex_time_cost)
    print("result_mean_ga_time_cost: ", result_mean_ga_time_cost)
    print("result_mean_random_time_cost: ", result_mean_random_time_cost)
    print("result_mean_greedy_time_cost: ", result_mean_greedy_time_cost)

    # line 1 points
    x1 = number_of_requests
    y1 = result_mean_cplex_res_value
    plt.plot(x1, y1, 's-', color='r', label="CPLEX",
             markersize=8, linewidth=2.5)

    # line 2 points
    x2 = number_of_requests
    y2 = result_mean_ga_res_value
    plt.plot(x2, y2, 'o-', color='g', label="GA", markersize=8, linewidth=2.5)

    # line 3 points
    x2 = number_of_requests
    y2 = result_mean_random_res_value
    plt.plot(x2, y2, 'D-', color='b', label="Random",
             markersize=8, linewidth=2.5)

    # line 4 points
    x2 = number_of_requests
    y2 = result_mean_greedy_res_value
    plt.plot(x2, y2, '*-', color='y', label="Greedy",
             markersize=8, linewidth=2.5)

    plt.xlabel('Number of requests')
    plt.ylabel('Profit')
    plt.title('number_of_iteration=50, iteration_for_ga=50')
    plt.legend()
    plt.show()
