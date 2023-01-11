from docplex.mp.model import Model
import settings
import matplotlib.pyplot as plt
import time
import my_ga_method
import my_random_method

if __name__ == "__main__":
    number_of_requests = [15]
    # number_of_requests = [15, 20, 25, 30, 35, 40, 45, 50, 55]
    # number_of_VNF_types = [5]
    number_of_VNF_types = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    number_of_iteration = 50

    result_mean_cplex_res_value = []
    result_mean_ga_res_value = []
    result_mean_random_res_value = []
    result_mean_cplex_time_cost = []
    result_mean_ga_time_cost = []
    result_mean_random_time_cost = []
    for nvt in range(len(number_of_VNF_types)):
        print("number of VNF types: ", number_of_VNF_types[nvt])
        print("---------------------------------------------------------------------------------------------------------------------------------------")
        # print("number of request: ", number_of_requests[0])
        cplex_res_value = []
        ga_res_value = []
        random_res_value = []

        cplex_time_cost = []
        ga_time_cost = []
        random_time_cost = []

        mean_cplex_res_value = 0
        mean_ga_res_value = 0
        mean_random_res_value = 0

        mean_cplex_time_cost = 0
        mean_ga_time_cost = 0
        mean_random_time_cost = 0

        # Initialize the input data
        settings.init(number_of_requests[0], number_of_VNF_types[nvt])
        for iteration in range(number_of_iteration):
            # print("iteration: ", iteration)
                        
            start_time = time.time()
            #------------------------------------------------------------------------------------------
            # Creating the model
            #------------------------------------------------------------------------------------------

            VNF_placement_model = Model("VNF_placement")

            #------------------------------------------------------------------------------------------
            # Creating decsision variables
            #------------------------------------------------------------------------------------------

            z = VNF_placement_model.binary_var_dict(number_of_requests[0], name="z")
            x = VNF_placement_model.binary_var_dict((
                (i, f, v)
                for i in range(number_of_requests[0])
                for f in settings.F_i[i]
                for v in settings.nodes),
                name="x"
            )
            y = VNF_placement_model.binary_var_dict((
                (f, v)
                for f in settings.F
                for v in settings.nodes),
                name="y"
            )

            #-------------------------------------------------------------------------------------
            # Adding the constraints
            #-------------------------------------------------------------------------------------

            # Delay requirement constraint
            tau_vnf_i = []
            for i in range(number_of_requests[0]):
                vnf_delay = 0
                for w in settings.nodes:
                    for v in settings.nodes:
                        if v != w:
                            for m in settings.F_i[i]:
                                for f in settings.F_i[i]:
                                    vnf_delay += (
                                        x[i, m, w]
                                        * x[i, f, v]
                                        * settings.check_are_neighbors(m, f, settings.F_i[i])
                                        * settings.v2v_shortest_path_length(settings.G, w, v)
                                    )
                tau_vnf_i.append(vnf_delay)

            tau_starting_i = []
            for i in range(number_of_requests[0]):
                start_delay = 0
                for v in settings.nodes:
                    for f in settings.F_i[i]:
                        start_delay += (
                            x[i, f, v]
                            * settings.check_is_first_vnf(f, settings.F_i[i])
                            * settings.v2v_shortest_path_length(settings.G, settings.s_i[i], v)
                        )
                tau_starting_i.append(start_delay)

            tau_ending_i = []
            for i in range(number_of_requests[0]):
                end_delay = 0
                for v in settings.nodes:
                    for f in settings.F_i[i]:
                        end_delay += (
                            x[i, f, v]
                            * settings.check_is_last_vnf(f, settings.F_i[i])
                            * settings.v2v_shortest_path_length(settings.G, settings.e_i[i], v)
                        )
                tau_ending_i.append(end_delay)

            tau_i = []
            for i in range(len(tau_ending_i)):
                tau_i.append(tau_vnf_i[i] + tau_starting_i[i] + tau_ending_i[i])

            sequence = set()
            removed_set = set()
            for i in range(number_of_requests[0]):
                sequence.add(i)
            for i in range(number_of_requests[0]):
                if len(settings.F_i[i]) <= 1:
                    VNF_placement_model.add_constraint(tau_i[i] <= settings.M * (1-z[i]) + settings.r_i[i])
                    removed_set.add(i)
            sequence -= removed_set

            VNF_placement_model.add_quadratic_constraints(
                tau_i[i] <= settings.M * (1-z[i]) + settings.r_i[i] for i in sequence
            )

            # # Number of same type VNF in a request constraint
            # for i in range(number_of_requests[0]):
            #     for f in settings.F:
            #         count = 0
            #         for l in range(len(settings.F_i[i])):
            #             if settings.F_i[i][l] == f:
            #                 count += 1
            #         VNF_placement_model.add_constraint(count <= 1, ctname="numbre_of_same_type_VNF_in_a_request")

            # Relation between z and x constraint
            VNF_placement_model.add_constraints((
                sum(x[i, f, v] for v in range(settings.number_of_nodes)) == z[i]
                for i in range(number_of_requests[0])
                for f in settings.F_i[i]),
                names="relation_between_z_and_x"
            )

            # Relation between y and x constraint
            VNF_placement_model.add_constraints((
                y[f, v] - x[i, f, v] >= 0
                for i in range(number_of_requests[0])
                for f in settings.F_i[i]
                for v in settings.nodes),
                names="relation_between_y_and_x"
            )

            # CPU capacity constraint
            for v in range(settings.number_of_nodes):
                occupied_cpu_resources = 0
                for i in range(number_of_requests[0]):
                    for f in settings.F_i[i]:
                        occupied_cpu_resources += x[i, f, v] * settings.cpu_f[f]
                VNF_placement_model.add_constraint(occupied_cpu_resources <= settings.cpu_v[v])
            
            # Memory capacity constraint
            for v in range(settings.number_of_nodes):
                occupied_mem_resources = 0
                for f in settings.F:
                    occupied_mem_resources += y[f, v]
                VNF_placement_model.add_constraint(occupied_mem_resources <= settings.mem_v[v])

            #-------------------------------------------------------------------------------------
            # Defineing the objective function
            #-------------------------------------------------------------------------------------

            obj_fn = sum(z[i] * settings.profit_i[i] for i in range(number_of_requests[0]))

            VNF_placement_model.set_objective('max', obj_fn)

            # Solve the model
            sol = VNF_placement_model.solve()
            if sol:
                cplex_res_value.append(sol.get_value(obj_fn))
            else:
                cplex_res_value.append(0)

            end_time = time.time()
            cplex_time_cost.append(end_time - start_time)

            # define input data for GA method
            class Data:
                number_of_VNF_types = number_of_VNF_types[nvt]
                number_of_requests = number_of_requests[0]
                number_of_nodes = settings.number_of_nodes
                F = settings.F
                G = settings.G
                number_of_nodes = settings.number_of_nodes
                nodes = settings.nodes
                cpu_v = settings.cpu_v
                mem_v = settings.mem_v
                cpu_f = settings.cpu_f
                F_i = settings.F_i
                profit_i = settings.profit_i
                r_i = settings.r_i
                s_i = settings.s_i
                e_i = settings.e_i
                number_of_individual = settings.number_of_individual
                number_of_gene_in_an_individual = settings.number_of_gene_in_an_individual
                elitism_rate = settings.elitism_rate
                iteration_for_one_ga = settings.iteration_for_one_ga
                maximum_of_iteration_for_one_ga_crossover = settings.maximum_of_iteration_for_one_ga_crossover
                maximum_of_iteration_for_one_ga_mutation = settings.maximum_of_iteration_for_one_ga_mutation
                number_of_individual_chose_from_population_for_tournament = settings.number_of_individual_chose_from_population_for_tournament
                crossover_rate = settings.crossover_rate
                mutation_rate = settings.mutation_rate

            # call other methods
            ga_res = my_ga_method.main(Data)
            random_res = my_random_method.main(Data)

            # result
            ga_res_value.append(ga_res["fittest_value"])
            ga_time_cost.append(ga_res["time_cost"])
            random_res_value.append(random_res["total_profit"])
            random_time_cost.append(random_res["time_cost"])

            mean_cplex_res_value += sol.get_value(obj_fn)
            mean_ga_res_value += ga_res["fittest_value"]
            mean_random_res_value += random_res["total_profit"]
            mean_cplex_time_cost += end_time - start_time
            mean_ga_time_cost += ga_res["time_cost"]
            mean_random_time_cost += random_res["time_cost"]

        mean_cplex_res_value /= number_of_iteration
        mean_ga_res_value /= number_of_iteration
        mean_random_res_value /= number_of_iteration
        mean_cplex_time_cost /= number_of_iteration
        mean_ga_time_cost /= number_of_iteration
        mean_random_time_cost /= number_of_iteration

        result_mean_cplex_res_value.append(mean_cplex_res_value)
        result_mean_ga_res_value.append(mean_ga_res_value)
        result_mean_random_res_value.append(mean_random_res_value)
        result_mean_cplex_time_cost.append(mean_cplex_time_cost)
        result_mean_ga_time_cost.append(mean_ga_time_cost)
        result_mean_random_time_cost.append(mean_random_time_cost)

    print("result_mean_cplex_res_value: ", result_mean_cplex_res_value)
    print("result_mean_ga_res_value:", result_mean_ga_res_value)
    print("result_mean_random_res_value:", result_mean_random_res_value)
    print("result_mean_cplex_time_cost: ", result_mean_cplex_time_cost)
    print("result_mean_ga_time_cost: ", result_mean_ga_time_cost)
    print("result_mean_random_time_cost: ", result_mean_random_time_cost)

    # line 1 points
    x1 = number_of_VNF_types
    y1 = result_mean_cplex_res_value
    plt.plot(x1, y1, 's-', color = 'r', label = "CPLEX", markersize = 8, linewidth = 2.5)
    
    # line 2 points
    x2 = number_of_VNF_types
    y2 = result_mean_ga_res_value
    plt.plot(x2, y2, 'o-', color = 'g', label = "GA", markersize = 8, linewidth = 2.5)

    # line 3 points
    x2 = number_of_VNF_types
    y2 = result_mean_random_res_value
    plt.plot(x2, y2, 'D-', color = 'b', label = "Random", markersize = 8, linewidth = 2.5)

    plt.xlabel('Number of VNF types')
    plt.ylabel('Profit')
    plt.title('number_of_iteration=50, iteration_for_one_ga=50')
    plt.legend()
    plt.show()
