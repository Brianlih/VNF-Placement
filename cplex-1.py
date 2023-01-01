from docplex.mp.model import Model
import settings

if __name__ == "__main__":
    #------------------------------------------------------------------------------------------
    # Initialize the input data
    #------------------------------------------------------------------------------------------
    
    settings.init()

    #------------------------------------------------------------------------------------------
    # Creating the model
    #------------------------------------------------------------------------------------------

    VNF_placement_model = Model("VNF_placement")

    #------------------------------------------------------------------------------------------
    # Creating decsision variables
    #------------------------------------------------------------------------------------------

    z = VNF_placement_model.binary_var_dict(settings.number_of_requests, name="z")
    x = VNF_placement_model.binary_var_dict((
        (i, f, v)
        for i in range(settings.number_of_requests)
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
    for i in range(settings.number_of_requests):
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
    for i in range(settings.number_of_requests):
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
    for i in range(settings.number_of_requests):
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
    for i in range(settings.number_of_requests):
        sequence.add(i)
    for i in range(settings.number_of_requests):
        if len(settings.F_i[i]) <= 1:
            VNF_placement_model.add_constraint(tau_i[i] <= settings.M * (1-z[i]) + settings.r_i[i])
            removed_set.add(i)
    sequence -= removed_set

    VNF_placement_model.add_quadratic_constraints(
        tau_i[i] <= settings.M * (1-z[i]) + settings.r_i[i] for i in sequence
    )

    # # Number of same type VNF in a request constraint
    # for i in range(settings.number_of_requests):
    #     for f in settings.F:
    #         count = 0
    #         for l in range(len(settings.F_i[i])):
    #             if settings.F_i[i][l] == f:
    #                 count += 1
    #         VNF_placement_model.add_constraint(count <= 1, ctname="numbre_of_same_type_VNF_in_a_request")

    # Relation between z and x constraint
    VNF_placement_model.add_constraints((
        sum(x[i, f, v] for v in range(settings.number_of_nodes)) == z[i]
        for i in range(settings.number_of_requests)
        for f in settings.F_i[i]),
        names="relation_between_z_and_x"
    )

    # Relation between y and x constraint
    VNF_placement_model.add_constraints((
        y[f, v] - x[i, f, v] >= 0
        for i in range(settings.number_of_requests)
        for f in settings.F_i[i]
        for v in settings.nodes),
        names="relation_between_y_and_x"
    )

    # CPU capacity constraint
    for v in range(settings.number_of_nodes):
        occupied_cpu_resources = 0
        for i in range(settings.number_of_requests):
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

    obj_fn = sum(z[i] * settings.profit_i[i] for i in range(settings.number_of_requests))

    # print(VNF_placement_model.print_information())

    VNF_placement_model.set_objective('max', obj_fn)

    # Solve the model and output the solution
    sol = VNF_placement_model.solve()
    if sol:
        print(sol)
    else:
        print("No solution found")
