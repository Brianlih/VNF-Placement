import random
import settings, pre_settings

def init(number_of_requests, number_of_VNF_types, number_of_nodes, seed):
    global F_i, profit_i, r_i, s_i, e_i

    lower_bound_of_F_i = 1
    upper_bound_of_F_i = 3
    lower_bound_of_r_i = 10
    upper_bound_of_r_i = 30

    s = seed
    F_i = []
    for i in range(number_of_requests):
        tmp = []
        random.seed(s)
        number_of_needed_VNF_by_request_i = random.randint(lower_bound_of_F_i, upper_bound_of_F_i)
        random.seed(s)
        tmp = random.sample(settings.F, number_of_needed_VNF_by_request_i)
        F_i.append(tmp)
        s += 1
    # print("F_i = ", F_i)

    psi_f = []
    count = 0
    for i in range(number_of_VNF_types):
        for j in range(len(F_i)):
            for k in range(len(F_i[j])):
                if F_i[j][k] == settings.F[i]:
                    count += 1
                    break
        psi_f.append(count / number_of_requests)
        count = 0
    # print("psi_f = ", psi_f)

    profit_i = []
    for i in range(number_of_requests):
        profit = 0
        for j in range(len(F_i[i])):
            profit += settings.eta_f[F_i[i][j]] * (1 + psi_f[F_i[i][j]]) * settings.cpu_f[F_i[i][j]]
        profit_i.append(profit)
    # print("profit_i = ", profit_i)

    s = seed
    r_i = []
    for i in range(number_of_requests):
        random.seed(s)
        r_i.append(random.randint(lower_bound_of_r_i, upper_bound_of_r_i))
        s += 1
    # print("r_i = ", r_i)

    s = seed
    s_i = []
    for i in range(number_of_requests):
        random.seed(s)
        s_i.append(random.randint(0, number_of_nodes - 1))
        s += 1
    # print("s_i = ", s_i)

    s = seed
    t = 123
    e_i = []
    for i in range(number_of_requests):
        random.seed(s)
        buffer = random.randint(0, number_of_nodes - 1)
        while(buffer == s_i[i]):
            random.seed(t)
            buffer = random.randint(0, number_of_nodes - 1)
            t += 1
        e_i.append(buffer)
        s += 1
    # print("e_i = ", e_i)
    # print("==================================================================================")