import random
import settings, pre_settings

def init(number_of_requests, number_of_VNF_types):
    global F_i, profit_i, r_i, s_i, e_i

    lower_bound_of_F_i = 2
    upper_bound_of_F_i = 4
    lower_bound_of_r_i = 2 * pre_settings.lower_bound_of_pi_wv
    upper_bound_of_r_i = 2 * pre_settings.upper_bound_of_pi_wv

    F_i = []
    for i in range(number_of_requests):
        tmp = []
        number_of_needed_VNF_by_request_i = random.randint(lower_bound_of_F_i, upper_bound_of_F_i)
        tmp = random.sample(settings.F, number_of_needed_VNF_by_request_i)
        F_i.append(tmp)
    print("F_i = ", F_i)

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
    print("psi_f = ", psi_f)

    profit_i = []
    for i in range(number_of_requests):
        profit = 0
        for j in range(len(F_i[i])):
            profit += settings.eta_f[F_i[i][j]] * (1 + psi_f[F_i[i][j]]) * settings.cpu_f[F_i[i][j]]
        profit_i.append(profit)
    print("profit_i = ", profit_i)

    r_i = []
    for i in range(number_of_requests):
        r = random.randint(lower_bound_of_r_i, upper_bound_of_r_i)
        r_i.append(r * len(F_i[i]))
    print("r_i = ", r_i)

    s_i = []
    for i in range(number_of_requests):
        s_i.append(random.randint(0, pre_settings.number_of_nodes - 1))
    print("s_i = ", s_i)

    e_i = []
    buffer = random.randint(0, pre_settings.number_of_nodes - 1)
    for i in range(number_of_requests):
        while(buffer == s_i[i]):
            buffer = random.randint(0, pre_settings.number_of_nodes - 1)
        e_i.append(buffer)
    print("e_i = ", e_i)
    print("==================================================================================")