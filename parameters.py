M = 1000000
number_of_VNF_types = 5
number_of_requests = 15
number_of_nodes = 4
number_of_topo = 1
F =  [0, 1, 2, 3, 4]
edge_weights =  [2, 8, 6, 5]
number_of_nodes =  4
nodes =  [0, 1, 2, 3]
cpu_v =  [8, 7, 9, 10]
mem_v =  [27, 27, 26, 27]
eta_f =  [2, 3, 1, 2, 2]
cpu_f =  [2, 5, 2, 5, 2]
mem_f =  [29, 23, 27, 25, 24]
psi_f =  [0.4, 0.4, 0.4666666666666667, 0.6666666666666666, 0.26666666666666666]
F_i =  [[1, 3], [0, 3], [2], [1, 4, 3, 0, 2], [0], [2, 3], [3, 2, 4, 0], [3], [3], [2, 3, 4, 1, 0], [1, 4], [2], [1], [3, 0, 2], [3, 1]]
r_i =  [73, 59, 53, 75, 75, 75, 71, 79, 53, 54, 70, 58, 54, 56, 70]
s_i =  [1, 2, 2, 2, 2, 2, 1, 1, 0, 3, 1, 2, 0, 3, 0]
e_i =  [2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 1, 1, 1]

edges =  [(0, 1), (0, 2), (0, 0), (1, 3), (1, 1), (2, 3), (2, 2), (3, 3)]

# Model: VNF_placement
#  - number of variables: 167
#    - binary=167, integer=0, continuous=0
#  - number of constraints: 236
#    - linear=227, quadratic=9
#  - parameters: defaults
#  - objective: none
#  - problem type is: MIQCP

# solution for: VNF_placement
objective = 96.3333
z_0=1
z_12=1
z_14=1
beta_0_1_3=1
beta_0_3_0=1
beta_12_1_3=1
beta_14_3_2=1
beta_14_1_1=1
delta_1_1=1
delta_1_3=1
delta_3_0=1
delta_3_2=1
delta_4_0=0
delta_4_1=0
delta_4_2=0
delta_4_3=0