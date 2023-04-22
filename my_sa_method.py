import random, time
from copy import deepcopy
import settings

def find_new_solution(data):
    pass

def main(data_from_cplex, my_greedy_sol):
    data = data_from_cplex

    current_temperature = 1000
    final_temperature = 0.01
    cooling_rate = 0.99

    while current_temperature > final_temperature:
        find_new_solution(my_greedy_sol, data)
