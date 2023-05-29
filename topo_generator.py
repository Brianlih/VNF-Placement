import random
import networkx as nx

def check_link_count(adjacency_matrix, number_of_nodes):
    link_count = 0
    checked_adjacency_matrix = [[False]*number_of_nodes for i in range(number_of_nodes)]
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if adjacency_matrix[i][j] == 1 and checked_adjacency_matrix[i][j] == False:
                link_count += 1
                checked_adjacency_matrix[i][j] = True
    return link_count

if __name__ == "__main__":
    number_of_nodes = 15
    number_of_links = 23
    link_count = 0
    adjacency_matrix = [[0]*number_of_nodes for i in range(number_of_nodes)]
    
    while link_count < number_of_links:
        node1 = random.randint(0,number_of_nodes - 1)
        while True:
            node2 = random.randint(0,number_of_nodes - 1)
            if node1 != node2 and adjacency_matrix[node1][node2] == 0:
                break
        adjacency_matrix[node1][node2] = 1
        link_count = check_link_count(adjacency_matrix, number_of_nodes)
    
    count = 0
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if adjacency_matrix[i][j] == 1:
                print(count, i, j)
                count += 1
    

    
