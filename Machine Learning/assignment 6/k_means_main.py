from k_means import *


data_file = "/Users/ltngu/Documents/CSE Files/CSE 4309/assignment 6/toy_data/set1a.txt"
#data_file = "/Users/ltngu/Documents/CSE Files/CSE 4309/assignment 6/toy_data/toy_data/set1a.txt"
#data_file = "toy_data/set2_1.txt"

K = 3
#initialization = "random"
initialization = "round_robin"
#iterations = 2

k_means(data_file, K, initialization)
#em_cluster(data_file, K, initialization, iterations)