from value_iteration import value_iteration

# When you test your code, you can select the function arguments you 
# want to use by modifying the next lines

#data_file = "data/environment1.txt"
data_file = "/Users/ltngu/Documents/CSE Files/CSE 4309/assignment 8/data/environment2.txt"
ntr = -0.04 # non_terminal_reward
gamma = 0.9
K = 20


value_iteration(data_file, ntr, gamma, K)
