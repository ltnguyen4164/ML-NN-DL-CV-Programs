# Long Nguyen
# 1001705873

import numpy as np

def k_means(data_file, K, initialization):
    data = []
    
    # try to open data file and extract data
    try:
        with open(data_file) as file:
            for line in file:
                data_list = [float(value) for value in line.split()]
                data.append(data_list)
    except FileNotFoundError:
        print(f"Error: The file '{data_file}' does not exist.")
        quit()
    except Exception as e:
        print(f"An error occurred: {e}")
        quit()  
    data = np.array(data)
    
    # get dimensions and a # of points
    n, d = data.shape

    # initialize clusters
    if initialization == "random":
        cluster = np.random.randint(0, K, size=n)
    elif initialization == "round_robin":
        cluster = np.array([i % K for i in range(n)])
    
    # compute mean of cluster
    center = np.array([data[cluster == k].mean(axis=0) for k in range(K)])

    # main loop that alternates between computing new assignments and means
    # until break condition
    while True:
        new = np.zeros(n, dtype=int)

        # assign/reassign each objects to clusters based on distances from 
        # each object to the mean of each current cluster
        for i in range(n):
            distance = np.linalg.norm(data[i] - center, axis=1)
            new[i] = np.argmin(distance)
        # check for break condition
        if np.array_equal(new, cluster):
            break
        # recompute the means of the clusters
        cluster = new
        for k in range(K):
            if np.any(cluster == k):
                center[k] = data[cluster == k].mean(axis=0)
            else:
                center[k] = center[k]
        
    # output final cluster assignments
    for i in range(n):
        cluster_id = cluster[i] + 1
        data_point = data[i]

        if d == 1: # 1-D case
            print('%10.4f --> cluster %d' % (data_point[0], cluster_id))
        elif d == 2: # 2-D case
            print('(%10.4f, %10.4f) --> cluster %d' % (data_point[0], data_point[1], cluster_id))
        else: # other case
            continue    