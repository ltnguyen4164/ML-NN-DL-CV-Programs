# Long Nguyen
# 1001705873

import numpy as np

def em_cluster(data_file, K, initialization, iterations):
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
