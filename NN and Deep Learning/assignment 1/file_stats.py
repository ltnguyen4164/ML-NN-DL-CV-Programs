# Long Nguyen
# 1001705873

import sys
import numpy as np

# Function that calculates the mean and standard deviation (stdev)
def calculate(data):
	# Convert data into numpy array
	data = np.array(data)

	# Compute mean and stdev
	for i in range(data.shape[1]):
		col = data[:, i]
		mean = np.mean(col)
		stdev = np.std(col, ddof=1)
		print("Column %d: mean = %.4f std = %.4f" % (i+1, mean, stdev))
		

# Get path name from command line argument
pathname = sys.argv[1]
data = []

# Open the path name of the data file
with open(pathname, "r") as file:
	for line in file:
		arr = [float(value) for value in line.split()]
		data.append(arr)
    
calculate(data)