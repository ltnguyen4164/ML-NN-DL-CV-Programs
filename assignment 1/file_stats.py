# Long Nguyen
# 1001705873

import sys
import math

# Enter the command line: python3 file_stats.py pathname (where the pathname is the path to the text file)

# Function that calculates the mean and standard deviation (stdev)
def calculate(data):
	mean = 0
	var = 0
	stdev = 0
	
	# i represents the column number, j represents the row number
	for i in range(0, len(data[0])):
		# reset mean and var back to zero
		mean = 0
		var = 0
		
		# Calculate the mean
		for j in range(0, len(data)):
			mean += data[j][i]
		mean = mean / len(data)
		
		# Calculate the variance
		for j in range(0, len(data)):
			diff = mean - data[j][i]
			sqr = pow(diff, 2)
			var += sqr
		var = var / (len(data) - 1)
		
		# Calculate the standard deviation
		stdev = math.sqrt(var)
		
		print("Column %d: mean = %.4f std = %.4f" % (i+1, mean, stdev))
		

# Get path name from command line argument
pathname = sys.argv[1]
data = []

# Check if file or path exist
try:
	with open(pathname, "r") as file:
		for line in file:
			arr = [float(value) for value in line.split()]
			data.append(arr)
except FileNotFoundError:
	print(f"Error: The file at '{pathname}' does not exist.")
	quit()
except Exception as e:
    print(f"An error occurred: {e}")
    quit()
    
calculate(data)
