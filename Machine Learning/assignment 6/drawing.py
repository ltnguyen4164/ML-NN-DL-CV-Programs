import matplotlib.pyplot as plot
import numpy as np

"""
draw_points(data):
- data is a numpy 2D array, of shape (dimensions, number), where every 
  column is a data point. The number of rows is equal to the number of 
  dimensions on the dataset. The number of columns is simply the number 
  of points in the dataset. The code can only handle 1D and 2D datasets.
  If the number of dimensions is higher, only the first two 
  dimensions are drawn.
"""
def draw_points(data):
    (fig, ax) = plot.subplots()
    (dimensions, number) = data.shape
    if (dimensions == 1):
        ax.plot(data[0,:], np.zeros((number)),'o')
    else:
        ax.plot(data[0,:], data[1,:],'o')


"""
draw_assignments(data, assignments):
- data is a numpy 2D array, of shape (dimensions, number), where every 
  column is a data point. The number of rows is equal to the number of 
  dimensions on the dataset. The number of columns is simply the number 
  of points in the dataset. The code can only handle 1D and 2D datasets.
  If the number of dimensions is higher, only the first two 
  dimensions are drawn.
- assignments is a 1D numpy array, and assignments[i] specifies the cluster
  that data[:,i] (the i-th data point) belongs to.
"""
def draw_assignments(data, assignments):
    K = np.max(assignments)
    clustering = [[] for c in range(0,K+1)]
    for c in range (0, K+1):
        clustering[c] = [data[:,i] for i in (assignments == c).nonzero()[0]]
    
    draw_clustering (clustering)


def draw_clustering(clustering):
    (fig, ax) = plot.subplots()
    number = len(clustering)
    for i in range (0, number):
        cluster = clustering[i]
        m = len(cluster)
        x = []
        y = []
        for j in range (0, m):
            v = cluster[j]
            x = x + [v[0]]
            if (len(v) > 1):
                y = y + [v[1]]
            else:
                y = y + [0]
        ax.plot(x, y,'o')

