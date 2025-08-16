import numpy as np
import pylab
import matplotlib.pyplot as plt

# This is where we import your functions.
from gradient_descent_solution import foo_gradient, gradient_descent 


# An auxiliary function, that draws an image visualizing the function 
# output. It is assumed that argument "function" takes two real numbers
# (x and y) as arguments. The image shows output values when x ranges
# between x_min and x_max, and y ranges between y_min and y_max. The 
# image uses a color code where red corresponds to low values
# and blue corresponds to high values.
def visualize(function, x_min, x_max, y_min, y_max):
    plt.figure()
    steps = 400
    x_step = (x_max - x_min) / (steps - 1)
    y_step = (y_max - y_min) / (steps - 1)
    x = np.arange(x_min, x_max, x_step)
    y = np.arange(y_min, y_max, y_step)
    
    (X,Y) = pylab.meshgrid(x, y)
    Z = function(X, Y)
    image = plt.imshow(Z,cmap=plt.cm.RdBu)   
    plt.colorbar(image)


# useful for printing the history of gradient descent.
def print_history(function, history):
    for i in range(0, len(history)):
        t = i+1
        (xt, yt) = history[i]
        print("t = %2d, xt = %7.4f, yt = %7.4f, function(xt, yt) = %7.4f" % 
              (t, xt, yt, function(xt, yt)))

# This is the function whose gradient you need to compute (and write code for).
def foo(x, y):
    result = np.sin(np.cos(x)+np.sin(2*y))
    return result


# This is the function we used as an example in the gradient descent slides.
def f1(x, y):
    result = x*x + 2*y*y - 600*x - 800*y + x*y + 50
    return result


# This is the gradient of the function we used as an example in the 
# gradient descent slides. Those slides explain how we computed this gradient.
def f1_gradient(x, y):
    dfdx = 2*x - 600 + y
    dfdy = 4*y - 800 + x
    return (dfdx, dfdy)
    
    



# visualize values of f1 for x in [0, 400] and y in [0, 400]. Note that
# the color code used here is different thani the color code used in the slides.
# Still, the image produced here should make it easy to identify the 
# minimum point.
visualize(f1, 0, 400, 0, 400)

# visualize values of foo for x in [0, 10] and y in [0, 10]. The local 
# minima are easy to spot
visualize(foo, 0, 10, 0, 10)

# These two lines are sample test cases using your two functions:
# foo_gradient, and gradient_descent.
(x_min, y_min, history) = gradient_descent(f1, f1_gradient, 300, 150, 1, 0.001)
print_history(f1, history)

(x_min, y_min, history) = gradient_descent(foo, foo_gradient, 2, 1, 1, 0.001)
print_history(foo, history)




