# Long Nguyen
# 1001705873

import numpy as np

def foo_gradient(x, y):
    return (-np.sin(x) * np.cos(np.cos(x) + np.sin(2 * y)), 2 * np.cos(2 * y) * np.cos(np.cos(x) + np.sin(2 * y)))
def gradient_descent(function, gradient, x1, y1, eta, epsilon):
    t = 1
    xt, yt = x1, y1
    history = [(x1, y1)]

    while np.linalg.norm(gradient(xt, yt)) > epsilon:
        # compute the gradient
        grad = gradient(xt, yt)
        xt_next = xt - eta * grad[0]
        yt_next = yt - eta * grad[1]

        if function(xt_next, yt_next) > function(xt, yt):
            eta /= 2
            continue

        xt, yt = xt_next, yt_next
        history.append((xt, yt))
        t += 1
    return xt, yt, history