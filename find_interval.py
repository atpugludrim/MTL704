import numpy as np


def find_interval(f):
    alpha_0 = np.random.randint(-10,10)
    print(alpha_0)
    fa0 = f(alpha_0)
    start_h = 0.1
    h = start_h
    alpha_1 = alpha_0 + h
    n_iter = 5000
    n = 0
    while f(alpha_1) < fa0 and n < n_iter:
        h = h * 2
        alpha_1 = alpha_0 + h
        n += 1
    U = alpha_1
    alpha_0 = alpha_1
    h = -start_h
    alpha_1 = alpha_0 + h
    fa0 = f(alpha_0)
    n = 0
    while f(alpha_1) < fa0 and n < n_iter:
        h = h * 2
        alpha_1 = alpha_0 + h
        n += 1
    L = alpha_1
    return L, U


def find_interval_2(f):
    alpha_0 = np.random.randint(-10,10)
    h = 1e-2
    # reverse direction if forward is increasing
    mult = 1
    if f(alpha_0) < f(alpha_0+h):
        h = -h
    while f(alpha_0) > f(alpha_0+mult*h):
        alpha_0 = alpha_0 + mult * h
        mult = mult * 2
    bounds = [alpha_0]
    h = -h/10
    mult = 1
    if f(alpha_0) < f(alpha_0+h):
        h = -h
    while f(alpha_0) > f(alpha_0+mult*h):
        alpha_0 = alpha_0 + mult * h
        mult = mult * 2
    bounds.append(alpha_0)
    return min(bounds), max(bounds)


if __name__=="__main__":
    def minimization_func(x):
        return x * x + 1
    interval = list(find_interval_2(minimization_func))
    print("Interval:",interval, "\nFunc values:",list(map(minimization_func, interval)))
