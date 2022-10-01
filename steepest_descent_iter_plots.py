import numpy as np
import matplotlib.pyplot as plt
from util import generate_problem, generate_problem_fix_r


def grad(Q, b, x_k):
    return np.matmul(Q, x_k) - b


def norm(v):
    return np.sqrt(np.sum(np.square(v)))


def steepest_descent(Q, b, n, eps, *, seed=None):
    if seed is None:
        seed = np.random.randint(1, 10000)
    rs = np.random.RandomState(seed=seed)
    x = rs.rand(n).reshape(-1, 1)
    print("Q:\n{}\nb:\n{}\nstarting x:\n{}\n".format(Q, b, x))
    # x in R^n, with each component iid distributed in (0, 1)
    g = grad(Q, b, x)
    MM = np.matmul # shorthand
    niter = 0
    while norm(g) >= eps:
        niter += 1
        g = grad(Q, b, x)
        alpha = MM(MM(MM(x.T, Q), Q), x) - 2 * MM(MM(x.T, Q), b) + MM(b.T, b)
        alpha /= MM(MM(MM(g.T, Q), Q), x) - MM(MM(g.T, Q), b)
        x = x - alpha * g
    return x, niter


def plot1():
    eps = 1e-3
    ns = range(2, 101)
    niters = []
    for n in ns:
        problem = generate_problem(n)
        Q, b = problem['Q'], problem['b']
        _, niter = steepest_descent(Q, b, n, eps)
        niters.append(niter)
    plt.figure(figsize=(10,10))
    plt.plot(ns,niters,'C9-.',label='Number of iterations taken to converge')
    plt.xlabel('n')
    plt.grid(axis='y')
    plt.ylabel('iterations')
    plt.savefig('niter.png')
    plt.show()


def plot2():
    n = 30
    eps = 1e-3
    niters = []
    for run in range(500):
        problem = generate_problem(n)
        Q, b = problem['Q'], problem['b']
        _, niter = steepest_descent(Q, b, n, eps)
        niters.append(niter)
    plt.figure(figsize=(10,10))
    plt.hist(niters,rwidth=0.9,color='C4')
    plt.xlabel('iterations')
    plt.title('How many iterations it takes for SD to converge for n = {}'.format(n))
    plt.savefig('niter_fix_n_{}.png'.format(n))
    plt.show()


def plot3():
    eps = 1e-3
    ns = range(10, 101)
    niters = []
    for n in ns:
        problem = generate_problem_fix_r(n)
        Q, b = problem['Q'], problem['b']
        _, niter = steepest_descent(Q, b, n, eps)
        niters.append(niter)
    plt.figure(figsize=(10,10))
    plt.plot(ns,niters,'C3-.',label='Number of iterations taken to converge')
    plt.xlabel('n')
    plt.grid(axis='y')
    plt.ylabel('iterations')
    plt.savefig('niter_fix_r.png')
    plt.show()


def plot4():
    eps = 1e-3
    ns = range(10, 101)
    niters = []
    for n in ns:
        problem = generate_problem_fix_r(n,good=True)
        Q, b = problem['Q'], problem['b']
        _, niter = steepest_descent(Q, b, n, eps)
        niters.append(niter)
    plt.figure(figsize=(10,10))
    plt.plot(ns,niters,'C3-.',label='Number of iterations taken to converge')
    plt.xlabel('n')
    plt.grid(axis='y')
    plt.ylabel('iterations')
    plt.savefig('niter_fix_r_good.png')
    plt.show()


def main():
    plot1()
    plot2()
    plot3()
    plot4()


if __name__=="__main__":
    main()
