import argparse
import numpy as np
from util import generate_problem


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",default=100,type=int)
    parser.add_argument("-eps",default=1e-3,type=float)
    parser.add_argument("-seed",default=None,type=int)
    args = parser.parse_args()
    n = args.n
    eps = args.eps
    print("Got args: n = {}, eps = {}, seed = {}".format(n, eps, args.seed))
    if eps < 1e-5:
        eps = 1e-4
        # gets too slow
        print("Setting eps = {} (otherwise gets too slow)".format(eps))

    problem = generate_problem(n, seed=args.seed)
    Q, b = problem['Q'], problem['b']

    x, niter = steepest_descent(Q, b, n, eps, seed=args.seed)
    print("x* :\n{}".format(x))

    x_star = np.matmul(np.linalg.inv(Q), b)
    print("Actual x* :\n{}".format(x_star))
    print("It took niter = {} iterations to reach this point".format(niter))

    err = norm(np.abs(x_star - x))/norm(x_star)
    print("Error = {}".format(err))


if __name__=="__main__":
    main()
