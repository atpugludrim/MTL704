import sys
import argparse
import autograd.numpy as np
from autograd import grad
import functions


def norm(v):
    return np.sqrt(np.sum(np.square(v)))


def steepest_descent(f, *, n, eps, alpha_hat, seed=None):
    if seed is None:
        seed = np.random.randint(1, 1000)
    rs = np.random.RandomState(seed=seed)
    x = rs.randn(n).reshape(-1, 1)
    print("starting x:\n{}\n".format(x))
    gradf = grad(f) # f needs to be a pure function
    g = gradf(x)
    MM = np.matmul
    niter = 0
    while norm(g) >= eps:
        if niter > 10000:
            print("Did not converge in 10000 iterations")
            break
        niter += 1
        g = gradf(x)
        x = x - alpha_hat * g
    return x, niter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-function","-f",type=str,required=True)
    parser.add_argument("-n",type=int,required=True)
    parser.add_argument("-eps",type=float,default=1e-3)
    parser.add_argument("-seed",type=int,default=None)
    parser.add_argument("-alpha_hat",type=float,default=1e-3)
    args = parser.parse_args()
    n = args.n
    eps = args.eps
    seed = args.seed
    alpha_hat = args.alpha_hat
    function = args.function
    print("Got args: n = {}, eps = {}, seed = {}, alpha_hat = {}, function = {}".format(
        n,
        eps,
        seed,
        alpha_hat,
        function))
    # define how to get function here
    if function in dir(functions):
        func = getattr(functions, function)
    else:
        print("-------Unknown function: ------")
        print(function)
        print("-------Known functions: -------")
        print("\n".join(k for k in dir(functions) if not k.startswith('__') and k != 'np'))
        sys.exit(1)
    function = func(n)
    x, niter = steepest_descent(
            function,
            n=n,
            eps=eps,
            seed=seed,
            alpha_hat=alpha_hat)
    print(x, niter)


if __name__=="__main__":
    main()
