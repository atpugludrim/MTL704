import sys
import argparse
import autograd.numpy as np
from autograd import grad
import functions
import matplotlib.pyplot as plt


def norm(v):
    return np.sqrt(np.sum(np.square(v)))


def estimate_alpha_k(f, x_k, alpha_hat, gradient_xk):
    x_k_hat = x_k - alpha_hat * gradient_xk
    f_k_hat = f(x_k_hat)
    norm_g_sq = np.square(norm(gradient_xk))
    alpha_k = norm_g_sq * alpha_hat * alpha_hat
    alpha_k /= 2 * (f_k_hat - f(x_k) + alpha_hat * norm_g_sq)
    return alpha_k


def conjugate_gradient(f, *, n, eps, alpha_hat, seed=None):
    if seed is None:
        seed = np.random.randint(1, 1000)
    rs = np.random.RandomState(seed=seed)
    x = rs.randn(n).reshape(-1, 1)
    fs = [f(x)]
    print("starting x:\n{}\n".format(x))
    gradf = grad(f) # f needs to be a pure function
    g = gradf(x)
    d = -g
    MM = np.matmul
    niter = 0
    while norm(g) >= eps:
        if niter > 10000:
            print("Did not converge in 10000 iterations")
            break
        niter += 1
        g = gradf(x)
        alpha_hat = estimate_alpha_k(f, x, alpha_hat, g)
        x = x + alpha_hat * d
        g_plus_1 = gradf(x)
        beta = MM(g_plus_1.T,g_plus_1)/MM(g.T,g)
        d = -g_plus_1+beta*d
        if niter % n == 0:
            d = -g_plus_1
        fs.append(f(x))
    return x, niter, fs


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("Arguments")
    group.add_argument("-list",default=False,action='store_true')
    group.add_argument("-function","-f",type=str)
    group.add_argument("-n",type=int)
    group.add_argument("-eps",type=float,default=1e-3)
    group.add_argument("-seed",type=int,default=None)
    group.add_argument("-alpha_hat",type=float,default=1e-3)
    group.add_argument("-save_graph",default=False,action='store_true')
    args = parser.parse_args()
    if args.list:
        print("------ Known functions: -------")
        print("\n".join(k for k in dir(functions) if not k.startswith('__') and k != 'np'))
        sys.exit()
    elif args.function is None or args.n is None:
        parser.print_usage()
        print("{}: error: the following arguments are required: -function/-f, -n"\
                .format(sys.argv[0]))
        sys.exit(1)
    n = args.n
    eps = args.eps
    seed = args.seed
    alpha_hat = args.alpha_hat
    function_name = args.function
    print("Got args: n = {}, eps = {}, seed = {}, alpha_hat = {}, function = {}".format(
        n,
        eps,
        seed,
        alpha_hat,
        function_name))
    # define how to get function here
    if function_name in dir(functions):
        func = getattr(functions, function_name)
    else:
        print("------ Unknown function: ------")
        print(function_name)
        print("------ Known functions: -------")
        print("\n".join(k for k in dir(functions) if not k.startswith('__') and k != 'np'))
        sys.exit(1)
    function = func(n)
    x, niter, fs = conjugate_gradient(
            function,
            n=n,
            eps=eps,
            seed=seed,
            alpha_hat=alpha_hat)
    print(x, niter)
    if args.save_graph:
        figure = plt.figure(figsize=(10,7))
    plt.plot(range(1, niter + 2),fs,label='Function value')
    plt.xlabel('iteration number')
    plt.ylabel('function value')
    plt.title('Plot for {} function'.format(function_name))
    if args.save_graph:
        plt.savefig('SD_on_{}.png'.format(function_name))
    plt.show()


if __name__=="__main__":
    main()
