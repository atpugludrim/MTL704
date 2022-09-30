import numpy as np
import matplotlib.pyplot as plt
from util import generate_problem


def grad(Q, b, x_k):
    return np.matmul(Q, x_k) - b


def norm(v):
    return np.sqrt(np.sum(np.square(v)))


def steepest_descent(Q, b, n, eps, *, seed=None):
    if seed is None:
        seed = np.random.randint(1, 10000)
    rs = np.random.RandomState(seed=seed)
    xs = [] # the history
    x = rs.rand(n).reshape(-1, 1)
    xs.append(x)
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
        xs.append(x)
    return xs


def main():
    n = 2
    eps = 1e-3
    seed = 5#np.random.randint(1,10000)
    print("Got args: n = {}, eps = {}, seed = {}".format(n, eps, seed))

    problem = generate_problem(n, seed=seed)
    Q, b = problem['Q'], problem['b']

    xs = steepest_descent(Q, b, n, eps, seed=seed)
    x_star = np.matmul(np.linalg.inv(Q), b)

    n = [500 for _ in range(4)]
    n.append(1000)
    n.append(1400)
    n.append(1500)
    lims = [[[-0.4,1],[0,1]],
            [[0,0.3],[0,1]],
            [[0.16,0.18],[0.447,0.452]],
            [[0.1632,0.1637],[0.4496,0.4499]],
            [[0.1632,0.1637],[0.4496,0.4499]],
            [[0.1632,0.1637],[0.4496,0.4499]],
            [[0.1632,0.1637],[0.4496,0.4499]],
            ]
    # values are fine tuned for problem generated when seed=5
    for i, x in enumerate(xs):
        xx = np.linspace(*lims[i][0], n[i])
        yy = np.linspace(*lims[i][1], n[i])
        xx, yy = np.meshgrid(xx, yy)
        fk = 0.5 * np.matmul(np.matmul(x.T,Q), x) - np.matmul(b.T, x)
        fk = fk.item()
        print(fk)
        F = np.zeros((n[i], n[i]))
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                x1 = xx[i, j]
                x2 = yy[i, j]
                F[i, j] = 0.5 * x1 * x1 * Q[0, 0] + x1 * x2 * Q[0, 1] + 0.5 * x2 * x2 * Q[1, 1] - b[0, 0] * x1 - b[1, 0] * x2 - fk
        plt.contour(xx,yy,F,[0])
    xs = np.asarray(xs)
    plt.plot(xs[:,0,0], xs[:,1,0], 'C1--', marker='.')
    plt.gca().set_aspect('equal')
    plt.gca().set_xlim(-0.4,1)
    plt.show()


if __name__=="__main__":
    main()
