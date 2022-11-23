import numpy as np


def DFP(Q, b, x, H, k):
    r"""Running DFP for Quadratic case, 1/2 x.T Q x - b.T x
    """
    grad = lambda x: Q@x - b # Q, x and b are numpy matrices of correct shape
    f = lambda x: (0.5 * x.T@Q@x - b.T@x).reshape(-1)
    print(F"{f(x) = } {x.reshape(-1) = }")
    for i in range(k):
        g = grad(x)
        d = -H@g
        a = -g.T@d/(d.T@Q@d)
        x = x + a * d
        dx = a * d
        dg = grad(x) - g
        Hg = H@dg
        H = H + dx@dx.T/(dx.T@dg) - Hg@Hg.T/(dg.T@H@dg)
        print(F"{i} {f(x) = } {x.reshape(-1) = }")


def main():
    np.set_printoptions(formatter={'float':'{:.7f}'.format})
    Q = np.array([[2, 0],[0, 20]])
    H = Q/2
    x = np.array([[1],[1]])
    b = np.array([[0],[0]])
    k = 2
    DFP(Q,b,x,H,k)


if __name__=="__main__":
    main()
