#import time
import numpy as np


def generate_sym_pd_matrix(n, rs):
    R = rs.randint(-1000, 1000, (n, n))
    q, _ = np.linalg.qr(R, mode='complete')
    eigs = rs.permutation(1000)[:n] + 1
    A = np.matmul(np.matmul(q, np.diag(eigs)), q.T)
    return A


def generate_problem(n, *, seed=None):
    if seed is None:
        seed = np.random.randint(1,10000)
    rs = np.random.RandomState(seed=seed)
    Q = generate_sym_pd_matrix(n, rs)
    b = rs.randint(-1000, 1000, (n, 1))
    return {'Q': Q, 'b': b}


def generate_sym_pd_matrix_fix_r(n, rs):
    R = rs.randint(-1000, 1000, (n, n))
    q, _ = np.linalg.qr(R, mode='complete')
    eigs = rs.permutation(100)[:(n-2)] + 35
    eigs = eigs.tolist()
    eigs.extend([1, 1000]) # min and max eig val. ILL Conditioned
    eigs = np.asarray(eigs)
    A = np.matmul(np.matmul(q, np.diag(eigs)), q.T)
    return A


def generate_sym_pd_matrix_fix_r_good(n, rs):
    R = rs.randint(-1000, 1000, (n, n))
    q, _ = np.linalg.qr(R, mode='complete')
    eigs = np.ones(n-1)
    eigs = eigs.tolist()
    eigs.append(1.2)
    eigs = np.asarray(eigs)
    A = np.matmul(np.matmul(q, np.diag(eigs)), q.T)
    return A


def generate_problem_fix_r(n, *, seed=None, good=False):
    if seed is None:
        seed = np.random.randint(1,10000)
    rs = np.random.RandomState(seed=seed)
    if good:
        Q = generate_sym_pd_matrix_fix_r_good(n, rs)
    else:
        Q = generate_sym_pd_matrix_fix_r(n, rs)
    b = rs.randint(-1000, 1000, (n, 1))
    return {'Q': Q, 'b': b}


# def is_pos_def(x):
#     return np.all(np.linalg.eigvals(x) > 0)
# 
# 
# def main():
#     s = time.perf_counter()
#     M=generate_sym_pd_matrix(100)
#     e = time.perf_counter()
#     print("{}\n{}\ndone in {:.3f}s".format(M,is_pos_def(M),e-s))
# 
# 
# if __name__=="__main__":
#     main()
