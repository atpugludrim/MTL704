def find_n(L, U, e):
    a = 1
    b = 1
    # starting from 2
    F = a + b
    n = 2
    I = U - L
    while (I / F - e) >= 0:
        n += 1
        a = b
        b = F
        F = a + b
    Fn_1 = b
    return n, F, Fn_1


def find_pq(*, L=None, U=None, I_k_plus_1):
    if (L is None and U is None) or (L is not None and U is not None):
        raise ValueError("Exactly one of L and U should be specified")
    if L is None:
        return {'p': U - I_k_plus_1}
    elif U is None:
        return {'q': L + I_k_plus_1}


def reduce_interval(f, *, L, p, q, U, fp, fq):
    I_k_plus_2 = U - q
    if fp < fq:
        # keep left
        U = q
        q = p
        fq = fp
        res = find_pq(U=U, I_k_plus_1=I_k_plus_2)
        p = res['p']
        fp = f(p)
    else:
        # keep right
        L = p
        p = q
        fp = fq
        res = find_pq(L=L, I_k_plus_1=I_k_plus_2)
        q = res['q']
        fq = f(p)
    return {'L': L , 'p': p, 'q': q, 'U': U, 'fp': fp, 'fq': fq}


def line_search(f, *, I, eps):
    n, Fn, Fn_1 = find_n(I[0], I[1], eps)
    L = I[0]
    U = I[1]
    I_2 = (U-L) * Fn_1 / Fn

    res = find_pq(U=U,I_k_plus_1=I_2)
    p = res['p']

    res = find_pq(L=L,I_k_plus_1=I_2)
    q = res['q']

    fp = f(p)
    fq = f(q)

    kwargs = {'L' : L,
              'p' : p,
              'q' : q,
              'U' : U,
              'fp': fp,
              'fq': fq,
              }
    for k in range(1, n):
        kwret = reduce_interval(f, **kwargs)
        kwargs.update(kwret)
        if kwargs['p'] >= kwargs['q']:
            return (kwargs['p'] + kwargs['q']) / 2

    delta = (U - L) / (2 * Fn) * 0.7
    p = kwargs['p']
    q = kwargs['q']

    if p != q:
        return (p + q) / 2
    else:
        p_ = p - delta
        q_ = p + delta
        if f(p_) > f(q_):
            return q_
        else:
            return p_


if __name__=="__main__":
    def minimization_func(x):
        return x * x + 1
    x_star = line_search(minimization_func, I=[-0.1, 0.1], eps=1e-7)
    print(x_star, minimization_func(x_star))
