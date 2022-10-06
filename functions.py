import autograd.numpy as np


def sphere(d):
    r"""Minima at 0"""
    def function(x):
        # assert isinstance(x, np.ndarray)
        assert x.shape == (d, 1)
        return np.sum(np.square(x))
    return function


def sum_of_squares(d):
    r"""Minima at 0"""
    def function(x):
        # assert isinstance(x, np.ndarray)
        assert x.shape == (d, 1)
        return np.sum(np.arange(1, d + 1).reshape(-1, 1) * np.square(x))
    return function


def sum_of_diff_pow(d):
    r"""Minima at 0"""
    def function(x):
        # assert isinstance(x, np.ndarray)
        assert x.shape == (d, 1)
        s = 0
        for j, i in enumerate(x,start=2):
            s += np.power(np.abs(i), j)
        return s
    return function


def booth_function(d):
    r"""Minima at (1, 3)"""
    assert d == 2, "Booth function is defined for 2D inputs only."
    def function(x):
        assert x.shape == (d, 1)
        return (x[0]+2*x[1]-7)**2+(2*x[0]+x[1]-5)**2
    return function


def matyas(d):
    r"""Minima at (0, 0)"""
    assert d == 2, "Matyas function is defined for 2D inputs only."
    def function(x):
        assert x.shape == (d, 1)
        return 0.26*(x[0]**2+x[1]**2)-0.48*x[0]*x[1]
    return function


def six_humped_camel(d):
    r"""Minima at (0.0898, -0.7126) and (-0.0898, 0.7126)"""
    assert d == 2, "Six Humped Camel function is defined for 2D inputs only."
    def function(x):
        assert x.shape == (d, 1)
        t1 = (4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2
        t2 = x[0] * x[1]
        t3 = (-4 + 4 * x[1] ** 2) * x[1] ** 2
        return t1 + t2 + t3
    return function


def bukin_n6(d):
    r"""Global minima at (-10, 1)"""
    assert d == 2, "Bukin function is defined for 2D inputs only."
    def function(x):
        assert x.shape == (d, 1)
        t0 = np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2))
        t1 = 100 * t0
        t2 = 0.01 * np.abs(x[0] + 10)
        return t1 + t2
    return function


def drop_wave(d):
    r"""Global minima at (0, 0)"""
    assert d == 2, "Drop wave undefined for non 2D inputs."
    def function(x):
        assert x.shape == (d, 1)
        num = 1 + np.cos(12 * np.sqrt(np.sum(np.square(x))))
        denom = 0.5 * np.sum(np.square(x)) + 2
        return -num/denom
    return function


def beale(d):
    r"""Global minima at (3, 0.5)"""
    assert d == 2, "Beale function defined for 2D case."
    def function(x):
        assert x.shape == (d, 1)
        t1 = 1.5 - x[0] + x[0] * x[1]
        t2 = 2.25 - x[0] + x[0] * x[1] * x[1]
        t3 = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1]
        return t1 ** 2 + t2 ** 2 + t3 ** 2
    return function
