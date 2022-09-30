from fibonacci_section import line_search
from find_interval import find_interval


def func(x):
    return x * x * x * x + 1

def main():
    I = find_interval(func)
    print(I)
    eps = 1e-4
    x = line_search(func, I=I, eps=eps)
    print(x, func(x))

if __name__=="__main__":
    main()
