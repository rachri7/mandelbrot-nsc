"""
Mandelbrot Set Generator
Author : [ Your Name ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np

def compute_mandelbrot():
    num = 10  # basically resoulation in each direction
    x_min = -2
    x_max = 1
    y_min = -1.5
    y_max = 1.5
    all_c = []
    all_n = []

    x = np.linspace(x_min, x_max, num)   # real axis
    y = np.linspace(y_min, y_max, num)   # imaginary axis

    # 2D arrays to store results
    all_c = np.zeros((num, num), dtype=complex)  # create matrix of all c, num X num (empty)
    all_n = np.zeros((num, num), dtype=int)     # Create matrix of all n, num X num (empty)

    for i in range(num):        # Iterate over all c 
        for j in range(num):
            c = x[i] + 1j * y[j]
            n = mandelbrot_point(c)     # comput iteration from c/constant

            all_c[i, j] = c             # save all c and iterations
            all_n[i, j] = n

    return all_c, all_n

def mandelbrot_point(c):
    z = 0
    max_iter = 100
    for n in range(max_iter):
        z = z**2 + c
        if abs(z) > 2:
            return n 
    return max_iter  



if __name__=="__main__":
    # c = constant, can be entered as real ... + ... j
    #c=0
    #n=mandelbrot_point(c)  # iteration n
    all_c, all_n = compute_mandelbrot()
    print(f"{all_c=},")
    print(f"{all_n=},")
