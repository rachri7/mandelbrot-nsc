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

    x = np.linspace(x_min, x_max, num)   # real axis
    y = np.linspace(y_min,y_max, num)   # imaginary axis

    for i in range(num):
        for j in range(num):
            c = x[i] + 1j * y[j]
            print(c)
    return c

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
    c= compute_mandelbrot()
    print(f"{c=},")
    # print(f"{n=},")
