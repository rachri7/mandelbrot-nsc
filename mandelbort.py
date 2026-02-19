"""
Mandelbrot Set Generator
Author : [ Your Name ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import time, statistics
import matplotlib.pyplot as plt

# probally need to remove all_c return as we just want to use iteration however just for debug for now
def compute_mandelbrot(x_min,x_max,y_min,y_max,num): 
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

def benchmark(func , *args , n_runs =3) :
    """ Time func , return median of n_runs . """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    median_t = statistics.median( times )
    print (f" Median :{median_t:.4f}s "
    f"( min ={ min( times ):.4f}, max ={ max( times ):.4f})")
    return median_t , result 



if __name__=="__main__":
    # Running function with time
    median_time, result = benchmark(
    compute_mandelbrot,
    -2, 1, -1.5, 1.5, 1024,
    n_runs=5)  

    all_c,all_n = result

    # Debugging Shape, Type of the array
    print (f" Shape : {all_c.shape }")
    print (f" Type : {all_c.dtype }") 

    plt.imshow(all_n, cmap="hot")
    plt.title("Mandelbort Plot")
    plt.colorbar()
    plt.savefig("mandelbort_Plot_Naive.png")  # saves as PNG file
    plt.show()

    

