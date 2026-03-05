"""
Mandelbrot Set Generator
Author : [ Your Name ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import time, statistics
import matplotlib.pyplot as plt
from numba import njit, prange

# probally need to remove all_c return as we just want to use iteration however just for debug for now
def compute_mandelbrot_vectorized(x_min,x_max,y_min,y_max,num): 
    all_c = []
    all_n = []

    x = np.linspace(x_min, x_max, num)   # real axis
    y = np.linspace(y_min, y_max, num)   # imaginary axis

    X,Y = np.meshgrid(x,y)

    # grid of complex numbers
    all_c = X +1j*Y
    all_n = mandelbrot_point_vectorized(all_c)

    return all_n

def mandelbrot_point_vectorized(all_c):
    max_iter = 100

    # Z values of mandel_plot
    Z = np.zeros(all_c.shape, dtype=complex) 
    all_n = np.zeros(all_c.shape, dtype=int)    # iteration count for each point

    for _ in range(max_iter):
        mask = np.abs(Z) <= 2               # Check all elements in Z, is bool
        Z[mask] = Z[mask]**2 + all_c[mask]  # Mandelbrot set formula:
        all_n[mask] += 1 # Counter iteration
    return all_n

def compute_mandelbrot_naive(x_min,x_max,y_min,y_max,num): 
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
            n = mandelbrot_point_naive(c)     # comput iteration from c/constant

            all_c[i, j] = c             # save all c and iterations
            all_n[i, j] = n
    return all_n

def mandelbrot_point_naive(c):
    z = 0
    max_iter = 100
    for n in range(max_iter):
        z = z**2 + c
        if abs(z) > 2:
            return n 
    return max_iter 

def benchmark(func, *args, n_runs=3, meta_prefix=""):
    """ Time func , return median of n_runs . """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    median_t = statistics.median( times )
    print (f" {meta_prefix} with Median:{median_t:.4f}s "
    f"( min ={ min( times ):.4f}, max ={ max( times ):.4f})")
    return median_t , result 


def compute_row_sums(A, N):
    for i in range(N):  
        s = np.sum(A[i,:])

def compute_column_sums(A,N):
    for j in range(N):
        s = np.sum(A[:,j])


@njit
def mandelbrot_point_numba(c):
    z = 0j
    max_iter = 100
    for n in range(max_iter):
        if z.real*z.real +  z.imag*z.imag > 4.0:
            return n 
        z = z**2 + c
    return max_iter 

@njit
def compute_mandelbrot_numba(x_min, x_max, y_min, y_max, num,d_type=np.float64):
    x = np.linspace(x_min, x_max, num).astype(d_type)   # real axis
    y = np.linspace(y_min, y_max, num).astype(d_type)  # imaginary axis

    # 2D arrays to store results
    all_c = np.zeros((num, num), dtype=np.complex128)
    all_n = np.zeros((num, num), dtype=np.int64)

    for i in range(num):
        for j in range(num):
            c = x[i] + 1j * y[j]
            n = mandelbrot_point_numba(c)
            all_c[i, j] = c
            all_n[i, j] = n

    return all_n

def compute_mandelbrot_hybrid(x_min, x_max, y_min, y_max, num):
    x = np.linspace(x_min, x_max, num)   # real axis
    y = np.linspace(y_min, y_max, num)   # imaginary axis

    # 2D arrays to store results
    all_c = np.zeros((num, num), dtype=np.complex128)
    all_n = np.zeros((num, num), dtype=np.int64)

    for i in range(num):
        for j in range(num):
            c = x[i] + 1j * y[j]
            n = mandelbrot_point_numba(c)
            all_c[i, j] = c
            all_n[i, j] = n

    return all_n

@njit(parallel=True)
def mandelbrot_point_numba_parallel(c):
    z = 0j
    max_iter = 100
    for n in prange(max_iter):
        if z.real*z.real +  z.imag*z.imag > 4.0:
            return n 
        z = z**2 + c
    return max_iter 

@njit( parallel=True)
def compute_mandelbrot_numba_parallel(x_min, x_max, y_min, y_max, num):
    x = np.linspace(x_min, x_max, num)   # real axis
    y = np.linspace(y_min, y_max, num)  # imaginary axis

    # 2D arrays to store results
    all_c = np.zeros((num, num), dtype=np.complex128)
    all_n = np.zeros((num, num), dtype=np.int64)

    for i in prange(num):
        for j in prange(num):
            c = x[i] + 1j * y[j]
            n = mandelbrot_point_numba_parallel(c)
            all_c[i, j] = c
            all_n[i, j] = n

    return all_n

def run_algorithms(resolutions, algorithms, n_runs=1):

    results = {}
    timings = {}

    for res in resolutions:

        results[res] = {}
        timings[res] = {}

        for name, func in algorithms.items():

            meta = f"{name}_Res_{res}"

            # warmup for numba functions
            if "numba" in name:
                _ = func(-2, 1, -1.5, 1.5, res)

            median_time, output = benchmark(
                func,
                -2, 1, -1.5, 1.5, res,
                n_runs=n_runs,
                meta_prefix=meta
            )

            results[res][name] = output
            timings[res][name] = median_time

    return results, timings

if __name__=="__main__":
    # For testing the difference between row sum (c) or column sum (f)
    '''
    N = 10000
    A = np.random.rand(N,N)
    median_t, _ = benchmark(compute_row_sums, A, N, n_runs=5)
    median_t, _= benchmark(compute_column_sums, A, N, n_runs=5)
    # Results
    # Median :0.1323s ( min =0.1320, max =0.1342)
    # Median :0.4716s ( min =0.4713, max =0.4730)

    A_f= np.asfortranarray(A)
    print(f"Now using Fortran instead")
    median_t, _ = benchmark(compute_row_sums, A_f, N, n_runs=5)
    median_t, _= benchmark(compute_column_sums, A_f, N, n_runs=5)

    # Results
    # Median :0.4947s ( min =0.4905, max =0.5321)
    # Median :0.1308s ( min =0.1307, max =0.1342)
    '''

    algorithms = {
    "naive": compute_mandelbrot_naive,
    "vectorized": compute_mandelbrot_vectorized,
    "numba": compute_mandelbrot_numba,
    "hybrid_numba": compute_mandelbrot_hybrid,
    "numba_parallel": compute_mandelbrot_numba_parallel
    }
    n_runs = 1
    grid_res = [1024]

    results, timings = run_algorithms(grid_res,algorithms,n_runs=n_runs)

    # Timings for the grid res vectorized
    # Median :0.0290s ( min =0.0285, max =0.0297)       256
    # Median :0.1823s ( min =0.1763, max =0.1831)       512
    # Median :0.9823s ( min =0.9776, max =0.9862)       1024
    # Median :3.9466s ( min =3.9258, max =3.9655)       2048
    # Median :15.7511s ( min =15.5853, max =16.0727)    4096
    # Store results and timings in lists

    naive_time = timings[1024]["naive"]

    speedups = {}

    for name, t in timings[1024].items():
        speedups[name] = naive_time / t

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # --- Mandelbrot image ---
    im0 = axes[0].imshow(results[1024]["naive"], cmap="hot")
    axes[0].set_title(f"Naive Mandelbrot\nMedian time: {naive_time:.2f}s")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)


    # --- Speedup table ---
    axes[1].axis("off")

    table_data = []
    for name, s in speedups.items():
        table_data.append([name, f"{s:.2f}x"])

    table = axes[1].table(
        cellText=table_data,
        colLabels=["Algorithm", "Speedup vs Naive"],
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    axes[1].set_title("Algorithm Speedups")

    plt.tight_layout()
    plt.savefig("mandelbrot_comparison.png")
    plt.show()

    
    

    

