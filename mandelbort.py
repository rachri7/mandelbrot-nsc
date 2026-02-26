"""
Mandelbrot Set Generator
Author : [ Your Name ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import time, statistics
import matplotlib.pyplot as plt

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



    # Running function with time
    median_time_naive, all_n_naive = benchmark(
        compute_mandelbrot_naive,
        -2, 1, -1.5, 1.5, 1024,
        n_runs=5, meta_prefix="naive"
    )  

    grid_res = [1024]
    # Timings for the grid res vectorized
    # Median :0.0290s ( min =0.0285, max =0.0297)       256
    # Median :0.1823s ( min =0.1763, max =0.1831)       512
    # Median :0.9823s ( min =0.9776, max =0.9862)       1024
    # Median :3.9466s ( min =3.9258, max =3.9655)       2048
    # Median :15.7511s ( min =15.5853, max =16.0727)    4096
    # Store results and timings in lists
    all_n_all_res_vectorized = []
    timings_vectorized = [] 

    for i in grid_res:
        meta = f"Vectorized_Res_{i}"
        median_time_vectorized, all_n_vectorized = benchmark(
            compute_mandelbrot_vectorized,
            -2, 1, -1.5, 1.5, i,
            n_runs=5,meta_prefix=meta
        )  
        all_n_all_res_vectorized.append(all_n_vectorized)     # result = (all_c)
        timings_vectorized.append(median_time_vectorized)  # just the median time

    # need to fix so it find same res as naive here instaed just indexing 2, this would change depend on what we inset to res array
    all_n_vectorized_1024 = all_n_all_res_vectorized[0]
    median_time_vectorized_1024 = timings_vectorized[0]

    Speed_up_vectorized = median_time_naive / median_time_vectorized_1024

    # --- Create side-by-side subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

    # Naive plot
    im0 = axes[0].imshow(all_n_naive, cmap="hot")
    axes[0].set_title(f"Naive Mandelbrot\nMedian time: {median_time_naive:.2f}s")
    axes[0].axis('off')  # optional: hide axes
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Vectorized plot
    im1 = axes[1].imshow(all_n_vectorized_1024, cmap="hot")
    axes[1].set_title(f"Vectorized Mandelbrot\nMedian time: {median_time_vectorized_1024:.2f}s\n Speed up: {Speed_up_vectorized}")
    axes[1].axis('off')  # optional: hide axes
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("mandelbrot_comparison.png")  # single figure with both
    plt.show()

    
    

    

