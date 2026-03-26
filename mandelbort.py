"""
Mandelbrot Set Generator
Author : [ Your Name ]
Course : Numerical Scientific Computing 2026
"""
import numpy as np
import time, statistics
import matplotlib.pyplot as plt
from numba import njit, prange
from multiprocessing import Pool
import os
from dask import delayed, compute
from dask.distributed import Client, LocalCluster

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

@njit(cache = True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = 0.0
    z_imag = 0.0

    for i in range(max_iter):
        z_real_sq = z_real * z_real
        z_imag_sq = z_imag * z_imag
        if z_real_sq + z_imag_sq > 4.0:
            return i
        z_imag_new = 2.0 * z_real*z_imag + c_imag
        z_real = z_real_sq - z_imag_sq + c_real
        z_imag = z_imag_new
    return max_iter

@njit(cache = True)
def mandelbrot_chunk(row_start, row_end, N,
                     x_min, x_max, y_min, y_max, max_iter):

    out = np.empty((row_end - row_start, N), dtype=np.int32)

    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N

    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy

        for col in range(N):
            c_real = x_min + col * dx
            out[r, col] = mandelbrot_pixel(c_real, c_imag, max_iter)

    return out

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4,n_runs=1,n_chunks=None,meta_prefix = ""):

    if n_chunks is None:
        n_chunks = n_workers
    chunk_size = max(1, N // n_chunks)
    chunks = []
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append(
            (row, row_end, N, x_min, x_max, y_min, y_max, max_iter)
        )
        row = row_end
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as pool:
        # un-timed warm-up: triggers Numba JIT inside workers
        pool.map(_worker, tiny)
        time,parts=benchmark(pool.map,_worker, chunks,n_runs=n_runs,meta_prefix=meta_prefix)

    return time ,np.vstack(parts)

def mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter=100,n_chunks=32,meta_prefix = ""):

    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)
            (row, row_end, N, x_min, x_max, y_min, y_max, max_iter)
        )
        row = row_end

    parts = compute(*tasks)

    return np.vstack(parts)

def run_algorithms(resolutions, algorithms, n_runs=5):


    results = {}
    timings = {}
    client = None

    for res in resolutions:

        results[res] = {}
        timings[res] = {}

        for name, func in algorithms.items():

            # warmup for numba
            if "dask" in name.lower():
                # --- Distributed case ---
                if "dask_dist" in name.lower():
                    if client is None:
                        print("Create client dist")
                        client = Client("tcp://10.92.1.217:8786")
                        client_type = "dist"
                        client.run(lambda: mandelbrot_chunk(0, 8, 8, -2.5, 1, -1.25, 1.25, 10))
                    elif client is not None and client_type == "local":
                        print("Client closed starts dist")
                        client.close()
                        client = Client("tcp://10.92.1.217:8786")
                        client_type = "dist"
                # --- Local Dask case ---
                else:
                    if client is not None and client_type == "dist":
                        print("Client closed starts local" )
                        client.close()
                        cluster = LocalCluster(n_workers=8, threads_per_worker=1)
                        client = Client(cluster)
                        client_type = "local"
                    elif client is None:
                        print("Create client local")
                        cluster = LocalCluster(n_workers=8, threads_per_worker=1)
                        client = Client(cluster)
                        client_type = "local"
                        client.run(lambda: mandelbrot_chunk(0, 8, 8, -2.5, 1, -1.25, 1.25, 10))
            if "numba" in name.lower():
                _ = func(res)

            if not "parallel" in name.lower():
                median_time, output = benchmark(
                    func,
                    res,
                    n_runs=n_runs,
                    meta_prefix=name
                )
            else:
                median_time, output = func(res) 
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
    
    # The parameters for running the algorihms

    n_runs = 5

    algorithms = {
    # "naive": lambda res: compute_mandelbrot_naive(-2, 1, -1.5, 1.5, res),
    # "vectorized": lambda res: compute_mandelbrot_vectorized(-2, 1, -1.5, 1.5, res),
    # "numba": lambda res: compute_mandelbrot_numba(-2, 1, -1.5, 1.5, res),
    # "numba_parellel": lambda res: compute_mandelbrot_numba_parallel(-2, 1, -1.5, 1.5, res),
    # "hybrid_numba": lambda res: compute_mandelbrot_hybrid(-2, 1, -1.5, 1.5, res),
    # # res = chunk as of now,
    #"numba_lecture4": lambda res: mandelbrot_serial(res, -2, 1, -1.5, 1.5, max_iter=100),
    "parallel_workers_1": lambda res: mandelbrot_parallel(res, -2, 1, -1.5, 1.5,
                                                        n_workers=1,n_runs=n_runs,
                                                        meta_prefix="Numba parellel with workers = 1"),
    "parallel_workers_8": lambda res: mandelbrot_parallel(res, -2, 1, -1.5, 1.5,
                                                        n_workers=8,n_runs=n_runs,
                                                        meta_prefix="Numba parellel with workers = 8"),
    # Should function about the same as numba parelle true this with 8 workers
    }
    n_worker_all = [1, 2, 4, 8]
    
    grid_res = [1024]
    max_cores = max_workers = max(n_worker_all)

    chunks = [max_cores * x for x in [1,2,4,8,16]]
    chunk_dask = [1, 2, 4, 8, 16, 32, 64]


    for c in chunk_dask:
        name = f"dask_chunk_{c}_worker_8"
        algorithms[name] = lambda res, c=c: mandelbrot_dask(res, -2, 1, -1.5, 1.5,n_chunks=c,
                                                        meta_prefix="dask Numba {c} x chunks with workers = 8")
    for c in chunk_dask:
        name = f"dask_dist_chunk_{c}_worker_8"
        algorithms[name] = lambda res, c=c: mandelbrot_dask(res, -2, 1, -1.5, 1.5,n_chunks=c,
                                                        meta_prefix="dask Numba {c} x chunks with workers = 8")
    # for c in chunks:
    #     name = f"parallel_chunk_{c}x_worker_{max_cores}"
    #     algorithms[name] = lambda res, c=c: mandelbrot_parallel(res, -2, 1, -1.5, 1.5, 
    #                                                             n_workers=max_cores,n_runs=n_runs,n_chunks=c,
    #                                                             meta_prefix=f"Numba parellel {c} x chunks with workers = {max_cores}")

    # We see that at that we can drop LIF but also make it higher at overhead start to dominate

    # running the algorithms with timings
    results, timings = run_algorithms(grid_res,algorithms,n_runs=n_runs)

    # Here we take naive and compute speedup from it
    first_key = list(results[1024].keys())[0] 
    naive_time = timings[1024][first_key]
    T1 = timings[1024]["parallel_workers_1"]

    lif = {}
    speedups = {}
    efficiency = {}
    for name, t in timings[1024].items():
        speedups[name] = naive_time / t
        # compute efficiency only for parallel runs
        if "parallel" or "dask" in name:
            # extract worker count from the name, e.g., 'parallel_lecture4_4' -> 4
            n_workers = int(name.split("_")[-1])
            efficiency[name] = speedups[name] / n_workers
            lif[name] = n_workers * t / T1 - 1

    # Choose Optimal Chunk size for max cores
    chunk_lif = {k: v for k, v in lif.items() if "parallel_chunk" in k}

    # Find the one with the lowest LIF
    best_chunk_name = min(chunk_lif, key=chunk_lif.get)
    best_lif = chunk_lif[best_chunk_name]
    print(f"Best chunked algorithm: {best_chunk_name} with LIF = {best_lif:.4f}")

    # Update dictionaries: remove other chunk algorithms
    for d in [speedups, efficiency, lif, timings[1024]]:
        keys_to_remove = [k for k in d if "parallel_chunk" in k and k != best_chunk_name]
        for k in keys_to_remove:
            d.pop(k)

    for d in [speedups, efficiency, lif, timings[1024]]:
        d["chunk_opt"] = d.pop(best_chunk_name)
    

    # create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # mandelbrot image
    im0 = axes[0].imshow(results[1024][first_key], cmap="hot")

    axes[0].set_title(f"{first_key} Mandelbrot\nMedian time: {naive_time:.4f}s")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # speedup table 
    axes[1].axis("off")
    table_data = []
    for name in speedups.keys():
        s = speedups[name]
        e = efficiency.get(name, "")  # empty if efficiency doesn't exist
        l = lif.get(name, "")
        t = timings[1024][name]
        table_data.append([
            name,
            f"{t:.4f}",
            f"{s:.2f}",
            f"{e:.2f}" if e != "" else "",
            f"{l:.2f}" if l != "" else ""
        ])
        table = axes[1].table(
            cellText=table_data,
            colLabels=["Algorithm","Timings [s]", "Speedup", "Efficiency", "LIF"],
            loc="center"
        )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    axes[1].set_title("Algorithm Speedups")

    # im0 = axes[2].imshow(results[1024]["numba_lecture4"], cmap="hot")
    # numba_lecture_time_1024= timings[1024]["numba_lecture4"]
    # axes[2].set_title(f"Numba lecture 4 Mandelbrot\n Median time: {numba_lecture_time_1024:.2f} s")
    # axes[2].axis("off")
    # fig.colorbar(im0, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("mandelbrot_comparison.png")
    plt.show()

    times = []
    n_chunks = []

    for name, time in timings[1024].items():
        if "dask" in name.lower():
            n_chunk = int(name.split("chunk_")[1].split("_")[0])
            n_chunks.append(n_chunk)
            times.append(time)

    plt.plot(n_chunks, times)
    plt.xscale("log")
    plt.xticks(rotation=45)
    plt.show()


    
    

    

