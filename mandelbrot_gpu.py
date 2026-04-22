from numba import njit
import pyopencl as cl
import numpy as np
import time, matplotlib.pyplot as plt
def mandelbrot_opencl_f32(N: int = 1024, plot_enable: bool = False):
    KERNEL_SRC = """
    __kernel void mandelbrot(
        __global int *result,
        const float x_min, const float x_max,
        const float y_min, const float y_max,
        const int N, const int max_iter)
    {
        int col = get_global_id(0);
        int row = get_global_id(1);
        if (col >= N || row >= N) return;   // guard against over-launch

        float c_real = x_min + col * (x_max - x_min) / (float)N;
        float c_imag = y_min + row * (y_max - y_min) / (float)N;

        float zr = 0.0f, zi = 0.0f;
        int count = 0;
        while (count < max_iter && zr*zr + zi*zi <= 4.0f) {
            float tmp = zr*zr - zi*zi + c_real;
            zi = 2.0f * zr * zi + c_imag;
            zr = tmp;
            count++;
        }
        result[row * N + col] = count;
    }
    """

    ctx   = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    prog  = cl.Program(ctx, KERNEL_SRC).build()

    MAX_ITER = 200
    X_MIN, X_MAX = -2.5, 1.0
    Y_MIN, Y_MAX = -1.25, 1.25

    image = np.zeros((N, N), dtype=np.int32)
    image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

    # To setup kernel and then arguments for the kernel explicitly create the kernel once 
    # instead of re-fetching the kernel object from the compiled program
    kernel = cl.Kernel(prog, "mandelbrot")
    kernel.set_args(
        image_dev,
        np.float32(X_MIN), np.float32(X_MAX),
        np.float32(Y_MIN), np.float32(Y_MAX),
        np.int32(N), np.int32(MAX_ITER)
    )

    # reusing one compiled kernel, your code is repeatedly creating new kernel instances behind the scenes.

    # --- Warm up (first launch triggers a kernel compile) ---
    # prog.mandelbrot(queue, (64, 64), None, image_dev,
    #                 np.float32(X_MIN), np.float32(X_MAX),
    #                 np.float32(Y_MIN), np.float32(Y_MAX),
    #                 np.int32(64), np.int32(MAX_ITER))

    cl.enqueue_nd_range_kernel(queue, kernel, (N, N), None)
    queue.finish()

    # --- Time the real run ---
    t0 = time.perf_counter()
    cl.enqueue_nd_range_kernel(queue, kernel, (N, N), None)
    # prog.mandelbrot(queue, (N, N), None, image_dev,
    #                 np.float32(X_MIN), np.float32(X_MAX),
    #                 np.float32(Y_MIN), np.float32(Y_MAX),
    #                 np.int32(N), np.int32(MAX_ITER))
    queue.finish()
    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, image, image_dev)
    queue.finish()

    print(f"GPU {N}x{N}: {elapsed*1e3:.1f} ms")
    if plot_enable is True:
        plt.imshow(image, cmap='hot', origin='lower'); plt.axis('off')
        plt.savefig("mandelbrot_gpu_f32.png", dpi=150, bbox_inches='tight')
    return elapsed

def mandelbrot_opencl_f64(N: int = 1024, plot_enable: bool = False):
    KERNEL_SRC = """
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    __kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;

    double zr = 0.0;
    double zi = 0.0;

    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0) {
        double tmp = zr*zr - zi*zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;
        count++;
    }

    result[row * N + col] = count;
}
    """

    ctx   = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    prog  = cl.Program(ctx, KERNEL_SRC).build()

    MAX_ITER = 200
    X_MIN, X_MAX = -2.5, 1.0
    Y_MIN, Y_MAX = -1.25, 1.25

    image = np.zeros((N, N), dtype=np.int32)
    image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

    # To setup kernel and then arguments for the kernel explicitly create the kernel once 
    # instead of re-fetching the kernel object from the compiled program
    kernel = cl.Kernel(prog, "mandelbrot_f64")
    kernel.set_args(
        image_dev,
        np.float64(X_MIN), np.float64(X_MAX),
        np.float64(Y_MIN), np.float64(Y_MAX),
        np.int32(N), np.int32(MAX_ITER)
    )

    # reusing one compiled kernel, your code is repeatedly creating new kernel instances behind the scenes.

    # --- Warm up (first launch triggers a kernel compile) ---
    # prog.mandelbrot(queue, (64, 64), None, image_dev,
    #                 np.float32(X_MIN), np.float32(X_MAX),
    #                 np.float32(Y_MIN), np.float32(Y_MAX),
    #                 np.int32(64), np.int32(MAX_ITER))

    cl.enqueue_nd_range_kernel(queue, kernel, (N, N), None)
    queue.finish()

    # --- Time the real run ---
    t0 = time.perf_counter()
    cl.enqueue_nd_range_kernel(queue, kernel, (N, N), None)
    # prog.mandelbrot(queue, (N, N), None, image_dev,
    #                 np.float32(X_MIN), np.float32(X_MAX),
    #                 np.float32(Y_MIN), np.float32(Y_MAX),
    #                 np.int32(N), np.int32(MAX_ITER))
    queue.finish()
    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, image, image_dev)
    queue.finish()

    print(f"GPU {N}x{N}: {elapsed*1e3:.1f} ms")
    if plot_enable is True:
        plt.imshow(image, cmap='hot', origin='lower'); plt.axis('off')
        plt.savefig("mandelbrot_gpu_f64.png", dpi=150, bbox_inches='tight')
    return elapsed

if __name__=="__main__":
    mandelbrot_opencl_f32()
    mandelbrot_opencl_f64()
    mandelbrot_opencl_f32(N=2048)
    mandelbrot_opencl_f64(N=2048)
