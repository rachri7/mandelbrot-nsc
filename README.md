# Mandelbrot Set Project (CPU + GPU)

**Author:** [Rasmus Mellergaard Christensen]
**Course:** Numerical Scientific Computing 2026

---

## Overview

This project implements multiple approaches to compute the Mandelbrot set, focusing on performance and parallelisation.

Implemented methods include:

* Naive (pure Python)
* Vectorized (NumPy)
* Numba (JIT compiled)
* Numba Parallel
* Multiprocessing (chunk-based)
* Dask (task-based parallelism)
* GPU (OpenCL, float32 and float64)

The goal is to compare correctness and performance across implementations.

---

## Project Structure

```
mandelbort.py          # CPU implementations (naive, numba, parallel, dask)
mandelbrot_gpu.py      # GPU implementations using OpenCL
test_mandelbrot.py     # Pytest test suite
mandelbrot_comparison.png  # Benchmark output (generated)
```

---

## Requirements

Install dependencies using:

```bash
pip install numpy matplotlib numba pytest pyopencl dask distributed
```

---

## Running the Code

### CPU Implementations

Run:

```bash
python mandelbort.py
```

This will:

* Run selected implementations with the use of run_algorithms
* Use Timings for Benchmark performance of each method
* Generate a Mandelbrot comparassion including a escape count, and also summary table

In the summary table there is included:

* Median runtime over multiple runs
* Speedup vs naive implementation
* Efficiency for parallel methods
* Load imbalance factor (LIF)

Results are visualised in:

```
mandelbrot_comparison.png
```

---

### GPU Implementations

Run:

```bash
python mandelbrot_gpu.py
```

This executes:

* Float32 GPU version
* Float64 GPU version
* Specifically for both resulations 1024 and 2048

Optional plotting can be enabled inside the function:

```python
mandelbrot_opencl_f32(plot_enable=True)
```

---

## Running Tests

To verify correctness across implementations:

```bash
pytest
```

The test suite includes:

* Known-value unit tests
* CPU implementation consistency checks
* Parallel vs serial correctness
* Dask correctness tests
* Performance tests (ensuring speedup over naive)

---


