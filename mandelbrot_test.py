import pytest
import numpy as np
import time
import statistics
import dask
from dask import delayed
from dask.distributed import Client

from mandelbort import (compute_mandelbrot_naive,
                        mandelbrot_point_naive,
                        compute_mandelbrot_vectorized,
                        compute_mandelbrot_numba,
                        mandelbrot_point_numba,
                        mandelbrot_point_numba_parallel,
                        compute_mandelbrot_numba_parallel,
                        mandelbrot_dask,
                        mandelbrot_pixel,
                        mandelbrot_chunk,
                        mandelbrot_serial,
                        mandelbrot_parallel
                        )


Known_cases = [
        (0+0j, 100, 100), # origin: never escapes
        (5.0+0j, 100, 1), # far outside, escapes on iteration 1
        (-2.5+0j, 100, 1), # left tip of set
]

#-------------------------------------------
# Numba test functions
#-------------------------------------------
IMPLEMENTATIONS = [
    ("naive", mandelbrot_point_naive),
    ("numba", mandelbrot_point_numba),
    ("numba_parallel", mandelbrot_point_numba_parallel),
]

@pytest.mark.parametrize("name, impl", IMPLEMENTATIONS)
@pytest.mark.parametrize("c, max_iter, expected", Known_cases)
def test_mandelbrot_point_matches_expected(name, impl, c, max_iter, expected):
    '''
    Known Cases Test:
    Ensure that Functions result in the same as known cases (expected)
    '''
    assert impl(c,max_iter) == expected

COMPUTE_IMPLEMENTATIONS = [
    compute_mandelbrot_naive,
    compute_mandelbrot_numba,
    compute_mandelbrot_numba_parallel,
]

#Testing that the array output result in same
@pytest.mark.parametrize("impl", COMPUTE_IMPLEMENTATIONS)
def test_compute_mandelbrot_matches_naive(impl):
    """
    Integration test: 
    Numba Grid need to match Naive Grid
    """
    # Parameters
    # x_min,x_max,y_min,y_max,num
    params = (-2.0, 1.0, -1.5, 1.5, 32)

    expected = compute_mandelbrot_naive(*params)
    actual = impl(*params)

    np.testing.assert_array_equal(actual, expected)


#-------------------------------------------
# Multiprocessing test fucntions
#-------------------------------------------

@pytest.mark.parametrize(
    "c_real, c_imag, max_iter, expected",
    [
        (0.0, 0.0, 100, 100),   # origin: never escapes
        (5.0, 0.0, 100, 1),     # far outside, escapes on iteration 1
        (-2.5, 0.0, 100, 1),    # left tip of set
    ],
)

# 
def test_mandelbrot_pixel_matches_expected(c_real, c_imag, max_iter, expected):
    result = mandelbrot_pixel(c_real, c_imag, max_iter)
    '''
    Known Cases Test:
    Ensure that Functions result in the same as known cases (expected)
    '''
    assert result == expected

def test_mandelbrot_parallel_matches_serial_32_grid():
    """
    Integration test: 
    Parallel Grid need to match Serial Grid
    """
    # Parameters
    # N, x_min, x_max, y_min, y_max, max_iter=100
    params = (32, -2.0, 1.0, -1.5, 1.5, 100)

    expected = mandelbrot_serial(*params)
    _, actual = mandelbrot_parallel(
        N=32, x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5,
        max_iter=100,
        n_workers=8,
        n_chunks=16,
    )

    np.testing.assert_array_equal(actual, expected)

def test_mandelbrot_pixel_is_pure():
    """
    Unit Test (Determinism):
    Ensures that mandelbrot_pixel is a pure function by verifying
    that repeated calls with identical inputs produce identical outputs.
    """
    result1 = mandelbrot_pixel(0.0, 0.0, 100)
    result2 = mandelbrot_pixel(0.0, 0.0, 100)

    assert result1 == result2

    np.testing.assert_array_equal(result1, result2)

# -------------------------------------------
# Dask test functions
# -------------------------------------------
def test_mandelbrot_dask_matches_serial_32_grid():
    """
    Integration test:
    Dask grid need to match Serial Grid.
    """
    # Parameters
    # N, x_min, x_max, y_min, y_max, max_iter=100
    params = (32, -2.0, 1.0, -1.5, 1.5, 100)

    expected = mandelbrot_serial(*params)

    actual = mandelbrot_dask(
        N=32,
        x_min=-2.0,
        x_max=1.0,
        y_min=-1.5,
        y_max=1.5,
        max_iter=100,
        n_chunks=4,
    )

    np.testing.assert_array_equal(actual, expected)

def test_dask_delayed_single_chunk_returns_expected():
    """
    Direct scheduler integration:
    delayed task computes expected chunk.
    """
    # Parameters
    # row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter
    expected = mandelbrot_chunk(
        0, 4, 8,
        -2.0, 1.0, -1.5, 1.5, 100
    )

    future = delayed(mandelbrot_chunk)(
        0, 4, 8,
        -2.0, 1.0, -1.5, 1.5, 100
    )

    actual = dask.compute(future)[0]

    np.testing.assert_array_equal(actual, expected)

def test_dask_client_submit_matches_expected():
    """
    Dask Integration Test:
    Ensures that submitting a computation via Dask Client
    and gathering the result produces the same output as
    the serial reference implementation.
    """
    params = (32, -2.0, 1.0, -1.5, 1.5, 100)
    expected = mandelbrot_serial(*params)

    with Client(processes=False) as client:
        future = client.submit(mandelbrot_dask, *params, n_chunks=4)
        actual = client.gather(future)

    np.testing.assert_array_equal(actual, expected)


# -------------------------------------------
# speedup test functions
# -------------------------------------------

COMPUTE_IMPLEMENTATIONS = [
    ("vectorized", compute_mandelbrot_vectorized),
    ("numba", compute_mandelbrot_numba),
    ("numba_parallel", compute_mandelbrot_numba_parallel),
]

@pytest.mark.parametrize("name, impl", COMPUTE_IMPLEMENTATIONS)
def test_faster_than_naive(name, impl):
    '''
    Speed Test:
    Ensuring implementation all are faster then naive version
    '''
    # Parameters
    # x_min, x_max, y_min, y_max, N
    params = (-2.0, 1.0, -1.5, 1.5, 256)

    # warmup for numba, still warmup the other's however does not effect results
    impl(*params)

    naive_times = []
    impl_times = []

    for _ in range(3):
        t0 = time.perf_counter()
        compute_mandelbrot_naive(*params)
        naive_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        impl(*params)
        impl_times.append(time.perf_counter() - t0)

    naive_time = statistics.median(naive_times)
    impl_time = statistics.median(impl_times)

    assert impl_time < naive_time


def test_numba_faster_than_naive():
    '''
    Speed Test:
    Ensuring Numba is faster then naive version
    '''
    # Parameters
    # x_min, x_max, y_min, y_max, N
    params = (-2.0, 1.0, -1.5, 1.5, 256)

    # warmup numba
    compute_mandelbrot_numba(*params)

    naive_times = []
    numba_times = []

    for _ in range(3):
        t0 = time.perf_counter()
        compute_mandelbrot_naive(*params)
        naive_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        compute_mandelbrot_numba(*params)
        numba_times.append(time.perf_counter() - t0)

    naive_time = statistics.median(naive_times)
    numba_time = statistics.median(numba_times)

    assert numba_time < 0.1 * naive_time