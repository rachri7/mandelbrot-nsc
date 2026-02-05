"""
Mandelbrot Set Generator
Author : [ Your Name ]
Course : Numerical Scientific Computing 2026
"""
def mandelbrot_point(c):
    z = 0
    max_iter = 100
    for n in range(max_iter):
        z = z**2 + c
        if abs(z) > 2:
            return n 
    return max_iter  
#def compute_mandelbrot(x):
#pass
if __name__=="__main__":
    # c = constant, can be entered as real ... + ... j
    c=0
    n=mandelbrot_point(c)
    print("iterations= ", n, " and point", c)