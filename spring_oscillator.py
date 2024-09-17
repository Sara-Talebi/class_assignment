import numpy as np
import matplotlib.pyplot as plt

##-----------------------------------------------------------------##
def explicit(m, k, x, v, dt):
    new_x = x + (dt * v)
    new_v = v + (dt * (-(k/m) * x))

    return new_x, new_v
##-----------------------------------------------------------------##
def test_function():
    m  = 10 
    k  = 0.1 
    x0 = 1 
    v0 = 0
    dt = 0.1
    
    x = [0 for i in range(10)]
    v = [0 for j in range(10)]
    
    for i in range(1, 10):
        x[0] = x0
        v[0] = v0
        x[i], v[i] = explicit(m, k, x[i-1], v[i-1], dt)
        print(x,v)
    


##-----------------------------------------------------------------##
test_function()
##-----------------------------------------------------------------##
