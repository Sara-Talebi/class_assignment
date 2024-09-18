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
    t0 = 0 
    
    x = []
    v = []
    t = []
    
    x.append(x0)
    v.append(v0)
    t.append(t0)
    
    for i in range(0, 10):
        x[i+1], v[i+1] = explicit(m, k, x[i], v[i], dt)
        x.append(x[i+1])
        v.append(v[i+1])
        new_t = t[i] + dt
        t.append(new_t)
       
        
    plt.figure()
    plt.plot(t, x)
    plt.show()
    


##-----------------------------------------------------------------##
test_function()
##-----------------------------------------------------------------##
