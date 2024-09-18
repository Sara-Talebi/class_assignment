import numpy as np
import matplotlib.pyplot as plt


##-----------------------------------------------------------------##
def symplectic(m, k, x, v, dt):
    new_v = v + (dt * (-(k/m) * x))
    new_x = x + (dt * new_v)
    return new_x, new_v
##-----------------------------------------------------------------##
def explicit(m, k, x, v, dt):
    new_x = x + (dt * v)
    new_v = v + (dt * (-(k/m) * x))

    return new_x, new_v
##-----------------------------------------------------------------##
def test_function():
    m  = 10 
    k  = 0.9
    x0 = 1.0
    v0 = 0.0
    dt = 0.1
    t0 = 0 
    
    x = []
    v = []
    t = []
    
    x.append(x0)
    v.append(v0)
    t.append(t0)
    t_curr = t0
    
    while t_curr <= 100:
        x_last = x[-1]
        v_last = v[-1]
        #x_new, v_new = explicit(m, k, x_last, v_last, dt)
        x_new, v_new = symplectic(m, k, x_last, v_last, dt)
        x.append(x_new)
        v.append(v_new)
        t_curr += dt
        t.append(t_curr)
        
       
    #print(x,'\n', t)    
    plt.figure()
    plt.plot(t, x)
    plt.xlabel('time')
    plt.ylabel('position')
    plt.title('x vs t')
    plt.show()
    


##-----------------------------------------------------------------##
test_function()
##-----------------------------------------------------------------##
