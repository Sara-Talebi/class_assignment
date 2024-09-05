import numpy as np
import matplotlib

##-----------------------------------------------------------------##
def subfunction1():
    print()
    return 

##-----------------------------------------------------------------##
def subfunction2():
    print()
    return 
    
##-----------------------------------------------------------------##
def tot_force_vec(velocity,mass,g,wind_res_cons):
    f_grav = -mass * g
    
    v_x, v_y    = velocity
    v_magnitude = np.sqrt(v_x**2 + v_y**2)
    
    f_wind_x = -wind_res_cons * v_x * v_magnitude
    f_wind_y = -wind_res_cons * v_y * v_magnitude
    
    f_tot_x = f_wind_x
    f_tot_y = f_grav + f_wind_y
    f_tot   = np.array([f_tot_x,f_tot_y])

    return f_tot

##-----------------------------------------------------------------##
def test_function():
 	mass          = 1000 #kg
 	ini_pos_vec   = [0,100]
 	ini_vel_vec   = [10,10]
 	time	      = 0.0
 	dt            = 0.1
 	wind_res_cons = 0.1
 	g	      = 9.8 #m/s^2
 	
 	position = np.array(ini_pos_vec)
 	velocity = np.array(ini_vel_vec)
 	
 	print(tot_force_vec(velocity,mass,g,wind_res_cons))


##-----------------------------------------------------------------##
test_function()
##-----------------------------------------------------------------##
