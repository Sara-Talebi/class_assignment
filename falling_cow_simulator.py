import numpy as np
import matplotlib

##-----------------------------------------------------------------##
def subfunction1():
    print()
    return 

##-----------------------------------------------------------------##
def curr_pos_vel_calc(position,velocity,f_tot,mass,dt):
    #f=ma
    accel = f_tot / mass
    
    #v=v0+adt
    curr_vel = velocity + (accel * dt)
    
    #r=r0+vdt
    curr_pos = position + (velocity * dt)
    
    return curr_vel, curr_pos
    
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
 	
 	f_tot = tot_force_vec(velocity,mass,g,wind_res_cons)
 	curr_vel, curr_pos = curr_pos_vel_calc(position,velocity,f_tot,mass,dt)
 	print(curr_vel,curr_pos)
 	


##-----------------------------------------------------------------##
test_function()
##-----------------------------------------------------------------##
