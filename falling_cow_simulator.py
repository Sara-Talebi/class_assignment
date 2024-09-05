import numpy as np
import matplotlib

##-----------------------------------------------------------------##
def plot_energy(time_list, eng_list):
    plt.plot(time_list, eng_list[:, 2], label="Total Energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.title("Energy vs Time")
    plt.legend()
    plt.show()
    
##-----------------------------------------------------------------##
def plot_velocity(time_list, vel_list):
    plt.plot(time_list, vel_list[:, 1], label="y-velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity vs Time")
    plt.legend()
    plt.show()
    
##-----------------------------------------------------------------##
def plot_position(time_list, pos_list):
    plt.plot(time_list, pos_list[:, 1], label="y-position")
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title("Height vs Time")
    plt.legend()
    plt.show()
    
##-----------------------------------------------------------------##
def eng_calculator(position,velocity,mass,g):
    #KE=0.5*m*v^2
    eng_kinetic = 0.5 * mass * np.sum(velocity**2)
    
    #PE=m*g*h
    eng_poten = mass * g * position[1]
    
    #E=KE+PE
    eng_tot = eng_kinetic + eng_poten
    
    return eng_kinetic, eng_poten, eng_tot

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
    
    v_magnitude = np.sqrt(velocity[0]**2 + velocity[1]**2)
    
    f_wind_x = -wind_res_cons * velocity[0] * v_magnitude
    f_wind_y = -wind_res_cons * velocity[1] * v_magnitude
    
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
 	
 	pos_list  = []
 	vel_list  = []
 	time_list = []
 	eng_list  = []
 	
 	while position[1] > 0.0:
 	
 		f_tot = tot_force_vec(velocity,mass,g,wind_res_cons)
 		velocity, position = curr_pos_vel_calc(position,velocity,f_tot,mass,dt)
 		eng_kinetic, eng_poten, eng_tot = eng_calculator(position,velocity,mass,g)
 		
 		pos_list.append(position)
 		vel_list.append(velocity)
 		eng_list.append(eng_tot)
 		time_list.append(time)
 		
 		time += dt
 	##------------
 	plot_position(time_list, pos_list)
 	plot_velocity(time_list, vel_list)
 	plot_energy(time_list, eng_list)

##-----------------------------------------------------------------##
test_function()
##-----------------------------------------------------------------##
