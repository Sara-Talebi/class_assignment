import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# Open the HDF5 file
f = h5py.File('./data.hdf', 'r')

# Load data
xpos = f['data/xpos'][:]
ypos = f['data/ypos'][:]

# Define the model function (2D Gaussian)
def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y):
    x, y = coords
    return A * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))

# Prepare the data for curve fitting
coords = np.vstack((xpos, ypos))

# Initial guess for the parameters (amplitude, x0, y0, sigma_x, sigma_y)
initial_guess = [1, np.mean(xpos), np.mean(ypos), 1, 1]

# Perform the curve fit
params, covariance = curve_fit(gaussian_2d, coords, np.ones_like(xpos), p0=initial_guess)

# Extract the fitted parameters
A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit = params

# Print the fitted parameters
print(f"Fitted parameters: A = {A_fit}, x0 = {x0_fit}, y0 = {y0_fit}, sigma_x = {sigma_x_fit}, sigma_y = {sigma_y_fit}")

# Create a grid of points for plotting the fitted Gaussian
x = np.linspace(min(xpos), max(xpos), 100)
y = np.linspace(min(ypos), max(ypos), 100)
x, y = np.meshgrid(x, y)

# Evaluate the fitted Gaussian on the grid
z = gaussian_2d((x, y), A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit)

# Plot the original scatter data
plt.scatter(xpos, ypos, label='Data', color='blue')

# Plot the fitted Gaussian function as a contour plot
plt.contour(x, y, z, levels=10, cmap='viridis')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.colorbar(label='Gaussian Intensity')
plt.show()
