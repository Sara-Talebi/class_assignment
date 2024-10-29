import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# Open the HDF5 file
f = h5py.File('./data.hdf', 'r')

# Load data
xpos = f['data/xpos'][:]
ypos = f['data/ypos'][:]

# Define the model function (simplified 2D Gaussian)
def gaussian_2d(coords, A, x0, y0):
    x, y = coords
    return A * np.exp(-((x - x0)**2 + (y - y0)**2) / 2)

# Prepare the data for curve fitting
coords = np.vstack((xpos, ypos))

# Initial guess for the parameters (amplitude, x0, y0)
initial_guess = [1, np.mean(xpos), np.mean(ypos)]

# Perform the curve fit
params, covariance = curve_fit(gaussian_2d, coords, np.ones_like(xpos), p0=initial_guess)

# Extract the fitted parameters
A_fit, x0_fit, y0_fit = params

# Print the fitted parameters
print(f"Fitted parameters: A = {A_fit}, x0 = {x0_fit}, y0 = {y0_fit}")

# Plot the original scatter plot
plt.scatter(xpos, ypos, label='Data')

# Plot the fitted Gaussian center
plt.scatter(x0_fit, y0_fit, color='red', label='Fitted Center', marker='x')

plt.legend()
plt.show()
