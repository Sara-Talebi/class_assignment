import h5py
import numpy as np
from matplotlib import pyplot as plt

# Define the 2D Gaussian model
def gaussian_2d(x, y, A, x0, y0, sigma_x, sigma_y):
    return A * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))

# Define the loss function (sum of squared differences)
def loss_function(params, xpos, ypos):
    A, x0, y0, sigma_x, sigma_y = params
    predicted = gaussian_2d(xpos, ypos, A, x0, y0, sigma_x, sigma_y)
    actual = np.ones_like(xpos)  # assuming the actual data points have a value of 1
    return np.sum((predicted - actual) ** 2)

# Define the gradient descent function
def gradient_descent(xpos, ypos, initial_params, learning_rate=0.001, max_iters=10000):
    params = np.array(initial_params)
    for i in range(max_iters):
        # Calculate gradients numerically
        gradients = np.zeros_like(params)
        h = 1e-5  # a small step for numerical differentiation
        for j in range(len(params)):
            params_step = np.copy(params)
            params_step[j] += h
            loss1 = loss_function(params, xpos, ypos)
            loss2 = loss_function(params_step, xpos, ypos)
            gradients[j] = (loss2 - loss1) / h
        
        # Update the parameters using gradient descent
        params -= learning_rate * gradients
        
        # Optionally print loss every 1000 iterations
        if i % 1000 == 0:
            current_loss = loss_function(params, xpos, ypos)
            print(f"Iteration {i}, Loss: {current_loss}")
    
    return params

# Main function to run everything
def main():
    # Open the HDF5 file
    f = h5py.File('./data.hdf', 'r')

    # Load data
    xpos = f['data/xpos'][:]
    ypos = f['data/ypos'][:]

    # Initial guess for the parameters (A, x0, y0, sigma_x, sigma_y)
    initial_guess = [1, np.mean(xpos), np.mean(ypos), 1, 1]

    # Run gradient descent to fit the parameters
    fitted_params = gradient_descent(xpos, ypos, initial_guess, learning_rate=0.001, max_iters=10000)

    # Extract the fitted parameters
    A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit = fitted_params

    # Print the fitted parameters
    print(f"Fitted parameters: A = {A_fit}, x0 = {x0_fit}, y0 = {y0_fit}, sigma_x = {sigma_x_fit}, sigma_y = {sigma_y_fit}")

    # Create a grid of points for plotting the fitted Gaussian
    x = np.linspace(min(xpos), max(xpos), 100)
    y = np.linspace(min(ypos), max(ypos), 100)
    x, y = np.meshgrid(x, y)

    # Evaluate the fitted Gaussian on the grid
    z = gaussian_2d(x, y, A_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit)

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

# Call the main function
if __name__ == "__main__":
    main()
