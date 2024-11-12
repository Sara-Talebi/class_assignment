import numpy as np
import h5py
import matplotlib.pyplot as plt
import emcee
import corner

# Define Gaussian function
def gaussian(x, amplitude, mu1, sigma1, mu2, sigma2, mu3, sigma3):
    return amplitude * (np.exp(-(((x - mu1) ** 2) / (2 * sigma1 ** 2))) +
                       np.exp(-(((x - mu2) ** 2) / (2 * sigma2 ** 2))) +
                       np.exp(-(((x - mu3) ** 2) / (2 * sigma3 ** 2))))

# Define the log likelihood function
def log_likelihood(params, xpos, ypos):
    amplitude, mu1, sigma1, mu2, sigma2, mu3, sigma3 = params
    model = gaussian(xpos, amplitude, mu1, sigma1, mu2, sigma2, mu3, sigma3)
    sigma_obs = 0.1  # Assumed observational error
    return -0.5 * np.sum(((ypos - model) / sigma_obs) ** 2)

# Define the log prior function
def log_prior(params):
    amplitude, mu1, sigma1, mu2, sigma2, mu3, sigma3 = params
    if -10 < amplitude < 10 and 0 < mu1 < 10 and 0 < sigma1 < 10 and \
       0 < mu2 < 10 and 0 < sigma2 < 10 and 0 < mu3 < 10 and 0 < sigma3 < 10:
        return 0.0  # Uniform prior within specified bounds
    return -np.inf  # Log(0) for values outside of bounds

# Define the log probability function (log posterior)
def log_probability(params, xpos, ypos):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, xpos, ypos)

# Main function to run MCMC with autocorrelation
def run_mcmc():
    # Load data
    f = h5py.File('./Python/data.hdf', 'r')
    xpos = f['data/xpos'][:]
    ypos = f['data/ypos'][:]
    
    # Initial guess for parameters
    initial_params = [-8.3, 0.88, 0.058, 0.767, 0.081, 5.5, 2.6]
    
    # Set up MCMC sampler
    n_walkers = 500  # Number of walkers
    n_iterations = 10000  # Number of iterations
    ndim = len(initial_params)
    initial_pos = initial_params + 1e-4 * np.random.randn(n_walkers, ndim)
    
    # Initialize the sampler
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability, args=(xpos, ypos))
    sampler.run_mcmc(initial_pos, n_iterations, progress=True)
    
    # Plot the chains for each parameter
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(f"Param {i+1}")
    axes[-1].set_xlabel("Step number")
    plt.show()

    # Autocorrelation length
    tau = sampler.get_autocorr_time()
    print("Autocorrelation time:", tau)
    convergence = np.all(tau * 50 < sampler.iteration)
    print("Convergence reached:", convergence)
    
    # Flattened chain and corner plot
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    fig = corner.corner(flat_samples, labels=["amplitude", "mu1", "sigma1", "mu2", "sigma2", "mu3", "sigma3"],
                        truths=initial_params)
    plt.show()

    # Plot the final state of the fit
    best_fit_params = np.mean(flat_samples, axis=0)
    print("Best-fit parameters (mean of posterior):", best_fit_params)

    y_fit = [gaussian(x, *best_fit_params) for x in xpos]
    plt.plot(xpos, ypos, "o", label="Data")
    plt.plot(xpos, y_fit, "-", label="MCMC Fit", color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

run_mcmc()
