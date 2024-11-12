import numpy as np
import tqdm
import numpy.random
from matplotlib import pyplot as plt
import h5py
import emcee
import corner
from numpy.random import uniform

##-----------------------------------------------------------------------------##
def gaussian(x, amplitude, mu1, sigma1, mu2, sigma2, mu3, sigma3):
    value = amplitude * (np.exp(-(((x - mu1)**2)/(2 * sigma1**2))) + 
                         np.exp(-(((x - mu2)**2)/(2 * sigma2**2))) + 
                         np.exp(-(((x - mu3)**2)/(2 * sigma3**2))))
    return value
##-----------------------------------------------------------------------------##
def loglikelihood(x, y, amplitude, mu1, sigma1, mu2, sigma2, mu3, sigma3):
    gaus = gaussian(x, amplitude, mu1, sigma1, mu2, sigma2, mu3, sigma3)
    value = -0.5 * np.sum((y - gaus)**2)
    return value
##-----------------------------------------------------------------------------##
def logprior(amplitude, mu1, sigma1, mu2, sigma2, mu3, sigma3):
    if -20 < amplitude < 20 and 0 < mu1 < 10 and 0 < sigma1 < 5 and \
       0 < mu2 < 10 and 0 < sigma2 < 5 and 0 < mu3 < 10 and 0 < sigma3 < 5:
        return 0
    return -np.inf
##-----------------------------------------------------------------------------##
def logposterior(x, y, amplitude, mu1, sigma1, mu2, sigma2, mu3, sigma3):
    value = loglikelihood(x, y, amplitude, mu1, sigma1, mu2, sigma2, mu3, sigma3) + logprior(amplitude, mu1, sigma1, mu2, sigma2, mu3, sigma3)
    return value
##-----------------------------------------------------------------------------##
def proposal(x):
    value = np.random.uniform(-1,1) + x
    return value
##-----------------------------------------------------------------------------##
def post(x):
    # Fitted Parameters from the previous code
    amplitude, mu1, sigma1, mu2, sigma2, mu3, sigma3 = -8.270311462159512, 0.883806662220054, 0.0628731774756843, 0.7676583833354201, 0.0844830959792166, 5.562293523404577, 2.609653972793868  
    return gaussian(x, amplitude, mu1, sigma1, mu2, sigma2, mu3, sigma3)
##-----------------------------------------------------------------------------##
def mcmc(initial, post, prop, iterations):
    x = [initial]
    p = [post(x[-1])]
    for i in tqdm.tqdm(range(iterations)):
        x_test = prop(x[-1])
        p_test = post(x_test)

        acc = p_test / p[-1]
        u = np.random.uniform(0, 1)
        if u <= acc:
            x.append(x_test)
            p.append(p_test)
    return x, p
##-----------------------------------------------------------------------------##
def calculate_autocorrelation(chain, max_lag=100):
    n = len(chain)
    mean = np.mean(chain)
    var = np.var(chain)

    autocorr_values = []
    for lag in range(1, max_lag + 1):
        cov = np.sum((chain[:n - lag] - mean) * (chain[lag:] - mean)) / n
        autocorr = cov / var
        autocorr_values.append(autocorr)
    return autocorr_values
##-----------------------------------------------------------------------------##
def estimate_autocorrelation_length(autocorr_values, threshold=0.1):
    for lag, value in enumerate(autocorr_values, start=1):
        if value < threshold:
            return lag
    return len(autocorr_values)
##-----------------------------------------------------------------------------##
def main_A():
    # Reading actual data
    f = h5py.File('./Python/data.hdf', 'r')
    x = f['data/xpos'][:]
    y = f['data/ypos'][:]

    # Run the MCMC simulation
    num_iteration = int(1e4)
    start_x_pos = 0
    autocorr_lengths = []
    chains_list = []

    for k in range(500):
        chain, prob = mcmc(start_x_pos, post, proposal, num_iteration)
        chains_list.append(chain)
        autocorr_values = calculate_autocorrelation(chain)
        autocorr_length = estimate_autocorrelation_length(autocorr_values)
        autocorr_lengths.append(autocorr_length)
        print(f"Chain {k+1}: Autocorrelation Length = {autocorr_length}")
        

    average_auto_corr = np.mean(autocorr_lengths)
    var_auto_corr = np.var(autocorr_lengths)
    std_auto_corr = np.std(autocorr_lengths)

    print("Mean of the 500 autocorrelation lengths: ", average_auto_corr)
    print("Variance of the 500 autocorrelation lengths: ", var_auto_corr)
    print("Standard Deviation of the 500 autocorrelation lengths: ", std_auto_corr)

    plt.figure(figsize=(10, 6))  

    for i, chain in enumerate(chains_list):
        plt.plot(chain, label=f'Chain {i+1}', alpha=0.7)  # Optional: use alpha for transparency

    plt.xlabel('Evolution of the walker for all Chain')
    plt.ylabel('x-Value')
    plt.title('Plot of Chains')
    plt.legend(loc='upper right')  
    plt.show()
    
    # Plot histogram of autocorrelation lengths for all chains
    plt.figure()
    plt.hist(autocorr_lengths, bins=20, color='blue', alpha=0.7)
    plt.title("Histogram of Autocorrelation Lengths")
    plt.xlabel("Autocorrelation Length")
    plt.ylabel("Frequency")
    plt.show()

    # Sample posterior after burn-in and thinning
    plt.figure()
    plt.title("Posterior samples")
    _ = plt.hist(chain[100::100], bins=100)
    plt.show()
##-----------------------------------------------------------------------------##
def main_B():
    # Reading actual data
    f = h5py.File('./Python/data.hdf', 'r')
    x = f['data/xpos'][:]
    y = f['data/ypos'][:]
    
    # Initialize the emcee sampler
    n_walkers = 500
    n_params = 7  
    sampler = emcee.EnsembleSampler(n_walkers, n_params, logposterior)

    # Set the initial positions of the walkers within a plausible range
    initial_state = np.array([
        [uniform(-10, 10), uniform(0.5, 1.5), uniform(0.01, 0.1),
        uniform(0.5, 1.5), uniform(0.01, 0.1), uniform(5.0, 6.0), uniform(2.0, 3.0)]
        for _ in range(n_walkers)
    ])

    # Run MCMC sampling
    sampler.run_mcmc(initial_state, 1000)
    chain = sampler.get_chain()
    logp = sampler.get_log_prob()

    # Find the maximum posterior values
    i = logp[-1, :].argmax()
    maxp = chain[-1, i, :]
    print("Maximum posterior values:", maxp)

    # Plot the data and the best-fit model
    plt.figure()
    xref = np.linspace(min(x), max(x), 100)
    plt.scatter(x, y, label='Data')
    plt.plot(xref, gaussian(xref, maxp[0],  maxp[1], maxp[2], maxp[3], maxp[4], maxp[5], maxp[6]), label='Max Posterior Fit', color='orange')
    plt.legend()

    # Corner plot for parameter distribution
    _ = corner.corner(chain[-1, :, :], labels=["Amplitude", "mu1", "sigma1", "mu2", "sigma2", "mu3", "sigma3"])
    plt.show()

##-----------------------------------------------------------------------------##
def main_C():
    # Reading actual data
    f = h5py.File('./Python/data.hdf', 'r')
    x = f['data/xpos'][:]
    y = f['data/ypos'][:]
    
    # Initialize the emcee sampler
    n_walkers = 500
    n_params = 7  
    sampler = emcee.EnsembleSampler(n_walkers, n_params, logposterior)

    # Set the initial positions of the walkers within a plausible range
    initial_state = np.array([
        [uniform(-10, 10), uniform(0.5, 1.5), uniform(0.01, 0.1),
        uniform(0.5, 1.5), uniform(0.01, 0.1), uniform(5.0, 6.0), uniform(2.0, 3.0)]
        for _ in range(n_walkers)
    ])

    # Run MCMC sampling
    sampler.run_mcmc(initial_state, 1000)
    chain = sampler.get_chain()
    logp = sampler.get_log_prob()

    # Find the maximum posterior values
    i = logp[-1, :].argmax()
    maxp = chain[-1, i, :]
    print("Maximum posterior values:", maxp)

    # Plot the data and the best-fit model
    plt.figure()
    xref = np.linspace(min(x), max(x), 100)
    plt.scatter(x, y, label='Data')
    plt.plot(xref, gaussian(xref, maxp[0],  maxp[1], maxp[2], maxp[3], maxp[4], maxp[5], maxp[6]), label='Max Posterior Fit', color='orange')
    plt.legend()

    # Corner plot for parameter distribution
    _ = corner.corner(chain[-1, :, :], labels=["Amplitude", "mu1", "sigma1", "mu2", "sigma2", "mu3", "sigma3"])
    plt.show()

##-----------------------------------------------------------------------------##
main_A()
