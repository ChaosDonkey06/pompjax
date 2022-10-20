from scipy.stats import truncnorm
#import jax.numpy as np
import numpy as np
#import jax

#def sample_uniform(key, vmin, vmax, p, m=300):
#    """
#    Generate m sample from a uniform distribution
#    """
#    rn =  jax.random.uniform(key, shape=(m, p)) * (vmax - vmin) + vmin
#    return rn.T

#def truncated_normal(key, mean, var, lower, upper, p, m, dtype=np.float64):
#    """
#    Generate a truncated normal distribution
#    """
#    samples = jax.random.truncated_normal(key, shape=(m, p), lower= lower, upper=upper, dtype=dtype) * (var)**(1/2) + mean
#    return samples.T

#def truncated_multivariate_normal(key, μ, cov, shape, lower, upper):
#    """
#    Generate a truncated multivariate normal distribution
#    """
#    return np.clip(jax.random.multivariate_normal(key, μ, cov, shape), lower, upper)

#def sample_normal(key, θ_min, θ_max, μ, cov, p, m=300):
#    """
#    Generate a truncated normal distribution
#    """
#    return truncated_multivariate_normal(key, μ, cov, shape=(m, p), lower=θ_min, upper=θ_max).T

def sample_uniform2(xrange, m):
    p       = xrange.shape[0]
    samples = np.full((p, m), np.nan)
    for ip in range(p):
        samples = samples.at[ip, :].set(np.random.uniform(xrange[ip, 0], xrange[ip, 1], m))
    return samples

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm( (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd )

def sample_truncated_normal(mean, sd, xrange, m):
    p       = xrange.shape[0]
    samples = np.full((p, m), np.nan)
    for ip in range(p):
        samples = samples.at[ip, :].set(get_truncated_normal(mean=mean[ip], sd=sd[ip], low=xrange[ip, 0], upp=xrange[ip, 1]).rvs(m))
    return samples