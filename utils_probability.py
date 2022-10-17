import jax.numpy as np
import jax

def sample_uniform(key, vmin, vmax, p, m=300):
    """
    Generate m sample from a uniform distribution
    """
    rn =  jax.random.uniform(key, shape=(m, p)) * (vmax - vmin) + vmin
    return rn.T

def truncated_normal(key, mean, sd, lower, upper, p, m, dtype=np.float64):
    """
    Generate a truncated normal distribution
    """
    samples = jax.random.truncated_normal(key, shape=(m, p), lower= lower, upper=upper, dtype=dtype) * (sd)**(1/2) + mean
    return samples.T

def truncated_multivariate_normal(key, μ, cov, shape, lower, upper):
    """
    Generate a truncated multivariate normal distribution
    """
    return np.clip(jax.random.multivariate_normal(key, μ, cov, shape), lower, upper)

def sample_normal(key, θ_min, θ_max, μ, cov, p, m=300):
    """
    Generate a truncated normal distribution
    """
    return truncated_multivariate_normal(key, μ, cov, shape=(m, p), lower=θ_min, upper=θ_max).T