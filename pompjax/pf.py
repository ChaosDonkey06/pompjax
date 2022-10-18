import jax.numpy as np

def normalize_weights(w):
    """
    Weight normalization.
    w: Particle weights.
    """
    return w / np.sum(w)

def compute_effective_sample_size(w):
    """
    Effective sample size.
    """
    return 1/np.sum(w**2)

def importance_sampling(w, z, y, q):
    """
    Importance sampling:
            Approximate likelihood P(z|θ) using the importance density q(z|y).
            Where y=g(x;θ) is the observation model used after using the state space model f(x;θ).
            Compute the relative weight of each particle respect the previous PF iteration and normalize the weights.
    w: Particle weights.
    z: World observations.
    y: Modelled observations.
    q: Proposal distribution.
    """
    loglik  = q(z , y)

    # Recompute weights and normalize them
    w = w * loglik
    w = normalize_weights(w)

    return w

def naive_weights(m):
    """
    Naive weights.
        Assume all particles have the same weight.
    """
    return np.ones(m)*1/m

def resample_particles(particles, x, w, m, p=None):
    """
    Particle resample.
    """
    if p:
        fixed_particles = np.sort(np.random.choice(np.arange(m), size=int(m*(1-p)), replace=False, p=w))
        particles_index = np.random.choice(np.arange(m), size=m, replace=True, p=w)
        particles_index[fixed_particles] = fixed_particles
    else:
        particles_index = np.sort(np.random.choice(np.arange(m), size=m, replace=True, p=w))

    w         = naive_weights()

    particles = particles[:, particles_index] # Replace particles.
    x_post    = x[:, particles_index]

    return particles, x_post, w


def pf(pprior, x, y, z, q):
    """
    Particle filter.
    """

    # IS(w, z, y, q)
    m               = pprior.shape[1]
    w               = importance_sampling(w, z, y, q)
    ppost, xpost, w = resample_particles(pprior, x, w, m, p=0.1)

    return ppost, xpost, w

