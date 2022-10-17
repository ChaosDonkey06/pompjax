
import jax.numpy as np
import numpy as onp
import jax

def check_state_space(x, xrange):
    return np.clip(x.T, a_min=xrange.at[:,0].get(), a_max=xrange.at[:,1].get()).T

def check_param_space(key, θ, prange):
    key1, key2 = jax.random.split(key, 2)

    loww = np.expand_dims(prange.at[:, 0].get(), -1)
    uppp = np.expand_dims(prange.at[:, 1].get(), -1)

    teta_correct_low = loww*(1+0.1*jax.random.uniform(key1, θ.shape))
    teta_correct_up  = uppp*(1-0.1*jax.random.uniform(key2, θ.shape))

    θ = np.where(θ < loww, x=teta_correct_low, y=θ)
    θ = np.where(θ > uppp, x=teta_correct_up,  y=θ)

    return θ

def eakf(x, y, z, oev):
    μ_prior  = y.mean()
    σ2_prior = y.var()

    σ2_post  = σ2_prior * oev / (σ2_prior + oev)

    if μ_prior == 0:
        σ2_post  = 1e-3
        σ2_prior = 1e-3

    μ_post   = σ2_post  * (μ_prior/σ2_prior + z/oev)

    dy = (μ_post - y) +  (oev / (oev + σ2_prior)) ** (1/2)  * (y-μ_prior)
    A  = np.cov(x, y).at[-1,:-1].get() / σ2_prior
    dx = np.squeeze(np.dot(np.expand_dims(A, -1), np.expand_dims(dy, 0)))


    xpost = x + dx
    ypost = y + dy

    return xpost, ypost
