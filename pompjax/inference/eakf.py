
#import jax.numpy as jnp
import numpy as np
#import jax

def check_state_space(x, xrange):
    """
    Constrain the state space x to the defined range
    """
    return np.clip(x, a_min=xrange[0], a_max=xrange[1])

# def check_param_space(key, θ, prange):
#     key1, key2       = jax.random.split(key, 2)
# 
#     loww             = np.expand_dims(prange.at[:, 0].get(), -1)
#     uppp             = np.expand_dims(prange.at[:, 1].get(), -1)
# 
#     teta_correct_low = loww*(1+0.2*jax.random.uniform(key1, θ.shape))
#     teta_correct_up  = uppp*(1-0.2*jax.random.uniform(key2, θ.shape))
# 
#     θ                = np.where(θ < loww, x=teta_correct_low, y=θ)
#     θ                = np.where(θ > uppp, x=teta_correct_up,  y=θ)
#     return θ
# 

def checkbound_params(params, prange):
    p, _ = prange.shape

    params_update = []
    for ip in range(p):
        loww = np.array(prange[ip, 0])
        upp  = np.array(prange[ip, 1])

        p_ens = np.array(params[ip, :])

        idx_wrong_loww = np.where(p_ens < loww)[0]
        idx_wrong_upp  = np.where(p_ens > upp)[0]

        idx_wrong        = np.where(np.logical_or(p_ens <loww, p_ens > upp))[0]
        idx_good         = np.where(np.logical_or(p_ens >=loww, p_ens <= upp))[0]
        p_ens[idx_wrong] = np.median(p_ens[idx_good])

        np.put(p_ens, idx_wrong_loww, loww * (1+0.2*np.random.rand( idx_wrong_loww.shape[0])) )
        np.put(p_ens, idx_wrong_upp,  upp  * (1-0.2*np.random.rand( idx_wrong_upp.shape[0])) )

        params_update.append(p_ens)

    return np.array(params_update)

def inflate_ensembles(ens, inflation_value=1.2, m=300):
    return np.mean(ens,1, keepdims=True) * np.ones((1,m)) + inflation_value*(ens-np.mean(ens,1, keepdims=True) * np.ones((1,m)))

def eakf(x, y, z, oev):
    p, m = x.shape

    mu_prior  = y.mean()
    var_prior = y.var()

    if mu_prior == 0: # degenerate prior.
        var_post  = 1e-3
        var_prior = 1e-3

    var_post  = var_prior * oev / (var_prior + oev)
    mu_post   = var_post  * (mu_prior/var_prior + z/oev)
    alpha    = (oev / (oev + var_prior)) ** (0.5)
    dy       = (mu_post-y) + alpha * (y-mu_prior)

    rr = np.full((p, 1), np.nan)
    for ip in range(p):
        A  = np.cov(x[ip, :], y)
        rr[ip,:] =  A[1, 0] / var_prior
    dx       = np.dot(rr, dy)

    xpost = x + dx
    ypost = y + dy

    return xpost, ypost


def eakf_update(x, y, z, oev):
    p, m   = x.shape
    k, m   = y.shape
    k      = z.shape

    xpost  = x.copy()
    ypost  = y.copy()

    for ki in range(k):
        xpost, ypost = eakf(xpost, y[ki, :], z[ki], oev[ki])

    return xpost, ypost