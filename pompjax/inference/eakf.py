
#import jax.numpy as jnp
import numpy as np
#import jax

def check_state_space(x, xrange):
    """
    Constrain the state space x to the defined range
    """
    return np.clip(x, a_min=xrange[0], a_max=xrange[1])

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

    mu_prior  = np.mean(y, -1, keepdims=True)
    var_prior = np.var(y, -1, keepdims=True)

    idx_degenerate = np.where(mu_prior==0)[0]
    var_prior[idx_degenerate] =  1e-3

    var_post  = var_prior * oev / (var_prior + oev)
    mu_post   = var_post  * (mu_prior/var_prior + z/oev)
    alpha     = (oev / (oev + var_prior)) ** (0.5)
    dy        = (mu_post-y) + alpha * (y-mu_prior)

    rr = np.full((p, 1), np.nan)
    for ip in range(p):
        A  = np.cov(x[ip, :], y)
        rr[ip,:] =  A[1, 0] / var_prior
    dx       = np.dot(rr, np.expand_dims(dy, 0))

    xpost = x + dx
    ypost = y + dy

    return xpost, ypost

x1   = p_prior
y1   = cum_obs
z1   = z
oev1 = oev


p, m   = x1.shape
k, m   = y1.shape
k      = z1.shape[0]

xpost  = x1.copy()
ypost  = y1.copy()

def eakf_update(x, y, z, oev):
    p, m   = x.shape
    k, m   = y.shape
    k      = z.shape[0]

    xpost  = x.copy()
    ypost  = y.copy()

    for ki in range(k):
        xpost, ypost[ki, :] = eakf(xpost, ypost[ki, :], z[ki], oev[ki])

    return xpost, ypost