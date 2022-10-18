
import jax.numpy as np
import numpy as onp
import jax

def check_state_space(x, xrange):
    """
    Constrain the state space x to the defined range
    """
    return np.clip(x.T, a_min=xrange.at[:,0].get(), a_max=xrange.at[:,1].get()).T

def check_param_space(key, θ, prange):
    key1, key2       = jax.random.split(key, 2)

    loww             = np.expand_dims(prange.at[:, 0].get(), -1)
    uppp             = np.expand_dims(prange.at[:, 1].get(), -1)

    teta_correct_low = loww*(1+0.2*jax.random.uniform(key1, θ.shape))
    teta_correct_up  = uppp*(1-0.2*jax.random.uniform(key2, θ.shape))

    θ                = np.where(θ < loww, x=teta_correct_low, y=θ)
    θ                = np.where(θ > uppp, x=teta_correct_up,  y=θ)
    return θ


def checkbound_params(params, prange):
    p, _ = prange.shape

    params_update = []
    for idx_p, p in enumerate(range(p)):
        loww = onp.array(prange.at[p, 0].get())
        upp  = onp.array(prange.at[p, 1].get())

        p_ens = onp.array(params.at[idx_p, :].get())

        idx_wrong_loww = onp.where(p_ens < loww)[0]
        idx_wrong_upp  = onp.where(p_ens > upp)[0]

        onp.put(p_ens, idx_wrong_loww, loww * (1+0.2*onp.random.rand( idx_wrong_loww.shape[0])) )
        onp.put(p_ens, idx_wrong_upp,  upp  * (1-0.2*onp.random.rand( idx_wrong_upp.shape[0])) )

        params_update.append(p_ens)

    return np.array(params_update)

def inflate_ensembles(ens, inflation_value=1.2, m=300):
    return onp.mean(ens,1, keepdims=True) * np.ones((1,m)) + inflation_value*(ens-onp.mean(ens,1, keepdims=True) * np.ones((1,m)))

def eakf(x, y, z, oev):
    p, m = x.shape

    μ_prior  = y.mean()
    σ2_prior = y.var()

    if μ_prior == 0: # degenerate prior.
        σ2_post  = 1e-3
        σ2_prior = 1e-3

    σ2_post  = σ2_prior * oev / (σ2_prior + oev)
    μ_post   = σ2_post  * (μ_prior/σ2_prior + z/oev)
    α        = (oev / (oev + σ2_prior)) ** (0.5)
    dy       = (μ_post-y) + α *  (y-μ_prior)

    rr = np.full((p, 1), np.nan)
    for ip in range(p):
        A  = onp.cov(x.at[ip,:].get(), y)
        rr = rr.at[ip,:].set( A[1, 0] / σ2_prior )

    dx       = np.dot(rr, dy)
    #print("dy shape", dy.shape)
    #print("dx shape", dx.shape)
    #print("rr shape", rr.shape)

    #rr       = np.cov(x, y).at[-1,:-1].get() / σ2_prior
    #dx       = np.dot(np.expand_dims(rr, -1), dy )

    xpost = x + dx
    ypost = y + dy

    return xpost, ypost
