
import jax.numpy as np
import numpy as onp

def check_state_space(x, xrange):
    return np.clip(x, xrange.at[0,0].get(), xrange.at[1,0].get())

def check_param_space(θ, prange, p):
    θh = np.full_like(θ, np.nan)
    for ip in range(p):
        loww = prange[0, ip]
        upp  = prange[1, ip]

        θi = θ[ip, :].copy()


        idx_wrong_loww = np.where(θi < loww)[0]
        idx_wrong_upp  = np.where(θi > upp)[0]

        θi.at[idx_wrong_loww].set(loww * (1+0.1*onp.random.rand( idx_wrong_loww.shape[0])))
        θi.at[idx_wrong_upp].set( upp  * (1-0.1*onp.random.rand( idx_wrong_upp.shape[0])))
        θh.at[p, :].set(θi)

    return θh

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
