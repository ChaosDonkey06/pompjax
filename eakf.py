import jax.numpy as np

def eakf(x, y, z, oev, return_gain=False):
    """ Ensemble Adjustment Kalman Filter

    Args:
        x    : State space at time t. Shape (n: number of state variables, m: number of ensembles)
        y: Simulated observation.
        z    : Measured observation.
        oev  : Observation error variance.
        return_gain (bool, optional): If true return the gain of the filter. Defaults to False.

    Returns:
        xpost:   Posterior estimate of state space.
        obspost: Posterior estimate of observation.
    """
    # num_vars, num_ens = x.shape

    prior_mean_ct = y.mean(-1, keepdims=True)  # Average over ensemble members
    prior_var_ct  = y.var(-1, keepdims=True)   # Compute variance over ensemble members

    post_var_ct   = prior_var_ct * oev / (prior_var_ct + oev)
    post_mean_ct  = post_var_ct * (prior_mean_ct/prior_var_ct + z / oev)
    alpha         = oev / (oev+prior_var_ct); alpha = alpha**0.5
    dy            = post_mean_ct + alpha*( y - prior_mean_ct ) - y

    A       = np.cov(x, y)
    covars  = A.at[:-1, -1].get()
    rr      = covars / prior_var_ct
    dx      = np.dot( np.expand_dims(rr, -1), np.expand_dims(dy, 0))
    xpost   = x + dx
    obspost = y + dy

    if return_gain:
        return xpost, obspost, dx
    else:
        return xpost, obspost
