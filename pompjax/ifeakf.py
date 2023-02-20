#import jax.numpy as jnp
import pandas as pd
import numpy as np
#import jax

from tqdm import tqdm

from pompjax.stats import sample_uniform2, sample_truncated_normal
from pompjax.inference import check_state_space, eakf, checkbound_params, inflate_ensembles, eakf_update

#from stats import sample_uniform, truncated_normal, sample_uniform2, sample_truncated_normal
#from inference import check_param_space, check_state_space, eakf, checkbound_params, inflate_ensembles

def random_walk_perturbation(param, param_std):
    p, m = param.shape
    return param + np.expand_dims(param_std, -1) * np.random.normal(size=(p, m))

def geometric_cooling(if_iters, cooling_factor=0.9):
    """ Geometric cooling
        Args:
            if_iters       (int): number of iterations of the IF algorithm
            cooling_factor (float): cooling factor (variance shrinking rate)
    """
    alphas = cooling_factor**np.arange(if_iters)
    return alphas**2

def hyperbolic_cooling(if_iters, cooling_factor=0.9):
    alphas = 1/(1+cooling_factor*np.arange(if_iters))
    return alphas

def cooling(num_iteration_if, type_cool="geometric", cooling_factor=0.9):
    if type_cool=="geometric":
        return geometric_cooling(num_iteration_if, cooling_factor=cooling_factor)
    elif type_cool=="hyperbolic":
        return hyperbolic_cooling(num_iteration_if, cooling_factor=cooling_factor)


def ifeakf(process_model,
            observational_model,
            state_space_initial_guess,
            observations_df,
            parameters_range,
            state_space_range,
            model_settings,
            if_settings,
            cooling_sequence = None,
            perturbation     = None,
            leave_progress   = False):

    if any('adjust_state_space' in key for key in if_settings.keys()):
        adjust_state_space = if_settings["adjust_state_space"]
    else:
        adjust_state_space = True

    if cooling_sequence is None:
        cooling_sequence = cooling(if_settings["Nif"], type_cool=if_settings["type_cooling"], cooling_factor=if_settings["shrinkage_factor"])

    k           = model_settings["k"] # Number of observations
    p           = model_settings["p"] # Number of parameters (to be estimated)
    n           = model_settings["n"] # Number of state variable
    m           = model_settings["m"] # Number of stochastic trajectories / particles / ensembles

    sim_dates   = model_settings["dates"]
    assim_dates = if_settings["assimilation_dates"]

    param_range = parameters_range.copy()
    std_param   = param_range[:, 1] - param_range[:,0]
    SIG         = std_param ** 2 / 4; #  Initial covariance of parameters

    if perturbation is None:
        perturbation = std_param / 10

    assimilation_times = len(assim_dates)

    param_post_all = np.full((p, m, assimilation_times, if_settings["Nif"]), np.nan)
    param_mean     = np.full((p, if_settings["Nif"]+1), np.nan)

    for n in tqdm(range(if_settings["Nif"]), leave=leave_progress):
        if n==0:
            p_prior     = sample_uniform2(param_range, m)
            x           = state_space_initial_guess(p_prior)
            param_mean[:, n] = np.mean(p_prior, -1)
        else:
            pmean   = param_mean[:, n]
            pvar    = SIG * cooling_sequence[n]
            p_prior = sample_truncated_normal(pmean, pvar ** (0.5), param_range, m)
            x       = state_space_initial_guess(p_prior)

        t_assim    = 0
        cum_obs    = np.zeros((k, m))
        param_time = np.full((p, m, assimilation_times), np.nan)

        for t, date in enumerate(sim_dates):
            x     = process_model(t, x, p_prior)
            y     = observational_model(t, x, p_prior)
            cum_obs += y

            if pd.to_datetime(date) == pd.to_datetime(assim_dates[t_assim]):

                pert_noise  = perturbation*cooling_sequence[n]
                p_prior     = random_walk_perturbation(p_prior, pert_noise)
                p_prior     = checkbound_params(p_prior, param_range)

                # Measured observations
                z     = observations_df.loc[pd.to_datetime(date)][[f"y{i+1}" for i in range(k)]].values
                oev   = observations_df.loc[pd.to_datetime(date)][[f"oev{i+1}" for i in range(k)]].values

                print(z)

                x_prior = x.copy()
                p_post  = p_prior.copy()

                # Update state space
                #x_post, _ = eakf(x_prior, cum_obs, z, oev)
                #p_post, _ = eakf(p_prior, cum_obs, z, oev)

                if adjust_state_space:
                    x_post, _ = eakf_update(x_prior, cum_obs, z, oev)
                    x_post    = inflate_ensembles(x_post, inflation_value=if_settings["inflation"], m=m)

                p_post, _ = eakf_update(p_prior, cum_obs, z, oev)
                p_post    = inflate_ensembles(p_post, inflation_value=if_settings["inflation"], m=m)

                # check for a-physicalities in the state and parameter space.
                x_post = check_state_space(x_post, state_space_range)
                p_post = checkbound_params(p_post, param_range)

                p_prior = p_post.copy()
                x       = x_post.copy()


                # save posterior parameter
                param_time[:, :, t_assim] = p_post
                cum_obs                   = np.zeros((k, m))
                t_assim                   += 1

        param_post_all[:, :, :, n] = param_time
        param_mean[:, n+1]         = param_time.mean(-1).mean(-1) # average posterior over all assimilation times and them over all ensemble members

    return param_mean, param_post_all

