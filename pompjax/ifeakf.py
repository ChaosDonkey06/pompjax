import jax.numpy as np
import numpy as onp
import jax

from tqdm import tqdm

from utils_probability import sample_uniform, truncated_normal, sample_uniform2, sample_truncated_normal
from eakf import check_param_space, check_state_space, eakf, checkbound_params, inflate_ensembles


def random_walk_perturbation(key, x, σ, p, m):
    rw = x.T + σ * jax.random.uniform(key, shape=(m, p))
    return rw.T

def random_walk_perturbation(param, param_std, p, m):
    return param + onp.array([param_std]).T * onp.random.normal(size=(p, m))

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
            key              = jax.random.PRNGKey(0)):

    if cooling_sequence is None:
        cooling_sequence   = cooling(if_settings["Nif"], type_cool=if_settings["type_cooling"], cooling_factor=if_settings["shrinkage_factor"])

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

    θ_post_all = np.full((p, m, assimilation_times, if_settings["Nif"]), np.nan)
    θ_mean     = np.full((p, if_settings["Nif"]+1), np.nan)

    keys_if = jax.random.split(key, if_settings["Nif"])

    for n in tqdm(range(if_settings["Nif"])):
        if n==0:
            #θprior = sample_uniform(keys_if[n], param_range[:,0], param_range[:,1], p, m)
            θ_prior = sample_uniform2(param_range, m)
            x       = state_space_initial_guess(θ_prior)
            θ_mean  = θ_mean.at[:,n].set(np.mean(θ_prior, -1))
        else:
            pmean   = θ_mean.at[:,n].get()
            pvar    = SIG * cooling_sequence[n]
            θ_prior = sample_truncated_normal(pmean, pvar ** (0.5), param_range, m)
            x       = state_space_initial_guess(θ_prior)

        t_assim = 0
        ycum    = np.zeros((k, m))
        θ_time  = np.full((p, m, assimilation_times), np.nan)
        for t, date in enumerate(sim_dates):
            x   = process_model(t, x, θ_prior)
            y   = observational_model(t, x, θ_prior)
            ycum += y

            if date == assim_dates[t_assim]:
                σp      = perturbation*cooling_sequence.at[n].get()
                θ_prior  = random_walk_perturbation(θ_prior, σp, p, m)
                θ_prior  = checkbound_params(θ_prior, param_range)

                # Measured observations
                z     = observations_df.loc[date][[f"y{i+1}" for i in range(k)]].values
                oev   = observations_df.loc[date][[f"oev{i+1}" for i in range(k)]].values

                x_prior = x.copy()

                # Update state space
                x_post, _ = eakf(x_prior, ycum, z, oev)
                θ_post, _ = eakf(θ_prior,  ycum, z, oev)

                x_post = inflate_ensembles(x_post, inflation_value=if_settings["inflation"], m=m)
                θ_post  = inflate_ensembles(θ_post, inflation_value=if_settings["inflation"], m=m)

                # check for a-physicalities in the state and parameter space.
                x_post = check_state_space(x_post, state_space_range)
                θ_post = checkbound_params(θ_post, param_range)

                θ_prior = θ_post.copy()
                x       = x_post.copy()
                # Update parameter space
                #θ   = checkbound_params(keys_if[n], θ, param_range)
                # save posterior parameter
                θ_time   = θ_time.at[:, :, t_assim].set(θ_post)
                ycum     = np.zeros((k, m))
                t_assim += 1

        θ_post_all = θ_post_all.at[:, :, :, n].set(θ_time)
        θ_mean     = θ_mean.at[:, n+1].set(θ_time.mean(-1).mean(-1)) # average posterior over all assimilation times and them over all ensemble members

    return θ_mean, θ_post_all

