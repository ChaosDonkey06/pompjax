import jax.numpy as np
import numpy as onp
import jax

from tqdm import tqdm

from utils_probability import sample_uniform, truncated_normal
from eakf import check_param_space, check_state_space, eakf, checkbound_params, inflate_ensembles


def random_walk_perturbation(key, x, σ, p, m):
    a = x.T + σ * jax.random.uniform(key, shape=(m, p))
    return a.T

def random_walk_perturbation(param, param_std, p, m):
    return param + onp.array([param_std]).T * onp.random.normal(size=(p, m))

def geometric_cooling(if_iters, cooling_factor=0.9):
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

    assimilation_times = len(observations_df)

    θpost = np.full((p, m, assimilation_times, if_settings["Nif"]), np.nan)
    θmean = np.full((p,    if_settings["Nif"]+1), np.nan)

    keys_if = jax.random.split(key, if_settings["Nif"])

    for n in tqdm(range(if_settings["Nif"])):

        if n==0:
            θ     = sample_uniform(keys_if[n], param_range[:,0], param_range[:,1], p, m)
            x     = state_space_initial_guess(θ)
            θmean = θmean.at[:, n].set(np.mean(θ, -1))

        else:
            pmean     = θmean.at[:,n].get()
            pvar      = SIG * cooling_sequence[n]
            θ         = truncated_normal(keys_if[n], pmean, pvar,  param_range.at[:,0].get(), param_range.at[:,1].get(), p, m)
            x         = state_space_initial_guess(θ)

        t_assim = 0
        ycum    = np.zeros((k, m))

        for t, date in enumerate(sim_dates):
            x   = process_model(t, x, θ)
            x   = inflate_ensembles(x, inflation_value=if_settings["inflation"], m=m)
            x   = check_state_space(x, state_space_range)

            y    = observational_model(t, x, θ)
            ycum += y

            σp = perturbation*cooling_sequence.at[n].get()
            θ  = random_walk_perturbation(θ, σp, p, m)
            θ  = checkbound_params(θ, param_range)

            θ  = inflate_ensembles(θ, inflation_value=if_settings["inflation"], m=m)
            θ  = check_state_space(θ, param_range)

            if date == assim_dates[t_assim]:
                date_infer =  assim_dates[t_assim]

                # Measured observations
                z     = observations_df.loc[date_infer][[f"y{i+1}" for i in range(k)]].values
                oev   = observations_df.loc[date_infer][[f"oev{i+1}" for i in range(k)]].values

                # Update state space
                x, _ = eakf(x, ycum, z, oev)
                θ, _ = eakf(θ, ycum, z, oev)

                # check for a-physicalities in the state and parameter space.
                x    = check_state_space(x, state_space_range)
                θ    = checkbound_params(θ, param_range)

                # Update parameter space
                #θ   = checkbound_params(keys_if[n], θ, param_range)

                # save posterior parameter
                θpost    = θpost.at[:, :, t_assim, n].set(θ)
                ycum     = np.zeros((k, m))
                t_assim  += 1

        θtime = θpost.at[:, :, :, n].get()
        θmean = θmean.at[:,n+1].set(θtime.mean(-1).mean(-1)) # average posterior over all assimilation times and them over all ensemble members

    return θmean, θpost

