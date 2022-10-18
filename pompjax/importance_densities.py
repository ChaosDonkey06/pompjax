def poisson_loglikelihood(real_world_observations, model_observations, num_times=100):
    nll =  -np.sum(model_observations,1) + np.sum(real_world_observations*np.log(model_observations+1),1) # Poisson LL
    return - nll

def normal_loglikelihood(real_world_observations, model_observations, error_variance=None, A=0.1, num_times=100):
    if not error_variance:
        error_variance = 1+(0.2*real_world_observations)**2

    nll =  A * np.exp(-0.5 * (real_world_observations-model_observations)**2/error_variance) # Normal LL
    return - nll
