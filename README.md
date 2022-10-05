# pompjax
Partially Observed Markov process with JAX backbone.

## Notes to self
environment insstalled via virtualenv
    source pompjax/bin/activate
## what I want
The idea is to use the eakf/pf to provide points estimates, and also use the MCAP to construct the likelihood profiles (very similar to pomp but simpler, just for myself.)

## Dynamical system theory
In general any dynamical system can be writted as shown below where $x$ is the state space of the model and $\theta \in \mathbb{R}^{p}$ the vector of parameters. For purposes of the packages we assume $f$ is stochastic and therefore we consider $m$ trajectories for both the state space $x\in  \mathbb{R}^{m\times n}$ (n: number of variables) and  $\theta\in  \mathbb{R}^{m\times p}$. m is the number of **stochastic** trajectories considered. We also have an observational model $g$ to map to the observations (measures of the real-world), $y_{sim}\in \mathbb{R}^{m\times k}$ (k: number of observations).

In general we deal with partially observed systems which means that $k<n$.

$$\frac{dx}{dt}=f(x, \theta )$$
$$y_{sim}=g(x, \theta )$$

## Inference algorihtm theory and ideas.
The goal of the packages/algorithm is to given a set of real-world observations $y=[y_1, y_2, ..., y_T]$ of length $T$, we want to find the optimal set of parameters $\hat{\theta}$, such that using a simulator with those parameters we have $\hat{y}_{sim} = g(x_{1:T};\hat{\theta})$. And ideally CRPS$(\hat{y}_{sim}, y)$ $\rightarrow 0$, where CRPS is the continuos ranked probability score, or any error (not CRPS necesarilly) to be 0.

In programming we want a function $\hat{\theta}=\mathcal{F}(f, g, \Omega, m, n, k, p, T)$. Which is a function that receive the process model $f$, the observational model $g$ the size of the things indicated previously (number of stochastic trajectories, number of variables/size of the state space), size of the parameters space, number of observations. We also constrain the parameters to be drawn from a convex set $\Omega$ (a.k.a prior range). Another important thing is the initial guess of the state space $x_0$, that for general purposes will also be a function x_0=$f_0(\mathcal{X})$, where $\mathcal{X}$ is the range of the state variables. (I will not consider by now the case where the initial conditions of the system want to be estimated).

The function $\mathcal{F}$ is the Iterated Filtering (cite1, cite2) that also have hyperparameters such as the number of iteration $N_{IF}$, the cooling sequence $\sigma$ (or something like that, don't remember the name). We also need something that map from the prior of the parameter space to the posterior - I'll use in the examples the eakf.

## The Ensemble Adjustment Kalman Filter (EAKF)

The EAKF proceeds as following: given a current observation $y_t$ at time $t$ we assume is normally distributed with prescribed variance – observational error variance $c_t\sim \mathcal{N}(y_t,oev)$, then we can compute the simulated colonization across the ensemble members at the given time denoted $\mathbf{y}_t$, $\mathbf{y}_t=[y_t^1, y_t^2,y_t^3,…,y_t^{300}  ]$ normally distributed with computed mean and variance (the priors) $y_t^i\sim \mathcal{N}(\mu_{prior}, \sigma_{prior}^2)$. Then by convolution of two normal distribution the posterior distribution can be parametrized with posterior mean and variance given by the equation shown below.

$$\sigma^2_{post}= \sigma^2_{prior}\frac{oev}{\sigma^2_{prior}+oev}$$
$$\mu_{post}= \sigma^2_{post}\left(\frac{\mu_{prior}}{\sigma^2_{prior}} + \frac{c_t}{oev}\right)$$

Then the Kalman gain $dy$ of each ensemble member $\mathbf{y}_t$ is given by

$$\mathbf{dy}_t=(\mu_{post}-\mathbf{y}_t)+\sqrt{\frac{oev}{oev+\sigma^2_{prior}}}(\mathbf{y}_t-\mu_{prior})$$

The EAKF uses the covariance between the parameters and the observations to compute the Kalman gain of the parameters $d\theta$ of each ensemble member for each set of parameters$\theta=[\theta_1,\theta_2,…,\theta_{300}]$, note that here $\theta_i$ is the tuple of ensemble members $\theta_i=[\gamma, \beta]$ as described in previous section.

$$\mathbf{d\theta}=\frac{cov(\mathbf{\theta}, \mathbf{y}_t)}{\sigma^2_{prior}}\times \mathbf{dy}$$
The posterior of the number of colonization $\mathbf{y}_{post}$ and of the parameters $\theta_{post}$ is then given by


$$\mathbf{y_t}^{post}= \mathbf{y_t}+\mathbf{dy}$$
$$\bm{\theta}_{post}=\bm{\theta} + \bm{d\theta}$$