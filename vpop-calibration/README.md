**PySAEM: Python Implementation of the Stochastic Approximation Expectation Maximization Algorithm (SAEM) for Non Linear Mixed Effects Model (NLME)** 


**NLML model:**
A nonlinear mixed-effects (NLME) model is a model with fixed effects (population parameters) and random effects (individual variations).   
Fixed effects are composed of: 
- model parameters with no IIV (InterIndividual Variance) i.e. MI (Model Intrinsic)
- means of model parameters with IIV i.e. mus, means of PDUs (Patiend Descriptors Unknown)
- covariance of individual effects i.e. omega
- residual error variance
We denote $\theta\_i$ the individual parameters, composed of both MIs and PDUs with individual effects. 
In PySAEM, we assume that for each individual i and PDU j, theta_i_j = mu_pop_j * exp(eta_i_j) * exp(covariates_i_j * cov_coeffs_j)
eta_i is an individual effect, we assume eta_i follows a normal distribution N(0,omega).  
We assume a normal residual error between the predictions and the observations : Yij = f(theta_ij) + epsilon_ij (for individual i, observation j)  
with epsilon_ij following N(0,residual_error_var) 
We set betas_pop as the [log(mu1), coeff_cov1_having_effect_on_mu1, coeff_cov2_having_effect_on_mu1, ... log(mu2), ...]   so that theta_i = concat(MI, design_matrix_X_i @ betas_pop + eta_i).
We want to estimate all the fixed effects.

**SAEM:** 
PySAEM follows the SAEM principles as described in this paper Kuhn, E., & Lavielle, M. (2004). Coupling a stochastic approximation version of EM with an MCMC procedure. ESAIM Probability And Statistics, 8, 115‑131. https://doi.org/10.1051/ps:2004007.
PySAEM is a simplified version and is definitely not as complete as for example the nlmixr2 or SAEMIX R packages.  

The E-step is an MCMC procedure to sample from the log-posterior of eta. There is one chain per individual, the proposal covariance is a scaling factor times omega.
There is a burn-in phase.

The M-step performs the following updates using the learning rate alpha_k, equal to 1 during phase 1 then 1/iteration_in_phase_2 during phase 2.  

**Update of MIs**  
An optimizer L-BFGS-B is used to find the MI target. The objective function is the structural model log-likelihood of predictions made with the PDUs of the iteration and starting from the current MIs. 
MI_new = MI_old + akpha_k * (optimizer_result)
**Update of betas**  
betas_target = sum(A.inv) @ sum(B)     
with A = X_i.T @ omega.inv @ X_i
and B = X_i.T @ omega.inv @ X_i
and X_i the design matrix [nb_population_params x nb_betas], each row contains [0 , 0 ... 1, cov1, cov2, 0, 0 ...], the 1 will multiply the mu_pop, and the values for the covariates will multiply the coeffs_pop.
which is the normal equation for generalized least squares with theta_i = X_i @ betas_pop + etas_i.
so betas_new = max(betas_old + alpha_k * (sum(A.inv) @ sum(B)), simulated_annealing_factor * betas_old)
**Update of omega**  
omega_new = max(omega_old + alpha_k * (mean_of_the_outer_products_of_the_means_of_etas_per_individual), simulated_annealing_factor * omega_old)  this holds because the mean of etas tends to zero. applied to each diagonal element of omega.
**Update of residual error variance**  
var_res_new = max(sigma_res_old + sum_squared_residuals/nb_observations , simulated_annealing_factor * sigma_res_new) 

The simulated annealing helps SAEM not to get stuck in a local optimum. It is only used in phase 1 (i.e. simulated_annealing_factor = 0 during phase 2).