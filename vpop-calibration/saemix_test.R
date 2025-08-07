library(saemix)
library(deSolve)
rm(list = ls())

# Define the PK Two-Compartment Model 
# y[1] = A1 (Amount in central compartment)
# y[2] = A2 (Amount in peripheral compartment)
pk_ode_system <- function(t, y, parms) {
  k12 <- as.numeric(parms["k12"])
  k21 <- as.numeric(parms["k21"])
  k_el <- as.numeric(parms["k_el"])
  dA1dt <- k21 * y[2] - k12 * y[1] - k_el * y[1]
  dA2dt <- k12 * y[1] - k21 * y[2]
  return(list(c(dA1dt, dA2dt)))
}

# Define the structural model for saemix 
# xidep will contains time and ytype (1 for A1 and 2 for A2) columns
pk_saemix_model <- function(psi, id, xidep) {

  # Get unique IDs from the id vector passed to the function
  unique_ids <- unique(id)
  # Initialize predictions for all observations
  predictions <- numeric(nrow(xidep))
  
  # Initial conditions
  A1_initial <- 4
  A2_initial <- 0
  y0 <- c(A1 = A1_initial, A2 = A2_initial)
  
  # Solve ODEs once for each individual
  sol_list <- list() # To store ODE solutions for each individual
  
  for (ind_id in unique_ids) { # Iterate over unique IDs
    # Get parameters for current individual
    ind_k12 <- psi[ind_id, 1] 
    ind_k21 <- psi[ind_id, 2]
    ind_k_el <- psi[ind_id, 3]
    ind_parms <- c(k12 = ind_k12, k21 = ind_k21, k_el = ind_k_el)
    
    # Get time points for this individual (currently the same for all individuals)
    current_ind_times <- xidep$time[id == ind_id] 
    times_for_ode <- sort(unique(round(current_ind_times, digits = 8)))
    
    # Solve ODEs for this individual
    sol_ind <- ode(y = y0, times = times_for_ode, func = pk_ode_system, parms = ind_parms, method = "bdf")
    sol_list[[as.character(ind_id)]] <- sol_ind 
  }
  
  # iterate through each observation in sol_list (individual, time, A1, A2) to fill predictions (corresponding to xidep with time, ytype, value (the individual is indicated by id[same row]))
  for (j in 1:nrow(xidep)) {
    current_id_obs <- id[j]
    current_time_obs <- xidep$time[j]
    current_ytype_obs <- xidep$ytype[j]
    
    # Retrieve the ODE solution for the current individual
    sol_ind <- sol_list[[as.character(current_id_obs)]]

    time_diffs <- abs(sol_ind[, "time"] - current_time_obs)
    closest_index <- which.min(time_diffs)
    
    if (length(closest_index) == 0 || time_diffs[closest_index] >= 1e-2) {
      predictions[j] <- NA
      warning(paste("No sufficiently close ODE solution found for id", current_id_obs, "at time", current_time_obs, "after rounding."))
      next
    }
    
    predicted_row_index <- closest_index
    
    if (current_ytype_obs == 1) { # Compartment A1
      predictions[j] <- sol_ind[predicted_row_index, "A1"]
    } else if (current_ytype_obs == 2) { # Compartment A2
      predictions[j] <- sol_ind[predicted_row_index, "A2"]
    } else {
      predictions[j] <- NA 
      warning(paste("Unknown ytype for observation id", current_id_obs, "at time", current_time_obs))
    }
  }
  return(predictions)
}

# Simulate Data 
set.seed(42)

# True population parameters
V1 <- 15.0  # volume of compartment 1
V2 <- 50.0
Q <- 10.0  # intercompartmental clearance
true_k_el_pop <- 0.15 # elimination rate of compartment 1

true_k12_pop <- Q / V1 # 0.667
true_k21_pop <- Q / V2 # 0.2

# Variance of eta (log-transformed individual deviations)
true_omega <- matrix(c(0.1^2, 0.0, 0.0,
                       0.0, 0.1^2, 0.0,
                       0.0, 0.0, 0.2^2), nrow = 3, byrow = TRUE)
true_residual_sigma_A1 <- 0.1
true_residual_sigma_A2 <- 0.1 
num_individuals <- 20
time_span_start <- 0
time_span_end <- 24
nb_steps <- 100 # 200
time_steps <- seq(from = time_span_start, to = time_span_end, length.out = nb_steps)
all_individual_data_list <- list()

# Initial conditions for ODEs (constant for all individuals in simulation)
A1_initial_sim <- 4
A2_initial_sim <- 0
y0_sim <- c(A1 = A1_initial_sim, A2 = A2_initial_sim)

# Helper function for multivariate normal random numbers (replacement for mvrnorm)
rmvnorm_custom <- function(n = 1, mu, Sigma) {
  p <- ncol(Sigma)
  if (missing(mu)) { mu <- rep(0, p) }
  L <- chol(Sigma)
  Z <- matrix(rnorm(n * p), nrow = p, ncol = n)
  X <- mu + t(L) %*% Z
  return(t(X))
}

for (i in 1:num_individuals) {
  # Simulate individual random effects (eta_i ~ N(0, Omega))
  eta_i <- rmvnorm_custom(n = 1, mu = c(0, 0, 0), Sigma = true_omega)
  
  # Simulate true individual parameters (log-normal distribution)
  k12_ind <- true_k12_pop * exp(eta_i[1])
  k21_ind <- true_k21_pop * exp(eta_i[2])
  k_el_ind <- true_k_el_pop * exp(eta_i[3])
  
  individual_params_sim <- c(k12 = k12_ind, k21 = k21_ind, k_el = k_el_ind)
  
  # Solve ODEs for true means
  sol_true <- ode(
    y = y0_sim,
    times = time_steps,
    func = pk_ode_system,
    parms = individual_params_sim,
    method = "bdf"
  )
  true_A1 <- sol_true[, "A1"]
  true_A2 <- sol_true[, "A2"]
  
  # Add noise to both A1 and A2 observations
  observed_A1 <- true_A1 + rnorm(length(time_steps), mean = 0, sd = true_residual_sigma_A1)
  observed_A2 <- true_A2 + rnorm(length(time_steps), mean = 0, sd = true_residual_sigma_A2)
  
  # Ensure observations are not negative
  observed_A1[observed_A1 < 0] <- 0
  observed_A2[observed_A2 < 0] <- 0
  
  # Combine A1 and A2 observations for this individual with 'ytype'
  df_A1 <- data.frame(id = i, time = time_steps, conc = observed_A1, ytype = 1) # ytype 1 for A1
  df_A2 <- data.frame(id = i, time = time_steps, conc = observed_A2, ytype = 2) # ytype 2 for A2
  
  all_individual_data_list[[i]] <- rbind(df_A1, df_A2)
}

simulated_data <- do.call(rbind, all_individual_data_list)
# sort data by ID, then YTYPE, then TIME 
simulated_data <- simulated_data[order(simulated_data$id, simulated_data$ytype, simulated_data$time), ]


# Prepare saemix.data object
saemix.data <- saemixData(
  name.data = simulated_data,
  header = TRUE,
  name.group = c("id"),
  name.predictors = c("time", "ytype"), 
  name.response = c("conc"),
  name.ytype = "ytype"
)

# Define saemix.model object
initial_pop_mus <- c(k12 = 0.6, k21 = 0.2, k_el = 0.15) 

saemix.model <- saemixModel(
  model = pk_saemix_model,
  description = "Two-compartment PK model with A1 and A2 outputs",
  psi0 = initial_pop_mus,
  transform.par = c(1, 1, 1), # individual params = pop params * exp (eta)
  fixed.estim = c(1, 1, 1),
  covariance.model = matrix(c(1, 0, 0,
                              0, 1, 0,
                              0, 0, 1), nrow = 3, byrow = TRUE),
  error.model = c("constant", "constant"), 
  error.init = c(0.1, 0.1) # sigma 
)

# Set saemix.options
num_individuals <- length(unique(simulated_data$id))
saemix.options <- list(
  seed = 42,
  nbiter.saemix = c(100, 50),
  nb.chains = num_individuals, # Setting total number of chains to the number of individuals
  displayProgress = TRUE,
  nbdisplay = 20,
  trace=TRUE
)

# Run
cat("Running SAEM algorithm...\n")
saemix.fit <- saemix(saemix.model, saemix.data, saemix.options)
cat("SAEM algorithm finished.\n")

# Display results
print(saemix.fit)
