# Load packages -----------------------------------------------------------
library(tidyverse)

# Plot y against all x ----------------------------------------------------
plot_xy <- function(df, features, output, colour = "#FF4F5C") {
  df |> 
    pivot_longer(features, names_to = "var", values_to = "value")
    #ggplot(aes(x = value, y = output)) +
    #geom_point(colour = colour) +
    #facet_wrap(~ var, scales = "free") +
    #theme_bw()
}

# Compute cost ------------------------------------------------------------
compute_cost <- function(x, y, w, b) {
  # Computes the cost function for linear regression
  
  # number of training examples
  m = dim(x_train)[1]
  
  # create a variable to store cost
  cost <- 0.0
  
  # loop through all training examples and sum all squared costs
  for (i in 1:m) {
    f_wb_i <- sum(x[i,] * w) + b
    cost <- cost + (f_wb_i - y[i]) ^ 2
  }
  
  # total cost is sum of squared errors divided by 2m
  total_cost <- 1 / (2 * m) * cost
  return(total_cost)
}

# Compute gradient --------------------------------------------------------
compute_gradient <- function(x, y, w, b) {
  # Computes the gradient for linear regression
  
  # number of training examples
  m <- dim(x_train)[1]
  
  # number of features
  n <- dim(x_train)[2]
  
  # create variables to store gradients
  dj_dw <-  matrix(0.0,,n)
  dj_db = 0.0
  
  # loop through all training examples to calculate the cost
  for (i in 1:m) {
    cost <- (sum(x[i,] * w) + b) - y[i,]
    for (j in 1:n) {
      dj_dw[j] <- dj_dw[j] + cost * x[i, j]
    }
    dj_db <- dj_db + cost
  }
  dj_dw <- dj_dw / m
  dj_db <- dj_db / m
  
  return(list(dj_db = dj_db, dj_dw = dj_dw))
}

# Compute gradient descent ------------------------------------------------
gradient_descent <- function(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters) {
  graph_iters <- min(num_iters, 100000)
  j_hist <- vector("numeric", graph_iters)
  iteration <- vector("integer", graph_iters)
  
  w <- w_in
  b <- b_in
  
  for (i in 1:num_iters) {
    grads <- gradient_function(x, y, w, b)
    
    w <- w - alpha * grads$dj_dw
    b <- b - alpha * grads$dj_db
    
    if (i <= graph_iters) {
      j_hist[i] <- cost_function(x, y, w, b)
      iteration[i] = i
    }
  }
  return(list(w = w, b = b, j_hist = j_hist, iteration = iteration))
}