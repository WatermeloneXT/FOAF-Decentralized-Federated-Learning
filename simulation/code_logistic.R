sigmoid <- function(z) 1 / (1 + exp(-z))

logistic_regression_nosharing <- function(X_list, y_list, graph,
                                          beta_true, epochs = 1e4, lr = 0.1) {
  K <- length(X_list)
  d <- ncol(X_list[[1]])
  beta_list <- lapply(1:K, function(k) rep(0, d))
  beta_avg_list <- list()
  beta_std_list <- list()
  epoch_rmse <- numeric(epochs)

  for (t in 1:epochs) {
    for (k in 1:K) {
      X <- X_list[[k]]
      y <- y_list[[k]]
      beta_k <- beta_list[[k]]

      # Logistic gradient step
      probs <- sigmoid(X %*% beta_k)
      grad <- t(X) %*% (probs - y) / nrow(X)
      beta_k <- beta_k - lr/sqrt(t) * grad

      # Gossip: pick one neighbor and average
      #neighbors <- which(graph[k, ] == 1)
      #j <- sample(neighbors, 1)
      #beta_j <- beta_list[[j]]
      #beta_k <- (beta_k + beta_j) / 2

      beta_list[[k]] <- beta_k
    }

    # Compute average beta and RMSE vs true_beta
    beta_mat <- do.call(cbind, beta_list)
    beta_avg_list[[t]] <- rowMeans(beta_mat)
    beta_std_list[[t]] <- apply(beta_mat, 1, sd)
    epoch_rmse[t] <- mean(sqrt(apply((beta_mat - true_beta)^2,1,mean)))

    if (t %% 100 == 0) {
      cat(sprintf("Epoch %d | RMSE: %.4f \n", t, epoch_rmse[t]))
    }

  }

  return(list(beta_list = beta_list, rmse = epoch_rmse,
              beta_avg = beta_avg_list,
              beta_std = beta_std_list))
}


logistic_regression_beta <- function(X_list, y_list, graph,
                                     beta_true, epochs = 1e4, lr = 0.1, rate = 0, neighb_num = 2) {
  K <- length(X_list)
  d <- ncol(X_list[[1]])
  beta_list <- lapply(1:K, function(k) rep(0, d))
  beta_avg_list <- list()
  beta_std_list <- list()
  epoch_rmse <- numeric(epochs)

  for (t in 1:epochs) {

    step = lr/t^rate

    for (k in 1:K) {
      X <- X_list[[k]]
      y <- y_list[[k]]
      beta_k <- beta_list[[k]]

      # Logistic gradient step
      probs <- sigmoid(X %*% beta_k)
      grad <- t(X) %*% (probs - y) / nrow(X)
      beta_k <- beta_k -  step * grad
      beta_list[[k]] <- beta_k
    }
    # Gossip: pick one neighbor and average

    for (k in 1:K) {

      X <- X_list[[k]]
      y <- y_list[[k]]

      neighbors <- which(graph[k, ] == 1)

      # Random Sample neighbors
      #neighbors <- sample(neighbors,neighb_num ,replace = FALSE)
      beta_avg <- 0
      for(j in neighbors){

        beta_avg <- beta_avg + beta_list[[j]]

      }

      beta_list[[k]] <- beta_avg/length(neighbors )
      #probs <- sigmoid(X %*% beta_k)
      #grad <- t(X) %*% (probs - y) / nrow(X)

      #beta_list[[k]] <- beta_list[[k]] - step * grad_avg
    }


    # Compute average beta and RMSE vs true_beta
    beta_mat <- do.call(cbind, beta_list)
    beta_avg_list[[t]] <- abs(rowMeans(beta_mat - beta_true))
    beta_std_list[[t]] <- mean(sqrt(rowMeans(  (beta_mat - rowMeans(beta_mat))^2 )))
    epoch_rmse[t] <- mean(sqrt(apply((beta_mat - beta_true)^2,1,mean)))


    if (t %% 100 == 0) {
      cat(sprintf("Epoch %d | RMSE: %.4f \n", t, epoch_rmse[t]))
    }

  }

  return(list(beta_list = beta_list, rmse = epoch_rmse,
              beta_avg = beta_avg_list,
              beta_std = beta_std_list))
}


logistic_regression_grad <- function(X_list, y_list, graph,
                                     beta_true, epochs = 1e4,
                                     lr = 0.1,rate = 0, neighb_num = 2,
                                     W = NULL) {
  K <- length(X_list)
  d <- ncol(X_list[[1]])
  beta_list <- lapply(1:K, function(k) rep(0, d))
  grads <- vector("list", K)
  epoch_rmse <- numeric(epochs)
  beta_avg_list <- list()
  beta_std_list <- list()


  for (t in 1:epochs) {

    if(t < 2000){
      step = lr
    }else{
      step = lr/t^rate
    }

    for (k in 1:K) {
      X <- X_list[[k]]
      y <- y_list[[k]]
      beta_k <- beta_list[[k]]

      # Logistic gradient step
      probs <- sigmoid(X %*% beta_k)
      grad <- t(X) %*% (probs - y) / nrow(X)
      #beta_k <- beta_k - step * grad
      #beta_list[[k]] <- beta_k
      grads[[k]] <- grad
    }


    for (k in 1:K) {

      neighbors <- 1:K#which(graph[k, ] == 1)
      neighbors <- sample(neighbors, size = 20, replace = FALSE)
      #neighbors <- sample(neighbors, size = 20, replace = FALSE, prob=W[k,])

      grad_avg = 0

      for(j in neighbors){

        grad_avg <- grad_avg + grads[[j]]

      }

      grad_avg <- grad_avg/(length(neighbors)+1) #+ grads[[k]]/(length(neighbors)+1)

      beta_list[[k]] <- beta_list[[k]] - step * grad_avg
    }

    # Gossip: pick one neighbor and average



    # Compute average beta and RMSE vs true_beta
    beta_mat <- do.call(cbind, beta_list)
    beta_avg_list[[t]] <- abs(rowMeans(beta_mat - beta_true))
    beta_std_list[[t]] <- mean(sqrt(rowMeans(  (beta_mat - rowMeans(beta_mat))^2 )))
    epoch_rmse[t] <- mean(sqrt(apply((beta_mat - beta_true)^2,1,mean)))

    if (t %% 100 == 0) {
      cat(sprintf("Epoch %d | RMSE: %.4f \n", t, epoch_rmse[t]))
    }
  }

  return(list(beta_list = beta_list, rmse = epoch_rmse,
              beta_avg = beta_avg_list,
              beta_std = beta_std_list))
}


centralized_logistic_gradient_aggregation <- function(X_list, y_list,
                                                      epochs = 100,
                                                      lr = 0.01, stopping_tol = 1e-6,
                                                      true_beta = NULL,verbose = TRUE) {
  K <- length(X_list)
  d <- ncol(X_list[[1]])
  beta <- rep(0, d)
  epoch_rmse <- numeric(epochs)
  beta_list <- lapply(1:K, function(k) rep(0, d))
  beta_avg_list <- list()
  beta_std_list <- list()

  for (t in 1:epochs) {

    grad_sum <- rep(0, d)

    # Each client computes its own gradient and sends to server
    for (k in 1:K) {
      X_k <- X_list[[k]]
      y_k <- y_list[[k]]
      nk <- nrow(X_k)
      beta_k <- beta_list[[k]]

      # Client computes local gradient
      p_hat <- sigmoid(X_k %*% beta_k)
      grad_k <- t(X_k) %*% (p_hat - y_k) / nrow(X_k)

      # Send to server (accumulate)
      grad_sum <- grad_sum + grad_k
    }

    # Server computes average gradient and updates beta
    grad_avg <- grad_sum / K
    beta <- beta - lr * grad_avg

    for (j in 1:K) {

      beta_list[[j]] = beta

    }

    beta_mat <- do.call(cbind, beta_list)

    # RMSE if true beta known
    if (!is.null(true_beta)) {
      epoch_rmse[t] <- mean(sqrt(apply((beta_mat - true_beta)^2,1,mean)))
    }

    if (verbose && t %% 100 == 0) {
      cat(sprintf("Epoch %d | RMSE: %.4f \n", t, epoch_rmse[t]))
    }

    beta_avg_list[[t]] <- mean(abs(rowMeans(  beta_mat - true_beta) ))
    beta_std_list[[t]] <- mean(sqrt(rowMeans(  (beta_mat - rowMeans(beta_mat))^2 )))

    if (!is.null(stopping_tol) && t >= 2e4) {
      if (abs(epoch_rmse[t] - epoch_rmse[t - 1]) < stopping_tol) {
        if (verbose) cat(sprintf("Early stopping at epoch %d due to RMSE change < %.1e\n", t, stopping_tol))
        epoch_rmse <- epoch_rmse[1:t]
        beta_avg_list <- beta_avg_list[1:t]
        beta_std_list <- beta_std_list[1:t]
        break
      }
    }

  }

  return(list(
    beta_list = beta_list,
    rmse = epoch_rmse,
    beta_avg = beta_avg_list,
    beta_std = beta_std_list
  ))
}


