decentralized_gradient_descent <- function(X_list, y_list, graph,
                                           epochs = 20, lr = 0.1,
                                           true_beta = NULL,
                                           stopping_tol = 1e-6, verbose = TRUE,
                                           share =TRUE, grad = TRUE,
                                           rate = 0, neighbor_num = 2,
                                           W = NULL) {
  K <- length(X_list)
  d <- ncol(X_list[[1]])
  beta_list <- lapply(1:K, function(k) rep(0, d))

  epoch_rmse <- numeric(epochs)
  epoch_comm <- numeric(epochs)
  beta_avg_list <- list()
  beta_std_list <- list()
  grads <- vector("list", K)

  for (t in 1:epochs) {

    total_comm <- 0


    if(t < 0){

      step = lr

    }else{

      step = lr/t^rate

    }


    # Compute local gradients and update
    for (k in 1:K) {

      X <- X_list[[k]]
      y <- y_list[[k]]
      beta_k <- beta_list[[k]]

      grad_k <- -2 * t(X) %*% (y - X %*% beta_k) / nrow(X)
      grads[[k]] <- grad_k
    }


    if(share){
      # Communication: share gradient or beta with neighbors

      if(grad){

        for (k in 1:K) {

          if(is.null(neighbor_num)){


            neighbors <- which(graph[k, ] == 1)

          }else{

            neighbors <- 1:K

            if(is.null(W)){

              neighbors <- sample(neighbors, size = neighbor_num, replace = FALSE)

            }else{

              neighbors <- sample(neighbors, size = neighbor_num, replace = FALSE, prob=W[k,])

            }

          }

          grad_avg = 0

          for(j in neighbors){

            grad_avg <- grad_avg + grads[[j]]

          }

          grad_avg <- grad_avg/(length(neighbors)) #+ grads[[k]]/(length(neighbors))

          beta_list[[k]] <- beta_list[[k]] - step * grad_avg
        }

      }else{

        for (k in 1:K) {
          beta_k <- beta_list[[k]]
          beta_k <- beta_k - step * grads[[k]]
          beta_list[[k]] <- beta_k
        }

        for (k in 1:K) {

          neighbors <- which(graph[k, ] == 1)
          beta_avg <- 0
          for(j in neighbors){

            beta_avg <- beta_avg + beta_list[[j]]

          }

          beta_list[[k]] <- beta_avg/length(neighbors )

        }



      }

    }else{

      for (k in 1:K) {

        beta_list[[k]] <- beta_list[[k]] - step * grads[[k]]


      }
    }

    beta_mat <- do.call(cbind, beta_list)
    # Evaluate RMSE
    if (!is.null(true_beta)) {

      rmse <- mean(sqrt(apply((beta_mat - true_beta)^2,1,mean)))#mean(sapply(beta_list, function(b) sqrt(mean((b - true_beta)^2))))
      epoch_rmse[t] <- rmse
    }

    # Record communication and beta summary
    epoch_comm[t] <- total_comm

    beta_avg_list[[t]] <- mean(abs(rowMeans(  beta_mat - true_beta) ))
    beta_std_list[[t]] <- mean(sqrt(rowMeans(  (beta_mat - rowMeans(beta_mat))^2 )))

    # Verbose output
    if (verbose && t %% 100 == 0) {
      cat(sprintf("Epoch %d | RMSE: %.4f \n", t, epoch_rmse[t]))
    }

    # Optional stopping rule
    if (!is.null(stopping_tol) && t >= 2e4) {
      if (abs(epoch_rmse[t] - epoch_rmse[t - 1]) < stopping_tol) {
        if (verbose) cat(sprintf("Early stopping at epoch %d due to RMSE change < %.1e\n", t, stopping_tol))
        epoch_rmse <- epoch_rmse[1:t]
        epoch_comm <- epoch_comm[1:t]
        beta_avg_list <- beta_avg_list[1:t]
        beta_std_list <- beta_std_list[1:t]
        break
      }
    }
  }

  return(list(
    beta_list = beta_list,
    rmse = epoch_rmse,
    comm = epoch_comm,
    beta_avg = beta_avg_list,
    beta_std = beta_std_list
  ))
}


centralized_gradient_aggregation <- function(X_list, y_list,
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
      Xk <- X_list[[k]]
      yk <- y_list[[k]]
      nk <- nrow(Xk)

      # Client computes local gradient
      grad_k <- t(Xk) %*% (Xk %*% beta - yk) / nk

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
