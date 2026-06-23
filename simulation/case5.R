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


# Homo
set.seed(123)
K <- 200
d <- 10
n <- 20

#true_beta <- rep(1,d)

true_beta = matrix(NA, nrow = d, ncol = K)


X_list <- lapply(1:K, function(i) matrix(rnorm(n * d), n, d))
#y_list <- lapply(X_list, function(X) X %*% true_beta + rnorm(n,sd = 1))
y_list <- list()
for (i in 1:K) {

  temp_beta = rep(1,d)
  true_beta[,i] = temp_beta
  y_list[[i]] <- X_list[[i]] %*% temp_beta + rnorm(n,sd = 1)

}


# Symmetric  graph
graph <- matrix(0, K, K)
for (i in 1:K) {
  neighbors <- sample(setdiff(1:K, i),5)
  graph[i, neighbors] <- 1
  graph[ neighbors, i] <- 1
}



beta_result1 <- decentralized_gradient_descent(X_list, y_list, graph = graph,
                                               epochs = 1e5, lr = 0.01, true_beta = true_beta,
                                               share =FALSE, rate =0.5 )

beta_result2 <- decentralized_gradient_descent(X_list, y_list, graph = graph+ diag(K) ,
                                               epochs = 1e5, lr = 0.015,
                                               true_beta = true_beta,share =TRUE, grad = TRUE,
                                               rate = 0.5,neighbor_num = 20)

beta_result3 <- decentralized_gradient_descent(X_list, y_list, graph = graph ,
                                               epochs = 1e5, lr = 0.01,
                                               true_beta = true_beta,share =TRUE, grad = FALSE,
                                               rate = 0.5)

beta_result4 <- centralized_gradient_aggregation(X_list, y_list,
                                                 epochs = 1e5, lr = 0.01,
                                                 true_beta = true_beta)




parameter =list(n = n, d= d, K =K, beta_num = 5, neighbor_num = 20, rate = c(0.5,0.5,0.5), beta= true_beta)
data_list = list(no_sharing = beta_result1, grad_aggregate = beta_result2,
                 beta_aggregate = beta_result3,central_aggregate = beta_result4, X_list = X_list, y_list = y_list, graph = graph, parameter = parameter)

#saveRDS(data_list, "linear_homo_case1.rds")


#################################################
# Bias and Variance Plot
##############################################
est_avg1 = unlist( beta_result1$beta_avg )
est_std1 = unlist(beta_result1$beta_std )

est_avg2 = unlist( beta_result2$beta_avg )
est_std2 = unlist(beta_result2$beta_std )

est_avg3 = unlist(beta_result3$beta_avg )
est_std3 = unlist(beta_result3$beta_std )

est_avg4 = unlist(beta_result4$beta_avg )
est_std4 = unlist(beta_result4$beta_std )


par(mfrow = c(1,3))

plot(est_avg1, type = "l", col = "black",lwd = 2,
     main = "Mean of Beta's over Iteration ",xlab = "epochs", ylab="bias",
     xlim = c(0,2e4), ylim=c(0,1))
lines(est_avg2, col = "red",lwd = 2)
lines(est_avg3, col = "blue",lwd = 2)
lines(est_avg4, col = "yellow",lwd = 2)
legend("topright", legend=c( "Local Learning",
                             "Gossip Learning",
                             "FOAF",
                             "Centralized Federated Learning"),
       fill = c("black","blue","red", "yellow")
)


plot(est_std1, type = "l", col = "black",lwd = 2,
     main = "Variance of Beta's over Iteration ", ylab = "std",
     xlim = c(0,2e4), ylim=c(0,0.5))
lines(est_std2, col = "red",lwd = 2)
lines(est_std3, col = "blue",lwd = 2)
lines(est_std4, col = "yellow",lwd = 2)
legend("topright", legend=c( "Local Learning",
                             "Gossip Learning",
                             "FOAF",
                             "Centralized Federated Learning"),
       fill = c("black","blue","red", "yellow")
)

plot((beta_result1$rmse), type = "l", lwd = 2,
     ylim = c(0,1),xlim = c(0,2e4) ,
     main = "RMSE over Iteration", ylab = "rmse",xlab = "epochs")

lines((beta_result2$rmse), type = "l", lwd = 2,col = "red")
lines((beta_result3$rmse), type = "l", lwd = 2,col = "blue")
lines(beta_result4$rmse, col = "yellow",lwd = 2)

legend("topright", legend=c( "Local Learning",
                             "Gossip Learning",
                             "FOAF",
                             "Centralized Federated Learning"),
       fill = c("black","blue","red", "yellow")
)
#################################################
# Result Table
##############################################
result = data.frame(std = c(est_std1[length(est_std1)],
                            est_std2[length(est_std2)],
                            est_std3[length(est_std3)],
                            est_std4[length(est_std4)]),
                    bias = c(est_avg1[length(est_avg1)],
                             est_avg2[length(est_avg2)],
                             est_avg3[length(est_avg3)],
                             est_avg4[length(est_avg4)]),
                    rmse = c(beta_result1$rmse[length(beta_result1$rmse)],
                             beta_result2$rmse[length(beta_result2$rmse)],
                             beta_result3$rmse[length(beta_result3$rmse)],
                             beta_result4$rmse[length(beta_result4$rmse)]))

rownames(result) = c( "No sharing",
                      "Gradient Aggregation",
                      "Beta Aggregation",
                      "Centralized Learning")


result

###############################################
# ggplot2
##############################################


library(ggplot2)
library(dplyr)
library(tidyr)

# Prepare combined dataframe
epochs <- 1:length(est_avg1)

df_all <- bind_rows(
  data.frame(Epoch = epochs, Method = "Local Learning", Value = est_avg1, Measure = "Bias"),
  data.frame(Epoch = epochs, Method = "Gossip Learning", Value = est_avg3, Measure = "Bias"),
  data.frame(Epoch = epochs, Method = "FOAF", Value = est_avg2, Measure = "Bias"),
  data.frame(Epoch = epochs, Method = "Centralized Federated Learning", Value = est_avg4, Measure = "Bias"),

  data.frame(Epoch = epochs, Method = "Local Learning", Value = est_std1, Measure = "Std"),
  data.frame(Epoch = epochs, Method = "Gossip Learning", Value = est_std3, Measure = "Std"),
  data.frame(Epoch = epochs, Method = "FOAF", Value = est_std2, Measure = "Std"),
  data.frame(Epoch = epochs, Method = "Centralized Federated Learning", Value = est_std4, Measure = "Std"),

  data.frame(Epoch = epochs, Method = "Local Learning", Value = beta_result1$rmse, Measure = "RMSE"),
  data.frame(Epoch = epochs, Method = "Gossip Learning", Value = beta_result3$rmse, Measure = "RMSE"),
  data.frame(Epoch = epochs, Method = "FOAF", Value = beta_result2$rmse, Measure = "RMSE"),
  data.frame(Epoch = epochs, Method = "Centralized Federated Learning", Value = beta_result4$rmse, Measure = "RMSE")
)


# Ensure consistent factor levels
# Custom prestige-style colors
prestige_colors <- c(
  "Local Learning" = "#00BFC4",                 # black
  "Gossip Learning" = "#0072B2",               # blue
  "FOAF" = "#F8766D",                          # red-orange
  "Centralized Federated Learning" = "darkgray" # yellow
)
scale_color_manual(values = c(
  "Autumn" = "#F8766D",
  "Spring" = "#7CAE00",
  "Summer" = "#00BFC4",
  "Winter" = "#C77CFF"
))
# Ensure consistent factor ordering
df_all$Method <- factor(df_all$Method,
                        levels = c("Local Learning", "Gossip Learning", "FOAF", "Centralized Federated Learning"))

df_all$Measure <- factor(df_all$Measure,
                         levels = c("Bias", "RMSE", "Std"))

# Subset data
df_bias <- df_all %>% filter(Measure == "Bias")
df_rmse <- df_all %>% filter(Measure == "RMSE")
df_std  <- df_all %>% filter(Measure == "Std")


# Bias Plot with linetype
# Bias plot
p_bias <- ggplot(df_bias, aes(x = Epoch, y = Value, color = Method)) +
  geom_line(size = 1) +
  labs(title = "", x = "epochs", y = "bias") +
  scale_x_continuous(labels = function(x) x / 100) +
  scale_color_manual(values = prestige_colors) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "none")

# RMSE plot
p_rmse <- ggplot(df_rmse, aes(x = Epoch, y = Value, color = Method)) +
  geom_line(size = 1) +
  labs(title = "", x = "epochs", y = "rmse") +
  scale_x_continuous(labels = function(x) x / 100) +
  scale_color_manual(values = prestige_colors) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "none")

# Std plot (with legend)
p_std <- ggplot(df_std, aes(x = Epoch, y = Value, color = Method)) +
  geom_line(size = 1) +
  labs(title = "", x = "epochs", y = "std") +
  scale_x_continuous(labels = function(x) x / 100) +
  scale_color_manual(values = prestige_colors) + ylim(0,1) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    legend.key.width = unit(1.5, "cm")
  )

library(patchwork)


# Combine with merged legend below
(p_bias | p_rmse | p_std) +
  plot_layout(guides = "collect") &
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    legend.position = "bottom",
    legend.box.margin = margin(t = 10),
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    legend.key.width = unit(1.5, "cm")
  )



