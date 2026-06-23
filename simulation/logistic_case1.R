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
    epoch_rmse[t] <- mean(sqrt(apply((beta_mat - beta_true)^2,1,mean)))

    if (t %% 100 == 0) {
      cat(sprintf("Epoch %d | RMSE: %.4f \n", t, epoch_rmse[t]))
    }

  }

  return(list(beta_list = beta_list, rmse = epoch_rmse,
              beta_avg = beta_avg_list,
              beta_std = beta_std_list))
}


logistic_regression_beta <- function(X_list, y_list, graph,
                                     beta_true, epochs = 1e4, lr = 0.1, rate = 0) {
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
                                     lr = 0.1,rate = 0.5, neighbor_num = 2,
                                     W = NULL) {
  K <- length(X_list)
  d <- ncol(X_list[[1]])
  beta_list <- lapply(1:K, function(k) rep(0, d))
  grads <- vector("list", K)
  epoch_rmse <- numeric(epochs)
  beta_avg_list <- list()
  beta_std_list <- list()


  for (t in 1:epochs) {

    if(t < 1000){
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
      neighbors <- sample(neighbors, size = neighbor_num, replace = FALSE)
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

    if (!is.null(stopping_tol) && t >= 5e4) {
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




# ---- Helper Functions ----


set.seed(42)
K <- 100
n <- 20
d <- 5

X_list <- lapply(1:K, function(k) matrix(rnorm(n * d), n, d))
beta_true <- matrix(1, nrow = d, ncol = K)
y_list <- lapply(1:K, function(k) rbinom(n, 1, sigmoid(X_list[[k]] %*% beta_true[,k])))

# Symmetric  graph
graph <- matrix(0, K, K)
for (i in 1:K) {
  neighbors <- sample(setdiff(1:K, i),2)
  graph[i, neighbors] <- 1
  graph[ neighbors, i] <- 1
}



# ---- Run Decentralized Logistic Regression ----
beta_result1 <- logistic_regression_nosharing(X_list, y_list, graph,lr = 0.01,
                                              beta_true = beta_true, epochs = 5e4)

beta_result2 <- logistic_regression_grad(X_list, y_list, graph, beta_true = beta_true,lr = 0.1,
                                         epochs = 5e4, rate = 0.5, neighbor_num = 20)


beta_result3 <- logistic_regression_beta(X_list, y_list, graph, beta_true = beta_true,
                                         epochs = 5e4)


beta_result4 <- centralized_logistic_gradient_aggregation(X_list, y_list,
                                                          epochs = 5e4, lr = 0.5,
                                                          true_beta = beta_true)

parameter =list(n = n, d= d, K =K, neighbor_num = 20, rate = c(0.5,0.5,0.5), beta= beta_true)
data_list = list(no_sharing = beta_result1, grad_aggregate = beta_result2,
                 beta_aggregate = beta_result3,central_aggregate = beta_result4, X_list = X_list, y_list = y_list, graph = graph, parameter = parameter)

saveRDS(data_list, "logistic_homo_case1.rds")





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

par(mfrow = c(1,1))
plot(est_avg1, type = "l", col = "black",lwd = 2,
     main = "Mean of Beta's over Iteration ",xlab = "epochs", ylab="bias",
     xlim = c(0,2e4), ylim=c(0,2))
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
     xlim = c(0,2e4), ylim=c(0,1))
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
epochs <- 1:5e4#length(est_avg1)

df_all <- bind_rows(
  data.frame(Epoch = epochs, Method = "Local Learning", Value = est_avg1[epochs], Measure = "Bias"),
  data.frame(Epoch = epochs, Method = "Gossip Learning", Value = est_avg3[epochs], Measure = "Bias"),
  data.frame(Epoch = epochs, Method = "FOAF", Value = est_avg2[epochs], Measure = "Bias"),
  data.frame(Epoch = epochs, Method = "Centralized FL", Value = est_avg4[epochs], Measure = "Bias"),

  data.frame(Epoch = epochs, Method = "Local Learning", Value = est_std1[epochs], Measure = "Std"),
  data.frame(Epoch = epochs, Method = "Gossip Learning", Value = est_std3[epochs], Measure = "Std"),
  data.frame(Epoch = epochs, Method = "FOAF", Value = est_std2[epochs], Measure = "Std"),
  data.frame(Epoch = epochs, Method = "Centralized FL", Value = est_std4[epochs], Measure = "Std"),

  data.frame(Epoch = epochs, Method = "Local Learning", Value = beta_result1$rmse[epochs], Measure = "RMSE"),
  data.frame(Epoch = epochs, Method = "Gossip Learning", Value = beta_result3$rmse[epochs], Measure = "RMSE"),
  data.frame(Epoch = epochs, Method = "FOAF", Value = beta_result2$rmse[epochs], Measure = "RMSE"),
  data.frame(Epoch = epochs, Method = "Centralized FL", Value = beta_result4$rmse[epochs], Measure = "RMSE")
)


# Ensure consistent factor levels
# Custom prestige-style colors
prestige_colors <- c(
  "Local Learning" = "#00BFC4",                 # black
  "Gossip Learning" = "#0072B2",               # blue
  "FOAF" = "#F8766D",                          # red-orange
  "Centralized FL" = "darkgray" # yellow
)
scale_color_manual(values = c(
  "Autumn" = "#F8766D",
  "Spring" = "#7CAE00",
  "Summer" = "#00BFC4",
  "Winter" = "#C77CFF"
))
# Ensure consistent factor ordering
df_all$Method <- factor(df_all$Method,
                        levels = c("Local Learning", "Gossip Learning", "FOAF", "Centralized FL"))

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

p_rmse1 = p_rmse
#homogeneous coefficient, partially heterogenous coefficient, heterogeneous coefficient, sub homogenous coefficient
p_rmse1 = p_rmse1 + labs(title = "Homogeneous Coefficient")
p_rmse2 = p_rmse2 + labs(title = "Partially Heterogenous Coefficient")
p_rmse3 = p_rmse3 + labs(title = "Heterogeneous Coefficient")
p_rmse4 = p_rmse4 + labs(title = "Sub-homogenous Coefficient")


# Combine with merged legend below
( p_rmse1 |p_rmse2)/ (p_rmse3 |p_rmse4 ) +
  plot_layout(guides = "collect") &
  theme(
    plot.title = element_text(hjust = 0.5, size = 13, face = "bold"),
    legend.position = "bottom",
    legend.box.margin = margin(t = 10),
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    legend.key.width = unit(1.5, "cm")
  )


logistic_plot = list(case1 = p_rmse1, case2 = p_rmse2, case3 = p_rmse3, case4 = p_rmse4)
saveRDS(logistic_plot, "logistic_plot.rds")
