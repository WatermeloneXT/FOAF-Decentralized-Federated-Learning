# Case1

decentralized_gradient_descent <- function(X_list, y_list, graph,
                                           epochs = 20, lr = 0.1,
                                           true_beta = NULL,
                                           stopping_tol = 1e-6, verbose = TRUE,
                                           share =TRUE, grad = TRUE,
                                           rate = 0, neighbor_num = 2,
                                           W = NULL, symetric = FALSE) {
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
      graph_mat = matrix(0,nrow = K, ncol =K)


       if(!symetric ){


            for (k in 1:K) {

            neighbors <- 1:K

            if(is.null(W)){

              neighbors <- sample(neighbors, size = neighbor_num, replace = FALSE)

            }else{

              neighbors <- sample(neighbors, size = neighbor_num, replace = FALSE, prob=W[k,])

            }
            graph_mat[k,neighbors] <- 1

          }

          }else{

            for (k in 1:K) {

         neighbors <- 1:K

         if(is.null(W)){

           neighbors <- sample(neighbors, size = neighbor_num, replace = FALSE)

         }else{

           neighbors <- sample(neighbors, size = neighbor_num, replace = FALSE, prob=W[k,])

         }
         graph_mat[k,neighbors] <- 1
         graph_mat[neighbors,k] <- 1

       }
      }

       for (k in 1:K) {
          neighbors <- which(graph_mat[k,] == 1)
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

beta_result_noshare <- decentralized_gradient_descent(X_list, y_list, graph = graph,
                                               epochs = 3e4, lr = 0.01, true_beta = true_beta,
                                               share =FALSE, rate =0.5 )

beta_result_direc <- decentralized_gradient_descent(X_list, y_list, graph = graph ,
                                               epochs = 3e4, lr = 0.01,
                                               true_beta = true_beta,share =TRUE, grad = TRUE,
                                               rate = 0.5,neighbor_num = 1,,symetric = FALSE)


beta_result_undirec <- decentralized_gradient_descent(X_list, y_list, graph = graph ,
                                                    epochs = 3e4, lr = 0.01,
                                                    true_beta = true_beta,share =TRUE, grad = TRUE,
                                                    rate = 0.5,neighbor_num = 1,symetric = TRUE)


#################################################
# Bias and Variance Plot
##############################################
est_avg1 = unlist( beta_result_noshare$beta_avg )
est_std1 = unlist(beta_result_noshare$beta_std )

est_avg2 = unlist( beta_result_direc$beta_avg )
est_std2 = unlist(beta_result_direc$beta_std )

est_avg3 = unlist(beta_result_undirec$beta_avg )
est_std3 = unlist(beta_result_undirec$beta_std )




par(mfrow = c(1,3))

plot(est_avg1, type = "l", col = "black",lwd = 2,
     main = "Mean of Beta's over Iteration ",xlab = "epochs", ylab="bias",
     xlim = c(0,2e4), ylim=c(0,1))
lines(est_avg2, col = "red",lwd = 2)
lines(est_avg3, col = "blue",lwd = 2)
legend("topright", legend=c( "Local Learning",
                             "Directed Graph",
                             "Undirected Graph"
                             ),
       fill = c("black","red","blue")
)


plot(est_std1, type = "l", col = "black",lwd = 2,
     main = "Variance of Beta's over Iteration ", ylab = "std",
     xlim = c(0,2e4), ylim=c(0,0.5))
lines(est_std2, col = "red",lwd = 2)
lines(est_std3, col = "blue",lwd = 2)
legend("topright", legend=c( "Local Learning",
                             "Directed Graph",
                             "Undirected Graph"
),
fill = c("black","red","blue")
)


plot((beta_result_noshare$rmse), type = "l", lwd = 2,
     ylim = c(0,1),xlim = c(0,2e4) ,
     main = "RMSE over Iteration", ylab = "rmse",xlab = "epochs")

lines((beta_result_direc$rmse), type = "l", lwd = 2,col = "red")
lines((beta_result_undirec$rmse), type = "l", lwd = 2,col = "blue")

legend("topright", legend=c( "Local Learning",
                             "Directed Graph",
                             "Undirected Graph"
),
fill = c("black","red","blue")
)



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
  data.frame(Epoch = epochs, Method = "FOAF with Directed Graph" , Value = est_avg3, Measure = "Bias"),
  data.frame(Epoch = epochs, Method = "FOAF with Undirected Graph", Value = est_avg2, Measure = "Bias"),

  data.frame(Epoch = epochs, Method = "Local Learning", Value = est_std1, Measure = "Std"),
  data.frame(Epoch = epochs, Method = "FOAF with Directed Graph" , Value = est_std3, Measure = "Std"),
  data.frame(Epoch = epochs, Method = "FOAF with Undirected Graph", Value = est_std2, Measure = "Std"),

  data.frame(Epoch = epochs, Method = "Local Learning", Value = beta_result_noshare$rmse, Measure = "RMSE"),
  data.frame(Epoch = epochs, Method = "FOAF with Directed Graph" , Value = beta_result_direc$rmse, Measure = "RMSE"),
  data.frame(Epoch = epochs, Method = "FOAF with Undirected Graph", Value = beta_result_undirec$rmse, Measure = "RMSE"),
)



# Filter RMSE data
df_rmse <- subset(df_all, Measure == "RMSE" & Method %in% c("FOAF with Directed Graph", "FOAF with Undirected Graph"))

# Plot FOAF with Directed Graph
df_directed <- subset(df_rmse, Method == "FOAF with Directed Graph")
plot1 = ggplot(df_directed, aes(x = Epoch, y = Value)) +
  geom_line(color = "#0072B2") +
  ggtitle("FOAF with Directed Graph") +
  ylab("RMSE") + xlab("Epoch")

# Plot FOAF with Undirected Graph
df_undirected <- subset(df_rmse, Method == "FOAF with Undirected Graph")
plot2 = ggplot(df_undirected, aes(x = Epoch, y = Value)) +
  geom_line(color = "#F8766D") +
  ggtitle("FOAF with Undirected Graph") +
  ylab("RMSE") + xlab("Epoch")

plot1 + plot2

# Ensure consistent factor levels
# Custom prestige-style colors
prestige_colors <- c(
  "Local Learning" = "#00BFC4",                 # black
  "FOAF with Directed Graph" = "#0072B2",               # blue
  "FOAF with Undirected Graph" = "#F8766D"                        # red-orange
)
scale_color_manual(values = c(
  "Autumn" = "#F8766D",
  "Spring" = "#7CAE00",
  "Summer" = "#00BFC4",
  "Winter" = "#C77CFF"
))
# Ensure consistent factor ordering
df_all$Method <- factor(df_all$Method,
                        levels = c("Local Learning", "FOAF with Directed Graph", "FOAF with Undirected Graph"))

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



