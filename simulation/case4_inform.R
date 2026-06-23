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
    if (!is.null(stopping_tol) && t >= 6e4) {
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

    if (!is.null(stopping_tol) && t >= 6e4) {
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




library(igraph)
set.seed(123)
K <- 200
d <- 10
n <- 15
G <- 5  # number of subgroups

group_index <- rep(1:G, each = K / G)  # subgroup assignment
true_beta_group <- lapply(1:K, function(x) runif(d,0,5) + x )
true_beta <- lapply(1:K, function(x) true_beta_group[[group_index[x]]])
true_beta <- do.call(cbind, true_beta)
true_beta0 <- runif(K,0,1)

X_list <- lapply(1:K, function(i) matrix(rnorm(n * d), nrow = n))
y_list <- lapply(1:K, function(i) {
  X_list[[i]] %*% true_beta[,i]  + rnorm(n, 0, 1)
})


# Graph
# Define parameters
block_sizes <- rep(K/G,G)        # Sizes of each block/community

# Define block-to-block connection probabilities (2x2 matrix)
pref_matrix <- matrix(0.05, nrow = G,ncol = G,  byrow = TRUE)
pref_matrix <- pref_matrix  + diag(0.2, G,G)
# Generate SBM graph

graph <- sample_sbm(K, pref.matrix = pref_matrix, block.sizes = block_sizes, directed = FALSE, loops = FALSE)
graph <- as.matrix(as_adjacency_matrix(graph, sparse = FALSE))
# Assign community membership to vertex attribute
#V(graph)$community <- group_index

#par(mfrow=c(1,1))
#plot(graph, vertex.color = V(graph)$community, vertex.size = 5, vertex.label = NA,
#     main = "Stochastic Block Model Network")


#library(ggraph)
#library(tidygraph)

# Convert igraph to tidygraph object
#graph_tbl <- as_tbl_graph(graph)

# Plot with ggraph
#ggraph(graph_tbl, layout = "fr") +
#  geom_edge_link(color = "gray70", alpha = 0.3) +
#  geom_node_point(aes(color = as.factor(group_index), size = centrality_degree()), show.legend = FALSE) +
#  theme_void()


W = matrix(0, nrow = K, ncol =K)
for (i in 1:K) {

  for (j in 1:K) {

   W[i,j] = pref_matrix[group_index[i], group_index[j]]

  }

}

beta_result1 <- decentralized_gradient_descent(X_list, y_list, graph = graph ,
                                               epochs = 1e5, lr = 0.01, true_beta = true_beta,
                                               share =FALSE, rate =0.5 )

beta_result2 <- decentralized_gradient_descent(X_list, y_list, graph = graph  ,
                                               epochs = 1e5, lr = 0.01,
                                               true_beta = true_beta,share =TRUE, grad = TRUE,
                                               rate = 0.5, neighbor_num = 5, W = W)

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


saveRDS(data_list, "linear_subhomo_case4_inform.rds")

data_list = readRDS("linear_subhomo_case4_inform.rds")
beta_result1 = data_list$no_sharing
beta_result2 = data_list$grad_aggregate
beta_result3 = data_list$beta_aggregate
beta_result4 = data_list$central_aggregate
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
                            est_std3[length(est_std3)]),
                    bias = c(est_avg1[length(est_avg1)],
                             est_avg2[length(est_avg2)],
                             est_avg3[length(est_avg3)]),
                    rmse = c(beta_result1$rmse[length(beta_result1$rmse)],
                             beta_result2$rmse[length(beta_result2$rmse)],
                             beta_result3$rmse[length(beta_result3$rmse)] ))

rownames(result) = c( "No sharing",
                      "Gradient Aggregation",
                      "Beta Aggregation")


result

###############################################
# ggplot2
##############################################


library(ggplot2)
library(dplyr)
library(tidyr)

# Prepare combined dataframe
epochs <- 1:6e4#length(est_avg1)

df_all <- bind_rows(
  data.frame(Epoch = epochs, Method = "Local Learning", Value = est_avg1[1:6e4], Measure = "Bias"),
  data.frame(Epoch = epochs, Method = "Gossip Learning", Value = est_avg3, Measure = "Bias"),
  data.frame(Epoch = epochs, Method = "FOAF", Value = est_avg2, Measure = "Bias"),
  data.frame(Epoch = epochs, Method = "Centralized FL", Value = est_avg4, Measure = "Bias"),

  data.frame(Epoch = epochs, Method = "Local Learning", Value = est_std1[1:6e4], Measure = "Std"),
  data.frame(Epoch = epochs, Method = "Gossip Learning", Value = est_std3, Measure = "Std"),
  data.frame(Epoch = epochs, Method = "FOAF", Value = est_std2, Measure = "Std"),
  data.frame(Epoch = epochs, Method = "Centralized FL", Value = est_std4, Measure = "Std"),

  data.frame(Epoch = epochs, Method = "Local Learning", Value = beta_result1$rmse[1:6e4], Measure = "RMSE"),
  data.frame(Epoch = epochs, Method = "Gossip Learning", Value = beta_result3$rmse, Measure = "RMSE"),
  data.frame(Epoch = epochs, Method = "FOAF", Value = beta_result2$rmse, Measure = "RMSE"),
  data.frame(Epoch = epochs, Method = "Centralized FL", Value = beta_result4$rmse, Measure = "RMSE")
)

df_info = df_all
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
  labs(title = "Informative Communication Topology", x = "epochs", y = "rmse") +
  scale_x_continuous(limits = c(0, 35000),labels = function(x) x / 100) +
  scale_color_manual(values = prestige_colors) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "none")

# Std plot (with legend)
p_std <- ggplot(df_std, aes(x = Epoch, y = Value, color = Method)) +
  geom_line(size = 1) +
  labs(title = "Informative Communication Topology", x = "epochs", y = "std")  +
  scale_x_continuous(limits = c(0, 30000), labels = function(x) x / 100) +
  scale_color_manual(values = prestige_colors) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    legend.key.width = unit(1.5, "cm")
  )

library(patchwork)

p_rmse_info = p_rmse
# Combine with merged legend below
#(p_bias | p_rmse | p_std) +
( p_rmse_info) +
  plot_layout(guides = "collect") &
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    legend.position = "bottom",
    legend.box.margin = margin(t = 10),
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    legend.key.width = unit(1.5, "cm")
  )



