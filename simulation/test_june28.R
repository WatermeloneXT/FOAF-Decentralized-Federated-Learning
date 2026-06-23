source("wrapup/code_june28.r")
set.seed(42)
K <- 50
n <- 100
d <- 5

X_list <- lapply(1:K, function(k) matrix(rnorm(n * d), n, d))
beta_true <- matrix(1, nrow = d, ncol = K)
y_list <- lapply(1:K, function(k) rbinom(n, 1, sigmoid(X_list[[k]] %*% beta_true[,k])))

# Symmetric  graph
graph <- matrix(0, K, K)
for (i in 1:K) {
  neighbors <- sample(setdiff(1:K, i),20)
  graph[i, neighbors] <- 1
  graph[ neighbors, i] <- 1
}

#w = runif(K)
#W = w %*% t(w)
#W = matrix(1/K, nrow = K, ncol = K)

beta_result1 <- logistic_regression_nosharing(X_list, y_list, graph,
                                              beta_true = beta_true, epochs = 1e5)

beta_result2 <- logistic_regression_beta(X_list, y_list, graph, beta_true = beta_true,
                                         epochs = 1e4,  neighb_num = 5)

beta_result3 <- logistic_regression_grad(X_list, y_list, graph, beta_true = beta_true,
                                         epochs = 2e4, rate = 0.5, neighb_num = 20,W = W)


beta_result4 <- centralized_logistic_gradient_aggregation(X_list, y_list,
                                                          epochs = 1e5, lr = 0.5,
                                                          true_beta = true_beta)

#################################################
# rmse plot
#################################################
par(mfrow=c(1,1))
plot(beta_result1$rmse, type = "l", ylim = c(0,1),xlim = c(0,2e4) , main = "RMSE over Iteration", ylab = "rmse")

lines(beta_result2$rmse, type = "l", col = "blue", ylim = c(0,1))

lines(beta_result3$rmse, type = "l", col = "red")

lines(beta_result4$rmse, type = "l", col = "yellow")

legend("topright", legend=c( "No sharing",
                             "Beta Aggregation",
                             "Gradient Aggregation",
                             "Centralized FL"),
       fill = c("black","blue","red", "yellow")
)

