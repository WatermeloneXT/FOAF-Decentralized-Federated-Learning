import sys
from collections import Counter

import mlflow
import networkx as nx
import numpy as np
import torch
from networkx.utils import weighted_choice
from torch import nn
from tqdm import tqdm

from src.old.data import read_data


def random_k_out_graph(n, k, alpha, self_loops=False, seed=None):
    if alpha < 0:
        raise ValueError("alpha must be positive")
    G = nx.empty_graph(n, create_using=nx.MultiDiGraph)
    weights = Counter({v: alpha for v in G})
    for i in range(k * n):
        u = np.random.choice([v for v, d in G.out_degree() if d < k])

        if not self_loops:
            adjustment = Counter({u: weights[u]})
        else:
            adjustment = Counter()
        v = weighted_choice(weights - adjustment, seed=seed)
        G.add_edge(u, v)
        weights[v] += 1
    return G


# Optimizer selection
def get_optimizer(net):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    return optimizer


def validate(model_dict, val_loader):
    tbar = val_loader
    criterion = nn.MSELoss()
    loss_list = []
    for i in range(len(model_dict)):
        model_dict["model{0}".format(i)].eval()

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)
            logits = model_dict["model{0}".format(int(inputs[0]))](inputs)
            loss = criterion(logits, target)

            loss_list.append(loss.detach().cpu().item())
        avg_loss = np.sqrt(np.mean(loss_list))

    return avg_loss


def train(model_dict, optimizer_dict, train_loader, val_loader, graph, epochs):
    val_losses = []
    criterion = nn.MSELoss()
    val_loss = validate(model_dict, val_loader)
    mlflow.log_metric("Validation RMSE", val_loss, step=0)
    val_losses.append(val_loss)

    for e in range(epochs):
        tbar = tqdm(train_loader, file=sys.stdout)
        for i in range(len(model_dict)):
            model_dict["model{0}".format(i)].train()
        loss_list = []
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)
            optimizer = optimizer_dict["model{0}".format(int(inputs[0]))]
            optimizer.zero_grad()
            logits = model_dict["model{0}".format(int(inputs[0]))](inputs)
            loss = criterion(logits, target)
            loss.backward()  # Calculate Gradients

            # Get gradients of current Model
            model_gradient = {}  # Store current gradients
            for name, param in model_dict["model{0}".format(int(inputs[0]))].named_parameters():
                gradient = param.grad
                model_gradient[name] = gradient

            # Assign gradients to neighbors' Model
            user_n = graph.adj[int(inputs[0])]
            # user_n = list(Adjacency_List[int(inputs[0])])
            # user_n = []  #uncomment it for local learning
            for neighbor in user_n:
                neighbor_model = model_dict["model{0}".format(neighbor)]
                neighbor_optimizer = optimizer_dict["model{0}".format(neighbor)]
                neighbor_model.zero_grad()
                for name, param in neighbor_model.named_parameters():
                    param.grad = model_gradient[name]  # Assign gradients to the neighbors
                neighbor_optimizer.step()  # Neighbors' update

            optimizer.step()  # Current user's update
            loss_list.append(loss.detach().cpu().item())
            avg_loss = np.mean(loss_list)
            tbar.set_description(f"Epoch {e+1} Loss: {np.sqrt(np.round(avg_loss,7))} ")

        val_loss = validate(model_dict, val_loader)
        mlflow.log_metrics({"Train RMSE": avg_loss, "Validation RMSE": val_loss}, step=e + 1)
        mlflow.log_metric("epochs_started", e + 1)
        log_text = f"Epoch {e+1}\nTrain Loss: {np.sqrt(avg_loss)}\nValidation Loss: {val_loss}\n"
        print(log_text)
    return val_losses
