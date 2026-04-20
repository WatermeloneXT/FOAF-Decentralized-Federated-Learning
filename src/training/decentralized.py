from time import perf_counter

import mlflow
import numpy as np
import torch
from networkx import Graph
from rich import print
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from src.models.MatrixFactorization import MF
from src.training.train_utils import EarlyStopper


def init_mf_models(n_users, n_items, sparse=False):
    # Models Initialization
    models = {}
    for x in tqdm(range(n_users)):
        models["model{0}".format(x)] = MF(n_users, n_items, sparse=sparse)
    return models


# Optimizer selection
def get_optimizer(model, sparse=False, lr=0.01, n_obs=1):
    if sparse:
        return torch.optim.SparseAdam(model.parameters(), lr=lr)
    else:
        return torch.optim.SGD(model.parameters(), lr=lr)


def get_scheduler(optimizer, kind=None):
    if kind is not None:
        return ReduceLROnPlateau(optimizer, patience=3)
    else:
        return None

def decentralized_validate_loop(user_models, val_loader):
    tbar = val_loader
    loss_fn = nn.MSELoss(reduction="mean")
    for _, user in user_models.items():
        user.model.eval()

    total_obs = 0
    total_sum_loss = 0
    user_val_rmse = []
    with torch.no_grad():
        for idx, (inputs, target) in enumerate(tbar):
            # print(f"before: {inputs.shape=}")
            # print(f"before: {target.shape=}")
            if inputs.ndim == 3:
                inputs = inputs.squeeze(0)
                target = target.squeeze(0)
            # print(f"after: {inputs.shape=}")
            # print(f"after: {target.shape=}")
            n_obs = inputs.shape[0]
            if n_obs == 0:  # No data to validate
                continue
            user_id = int(inputs[0, 0])
            score = user_models[user_id].model(inputs[:, 0], inputs[:, 1])

            user_rmse = np.sqrt(loss_fn(score, target.float()).detach().numpy())
            # sum_loss = loss_fn(score, target.float()).detach().numpy()
            # mean_loss = sum_loss / n_obs
            #
            # total_sum_loss += sum_loss
            # total_obs += n_obs
            total_obs += 1

            sched = user_models[user_id].lr_scheduler
            if sched is not None:
                sched.step(user_rmse)

            user_val_rmse.append(user_rmse)
        # total_loss = np.sqrt(total_sum_loss / total_obs)
        mean_rmse = np.mean(user_val_rmse)
    return mean_rmse


def validate_one_at_a_time(models, val_loader, schedulers):
    tbar = val_loader
    loss_fn = nn.MSELoss()
    loss_list = []
    for i in range(len(models)):
        models["model{0}".format(i)].eval()

    with torch.no_grad():
        for idx, (inputs, target) in enumerate(tbar):
            user_id = int(inputs[0, 0])
            score = models["model{0}".format(int(inputs[:, 0]))](inputs[:, 0], inputs[:, 1])
            loss = loss_fn(score, target.float())
            sched = schedulers[f"model{user_id}"]
            if sched is not None:
                sched.step(loss)

            loss_list.append(loss.detach().cpu().item())
        avg_loss = np.sqrt(np.mean(loss_list))

    return avg_loss

def share_gradient(user, users, graph, item_id=None):
    user_neighbors = graph.adj[user.id]
    # user_gradients = {
    #     name: param.grad for name, param in user.model.named_parameters() if "item" in name
    # }
    item_parameter_names = ["item_factors.weight", "item_bias.weight"]
    if user.model_name == "gmf":
        item_parameter_names.append(f"item_layer_dict.item{item_id}.weight")
        item_parameter_names.append(f"item_layer_dict.item{item_id}.bias")
    elif user.model_name == "gmf_shared":
        item_parameter_names.append("item_layers.weight")
        item_parameter_names.append("item_layers.bias")

    user_gradients = {name: user.model.get_parameter(name).grad for name in item_parameter_names}
    # print(f"Number of elements being shared: {sum([g.numel() for _, g in user_gradients.items()])}")

    for neighbor_id in user_neighbors:
        neighbor = users[neighbor_id]
        neighbor_model = neighbor.model
        neighbor_optimizer = neighbor.optimizer
        neighbor_model.zero_grad() # required - diverges without this line
        for name in item_parameter_names:
            param = neighbor_model.get_parameter(name)
            param.grad = user_gradients[name].clone()
            
        # for name, param in neighbor_model.named_parameters():
        #     # print(name)
        #     if "item" in name:
        #         param.grad = user_gradients[name]  # Assign gradients to the neighbors
        neighbor_optimizer.step()  # Neighbors' update

def decentralized_train_loop(user_models, train_loader, graph, progress_bar=True):
    loss_fn = nn.MSELoss(reduction="mean")
    for _, user in user_models.items():
        user.model.train()
    losses = np.empty(len(train_loader))
    total_n_obs = 0
    total_sum_loss = 0
    avg_loss = 0
    tbar = tqdm(train_loader) if progress_bar else train_loader
    for idx, (inputs, target) in enumerate(tbar):
        if inputs.ndim == 3:
            inputs = inputs.squeeze(0)
            target = target.squeeze(0)
        n_obs = inputs.shape[0]
        user_id = int(inputs[0, 0])
        user = user_models[user_id]
        optimizer = user.optimizer
        optimizer.zero_grad()
        score = user.model(inputs[:, 0], inputs[:, 1])
        loss = loss_fn(score, target.float())
        loss.backward()  # Calculate Gradients

        if inputs[:,1].numel() == 1:
            share_gradient(user, user_models, graph, item_id=inputs[:,1].item())
        else:
            share_gradient(user, user_models, graph)
        optimizer.step()  # Current user's update

        total_n_obs += n_obs
        total_sum_loss += loss.detach().numpy() * n_obs
        losses[idx] = loss.detach().numpy()
        # loss_list.append(loss.detach().cpu().item())
        # avg_loss = losses[:idx + 1].mean()
        # avg_loss = total_sum_loss / total_n_obs
        avg_loss = avg_loss * (idx / (idx + 1)) + loss.detach().numpy() / (idx + 1) / n_obs
        if idx % 1000 == 0:
            if progress_bar:
                tbar.set_description(
                    f"Average Training Loss: {np.sqrt(avg_loss):.04f} | Loss: {loss.detach():.04f}"
                )
            if np.isnan(avg_loss):
                raise ValueError
    return avg_loss

def decentralized_train_n_epochs(user_models, train_loader, val_loader, epochs: int, graph: Graph):
    start_time = perf_counter()
    train_losses = []
    val_losses = []
    val_loss = decentralized_validate_loop(user_models, val_loader)
    mlflow.log_metrics({"Validation RMSE": val_loss}, step=0)
    val_losses.append(val_loss)
    time_per_epoch = []

    early_stopper = EarlyStopper(patience=1, min_rel_delta=0.0001)
    for epoch in range(epochs):
        start = perf_counter()
        avg_loss = decentralized_train_loop(
            user_models=user_models, train_loader=train_loader, graph=graph
        )

        train_losses.append(avg_loss)
        val_loss = decentralized_validate_loop(user_models, val_loader)
        val_losses.append(val_loss)

        time_elapsed = perf_counter() - start
        time_per_epoch.append(time_elapsed)
        log_text = (
            f"Epoch {epoch + 1} | "
            f"Train Loss: {np.sqrt(avg_loss):.04f} | "
            f"Validation Loss: {val_loss:.04f} | "
            f"Time Elapsed: {time_elapsed:03f} sec"
        )
        print(log_text)
        mlflow.log_metrics(
            {
                "Train RMSE": avg_loss,
                "Validation RMSE": val_loss,
                "epochs_started": epoch+1,
            },
            step=epoch + 1,
        )
        if early_stopper.early_stop(val_loss):
            print("Early stopping.")
            break
    total_time = perf_counter() - start_time
    print(f"Total time elapsed: {total_time}")
    return train_losses, val_losses, time_per_epoch
