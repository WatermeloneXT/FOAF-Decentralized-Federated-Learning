import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


def centralized_validate_loop(model, val_loader, optimizer, scheduler=None):
    tbar = val_loader
    loss_fn = nn.MSELoss(reduction="sum")
    model.eval()
    total_obs = 0
    total_sum_loss = 0
    with torch.no_grad():
        for idx, (inputs, target) in enumerate(tbar):
            if inputs.ndim == 3:
                inputs = inputs.squeeze(0)
                target = target.squeeze(0)
            n_obs = inputs.shape[0]
            score = model(inputs[:, 0], inputs[:, 1])

            sum_loss = loss_fn(score, target.float()).detach().numpy()
            total_sum_loss += sum_loss
            total_obs += n_obs

        avg_loss = np.sqrt(total_sum_loss / total_obs)
        if scheduler is not None:
            scheduler.step(avg_loss)

    return avg_loss


def centralized_train_loop(model, train_loader, optimizer, progress_bar=True) -> float:
    loss_fn = nn.MSELoss(reduction="mean")
    model.train()
    total_n_obs = 0
    total_sum_loss = 0
    tbar = tqdm(train_loader) if progress_bar else train_loader
    ten_percent = len(train_loader) // 10
    for idx, (inputs, target) in enumerate(tbar):
        if inputs.ndim == 3:
            inputs = inputs.squeeze(0)
            target = target.squeeze(0)
        n_obs = inputs.shape[0]
        optimizer.zero_grad()
        score = model(inputs[:, 0], inputs[:, 1])
        loss = loss_fn(score, target.float())
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Current user's update

        total_n_obs += n_obs
        total_sum_loss += loss.detach().numpy() * n_obs
        avg_mse_loss = total_sum_loss / total_n_obs
        if (ten_percent > 0) and (idx % ten_percent == 0):
            if progress_bar:
                tbar.set_description(
                    f"Average Training Loss: {np.sqrt(avg_mse_loss):.04f} | Loss: {loss.detach():.04f}"
                )
            if np.isnan(avg_mse_loss):
                print("Training Loss is NaN")
                raise ValueError
        # if idx % 1000 == 0:
        #     tbar.set_description(f"Training Loss: {np.sqrt(avg_mse_loss):.05f} ")
    return total_sum_loss / total_n_obs
