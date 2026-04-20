import argparse

import mlflow
from rich import print
from sklearn.model_selection import train_test_split
from torch.optim import SGD

from src.data_utils import DataName, create_dataloader, read_in_data
from src.models.MatrixFactorization import MF
from src.training.centralized import (centralized_train_loop,
                                      centralized_validate_loop)
from src.training.train_utils import EarlyStopper


def run(data_name, batch_size, n_epochs, lr, wd, mom, n_factors, sparse=False):
    mlflow.log_params(
        {
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "lr": lr,
            "wd": wd,
            "mom": mom,
            "n_factors": n_factors,
            "sparse": sparse,
        }
    )
    mlflow.set_tags({"data_name": data_name})
    train_df, test_df = read_in_data(data_name)
    n_users, n_items, _ = train_df.nunique()
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=0
    )
    train_dl = create_dataloader(train_df, dl_type="centralized", batch_size=batch_size)
    val_dl = create_dataloader(val_df, dl_type="centralized", batch_size=1000)
    test_dl = create_dataloader(test_df, dl_type="centralized", batch_size=1000)
    model = MF(n_users=n_users, n_items=n_items, n_factors=n_factors, sparse=sparse)
    # optimizer = AdamW(model.parameters(), lr=1e-4)
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=mom)
    early_stopper = EarlyStopper(patience=2)
    val_losses = []
    train_losses = []
    val_loss = centralized_validate_loop(model, val_dl, optimizer)
    mlflow.log_metric("Validation RMSE", val_loss, step=0)
    print(f"Val loss: {val_loss}")
    val_losses.append(val_loss)
    for epoch in range(n_epochs):
        train_loss = centralized_train_loop(model, train_dl, optimizer)
        val_loss = centralized_validate_loop(model, val_dl, optimizer)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch: {epoch+1}, Val loss: {val_loss}")
        mlflow.log_metrics(
            {
                "Train RMSE": train_loss,
                "Validation RMSE": val_loss,
                "Epochs started": epoch+1
            },
            step=epoch + 1,
        )
        if early_stopper.early_stop(val_loss):
            print("Stopping Early.")
            break
    test_loss = centralized_validate_loop(model, test_dl, optimizer)
    print(
        f"Train losses: {train_losses} | Validation losses: {val_losses} | "
        f"Test loss: {test_loss}"
    )
    mlflow.log_metric("Test RMSE", test_loss)
    return val_losses, test_loss


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", "-d", choices=DataName.__members__.keys())
    parser.add_argument("--batch-size", "--bs", type=int)
    parser.add_argument("--n-epochs", "-n", default=5, type=int)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--mom", type=float, default=0.001)
    parser.add_argument("--n-factors", type=int, default=30)
    parser.add_argument("--sparse", type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(f"{args}")
    experiment_name = "centralized"
    exp = mlflow.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(experiment_name)
    with mlflow.start_run(experiment_id=exp_id):
        val, test = run(
            args.data_name,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            lr=args.lr,
            wd=args.weight_decay,
            mom=args.mom,
            n_factors=args.n_factors,
            sparse=args.sparse,
        )
