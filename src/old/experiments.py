import argparse

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_utils import DataName, read_in_data
from src.old.data import OldDataset
from src.old.models import OldGMF, OldMF
from src.old.train import get_optimizer, random_k_out_graph, train, validate

MODEL_NAMES = ("OldMF", "OldGMF")


def get_dataloader(df):
    X, y = df.iloc[:, :2], df.iloc[:, 2]
    dataset = OldDataset(X, y)
    return DataLoader(dataset, batch_size=1, shuffle=True)


def init_models(n_users, n_items, model_name):
    model_dict = {}
    if model_name == "OldMF":
        for i in tqdm(range(n_users)):
            model_dict["model{0}".format(i)] = OldMF(n_users, n_items)
    elif model_name == "OldGMF":
        for i in tqdm(range(n_users)):
            model_dict["model{0}".format(i)] = OldGMF(n_users, n_items)
    return model_dict


def main(data_name, model_name, n_epochs):
    mlflow.log_params(
        {
            "data_name": data_name,
            "model_name": model_name,
            "max_epochs": n_epochs,
        }
    )
    train_df, test_df = read_in_data(data_name)
    n_users = train_df.iloc[:, 0].nunique()
    n_items = train_df.iloc[:, 1].nunique()
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=0)

    train_loader = get_dataloader(train_df)
    val_loader = get_dataloader(val_df)
    test_loader = get_dataloader(test_df)

    graph = random_k_out_graph(n_users, 5, 50, seed=1)

    model_dict = init_models(n_users, n_items, model_name)

    #Initialize Optimizers
    optimizer_dict = {}
    for i in tqdm(range(n_users)):
        optimizer_dict["model{0}".format(i)] = get_optimizer(model_dict["model{0}".format(i)])

    val_losses = train(
        model_dict=model_dict,
        optimizer_dict=optimizer_dict,
        train_loader=train_loader,
        val_loader=val_loader,
        graph=graph,
        epochs=n_epochs,
    )
    test_loss = validate(model_dict, test_loader)
    mlflow.log_metric("Test RMSE", test_loss)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", "-d", choices=DataName.__members__.keys())
    parser.add_argument("--n-epochs", "-n", default=1, type=int)
    parser.add_argument("--model_name", choices=MODEL_NAMES, default="OldMF")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    experiment_name = "Old Experiments"
    exp = mlflow.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(experiment_name)
    with mlflow.start_run(experiment_id=exp_id):
        main(args.data_name, args.model_name, args.n_epochs)
