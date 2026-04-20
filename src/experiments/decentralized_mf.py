import argparse

import mlflow
import torch
from rich import print
from sklearn.model_selection import train_test_split
from torch.optim import SGD
from tqdm import tqdm

from src.data_utils import DL_TYPES, DataName, create_dataloader, read_in_data
from src.graphs import GRAPH_TYPES, create_graph, create_random_5_out_graph
from src.models.configs import MODEL_NAMES
from src.models.GeneralizedMF import (GeneralizedDeepMF, GeneralizedMF,
                                      GeneralizedMFSharedItemLayer,
                                      NonlinearGMF)
from src.models.MatrixFactorization import MF, UMF
from src.models.simple import GlobalAverage
from src.training.decentralized import (decentralized_train_n_epochs,
                                        decentralized_validate_loop)
from src.users import User


def run(
    data_name: str,
    train_loader_type: str,
    val_loader_type: str,
    batch_size,
    n_epochs: int = 5,
    lr=0.01,
    weight_decay=0.001,
    mom=0,
    graph_seed=1,
    n_factors=30,
    sparse=False,
    model="mf",
    userprop=None,
    graph_type="random_5_out",
    order=1,
):
    mlflow.log_params(
        {
            "train_dl": train_loader_type,
            "val_dl": val_loader_type,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "lr": lr,
            "wd": weight_decay,
            "mom": mom,
            "n_factors": n_factors,
            "sparse": sparse,
            "model": model,
            "userprop": userprop,
            "graph_type": graph_type,
            "order": order,
        }
    )
    train_df, test_df = read_in_data(data_name=data_name)
    n_users = train_df.iloc[:, 0].nunique()
    n_items = train_df.iloc[:, 1].nunique()

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=0)
    train_data_loader = create_dataloader(
        df=train_df, dl_type=train_loader_type, batch_size=batch_size, p=userprop
    )
    val_data_loader = create_dataloader(df=val_df, dl_type=val_loader_type)
    test_data_loader = create_dataloader(df=test_df, dl_type=val_loader_type)

    users = {}
    for i in tqdm(range(n_users)):
        if model == "umf":
            user_model = UMF(n_items, n_factors=n_factors, sparse=sparse)
        elif model == "gmf":
            layer_size = 5
            user_model = GeneralizedMF(n_items, n_factors=n_factors, layer_size=layer_size)
            mlflow.log_param("layer_size", layer_size)
        elif model == "gmf_shared":
            layer_size = 5
            user_model = GeneralizedMFSharedItemLayer(n_items, n_factors, layer_size=layer_size)
            mlflow.log_param("layer_size", layer_size)
        elif model == "gmf_deep":
            layer_size = 5
            n_layers = 2
            user_model = GeneralizedDeepMF(
                n_items, n_factors=n_factors, layer_size=layer_size, n_layers=n_layers
            )
            mlflow.log_params({"layer_size": layer_size, "n_layers": n_layers})
        elif model == "ngmf":
            layer_size = 5
            n_layers = 1
            user_model = NonlinearGMF(n_items, n_factors, layer_size, n_layers)
            mlflow.log_params({"layer_size": layer_size, "n_layers": n_layers})
        elif model == "globalavg":
            user_model = GlobalAverage(torch.tensor(train_df.to_numpy()))
        else:
            user_model = MF(n_users, n_items, n_factors=n_factors, sparse=sparse)
        users[i] = User(
            id=i,
            model=user_model,
            optimizer=SGD(user_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=mom),
            model_name=model,
        )
    graph = create_graph(graph_type=graph_type, n_users=n_users, seed=graph_seed, order=order)
    # graph = create_random_5_out_graph(n_users=n_users, seed=graph_seed)
    train_losses, val_losses, time_per_epoch = decentralized_train_n_epochs(
        user_models=users,
        train_loader=train_data_loader,
        val_loader=val_data_loader,
        epochs=n_epochs,
        graph=graph,
    )

    test_loss = decentralized_validate_loop(users, test_data_loader)

    print(
        f"Train losses: {train_losses} | Validation losses: {val_losses} | "
        f"Time per epoch: {time_per_epoch} | Test loss: {test_loss}"
    )
    return train_losses, val_losses, test_loss, time_per_epoch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", "-d", choices=DataName.__members__.keys())
    parser.add_argument(
        "--train-loader",
        "-t",
        choices=DL_TYPES,
        default="oaat",
        help="oaat: One at a time, aaat: All at a time, rs: random sample",
    )
    parser.add_argument(
        "--val-loader",
        "-v",
        choices=DL_TYPES,
        default="oaat",
        help="oaat: One at a time, aaat: All at a time, rs: random sample",
    )
    parser.add_argument("--batch-size", "--bs", type=int)
    parser.add_argument("--n-epochs", "-n", default=5, type=int)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--mom", type=float, default=0.001)
    parser.add_argument("--graph-seed", type=int, default=1)
    parser.add_argument("--n-factors", type=int, default=30)
    parser.add_argument("--sparse", type=bool, default=False)
    parser.add_argument("--model", choices=MODEL_NAMES, default="mf")
    parser.add_argument("--userprop", default=None, type=float)
    parser.add_argument("--graph-type", default="random_5_out", choices=GRAPH_TYPES)
    parser.add_argument("--order", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(f"{args=}")
    experiment_name = f"{args.data_name}-graph{args.graph_seed}"
    exp = mlflow.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(experiment_name)
    with mlflow.start_run(experiment_id=exp_id):
        train, val, test, times = run(
            args.data_name,
            args.train_loader,
            args.val_loader,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            mom=args.mom,
            graph_seed=args.graph_seed,
            n_factors=args.n_factors,
            sparse=args.sparse,
            model=args.model,
            userprop=args.userprop,
            graph_type=args.graph_type,
            order=args.order,
        )
        mlflow.log_metrics(
            {
                "Test RMSE": test,
            }
        )
