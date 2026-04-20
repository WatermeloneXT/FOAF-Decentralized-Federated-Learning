import argparse
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD, Adam, SparseAdam
import mlflow
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from src.models.MatrixFactorization import MF
from src.graphs import random_k_out_graph
from src.users import User
from src.training.centralized import centralized_validate_loop, centralized_train_loop
from src.training.decentralized import (
    decentralized_validate_loop,
    decentralized_train_loop,
)
from src.data_utils import (
    read_in_data,
    create_dataloader,
)


def train_centralized_mf_with_raytune(config):
    hm, centralized = "ml100k", "centralized"
    train_df, val_df = read_in_data(data_name=hm)
    train_data_loader = create_dataloader(
        df=train_df, dl_type=centralized, batch_size=config["batch_size"]
    )
    val_data_loader = create_dataloader(df=val_df, dl_type=centralized, batch_size=1000)
    n_users = train_df.iloc[:, 0].nunique()
    n_items = train_df.iloc[:, 1].nunique()

    model = MF(n_users, n_items)
    optimizer = SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    for i in range(20):
        centralized_train_loop(
            model=model, train_loader=train_data_loader, optimizer=optimizer
        )
        val_rmse_loss = centralized_validate_loop(
            model=model, val_loader=val_data_loader, optimizer=optimizer
        )
        train.report({"rmse": val_rmse_loss})


def train_with_ray(config):
    hm, oaat = "hm", "oaat"
    train_df, val_df = read_in_data(data_name=hm)
    train_data_loader = create_dataloader(df=train_df, dl_type=oaat)
    val_data_loader = create_dataloader(df=val_df, dl_type=oaat)
    n_users = train_df.iloc[:, 0].nunique()
    n_items = train_df.iloc[:, 1].nunique()

    users = {}
    for i in range(n_users):
        model = MF(n_users, n_items, sparse=False)
        users[i] = User(
            id=i,
            model=model,
            optimizer=SGD(
                model.parameters(), lr=config["lr"], momentum=config["momentum"]
            ),
        )
    graph = random_k_out_graph(n_users, 5, 50)
    for i in range(3):
        decentralized_train_loop(users, train_data_loader, graph)
        val_rmse = decentralized_validate_loop(users, val_loader=val_data_loader)
        train.report({"rmse": val_rmse})


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", "-d", choices=["hm", "ml100k"])
    parser.add_argument(
        "--train-loader",
        "-t",
        choices=["oaat", "aaat", "rs"],
        default="oaat",
        help="oaat: One at a time, aaat: All at a time, rs: random sample",
    )
    parser.add_argument(
        "--val-loader",
        "-v",
        choices=["oaat", "aaat", "rs"],
        default="oaat",
        help="oaat: One at a time, aaat: All at a time, rs: random sample",
    )
    return parser.parse_args()


def main():
    search_space = {
        "lr": tune.loguniform(1e-3, 1e-1),
        "momentum": tune.uniform(0.1, 0.9),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([1, 2, 10, 100, 1000]),
    }

    # Uncomment this to enable distributed execution
    # `ray.init(address="auto")`

    tuner = tune.Tuner(
        train_centralized_mf_with_raytune,
        tune_config=tune.TuneConfig(
            num_samples=40,
            scheduler=ASHAScheduler(metric="rmse", mode="min"),
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    dfs = {result.path: result.metrics_dataframe for result in results}
    best_result = results.get_best_result("rmse", "min")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation rmse: {}".format(best_result.metrics["rmse"]))
    ax = None
    for d in dfs.values():
        print(d)
        ax = d["rmse"].plot(ax=ax, legend=True)
    plt.show()
    # [d["rmse"].plot() for d in dfs.values()]


if __name__ == "__main__":
    # args = parse_arguments()
    # run(args.data_name, args.train_loader, args.val_loader)
    main()
