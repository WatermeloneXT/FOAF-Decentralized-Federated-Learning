import argparse
from time import perf_counter
from functools import partial
import json

import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD
import mlflow

import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from sklearn.model_selection import train_test_split

from src.models.MatrixFactorization import MF
from src.graphs import create_random_5_out_graph
from src.users import User
from src.training.decentralized import (
    decentralized_validate_loop,
    decentralized_train_loop,
)
from src.training.train_utils import EarlyStopper
from src.data_utils import (
    read_in_data,
    create_dataloader,
)

ray.init(
    _system_config={
        "object_spilling_config": json.dumps(
            {
                "type": "filesystem",
                "params": {"directory_path": "~/git/research/fed-learning/tmp/"},
            }
        )
    }
)


def train_with_ray(config, data_name, dl_type, batch_size, n_epochs):
    train_df, test_df = read_in_data(data_name=data_name)
    n_users = train_df.iloc[:, 0].nunique()
    n_items = train_df.iloc[:, 1].nunique()

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=0)
    train_data_loader = create_dataloader(
        df=train_df, dl_type=dl_type, batch_size=batch_size
    )
    val_data_loader = create_dataloader(df=val_df, dl_type="oaat")
    test_data_loader = create_dataloader(df=test_df, dl_type="oaat")

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
    graph = create_random_5_out_graph(n_users, seed=1)
    early_stopper = EarlyStopper(patience=2, min_rel_delta=0.001)

    start_time = perf_counter()
    val_losses = []
    val_loss = decentralized_validate_loop(users, val_data_loader)
    val_losses.append(val_loss)

    for epoch in range(n_epochs):
        start = perf_counter()
        avg_loss = decentralized_train_loop(
            user_models=users,
            train_loader=train_data_loader,
            graph=graph,
            progress_bar=False,
        )

        val_loss = decentralized_validate_loop(users, val_data_loader)
        val_losses.append(val_loss)

        time_elapsed = perf_counter() - start
        log_text = (
            f"Epoch {epoch + 1}\n"
            f"Train Loss: {np.sqrt(avg_loss):.04f}\n"
            f"Validation Loss: {val_loss:.04f}\n"
            f"Time Elapsed: {time_elapsed:03f} sec"
        )
        print(log_text)
        if early_stopper.early_stop(val_loss):
            print("Early stopping.")
            break
        train.report({"rmse": val_loss})
    total_time = perf_counter() - start_time
    print(f"Total time elapsed: {total_time}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", "-d", choices=["hm", "ml100k"])
    parser.add_argument(
        "--train-dl-type",
        "-t",
        choices=["oaat", "aaat", "rs"],
        default="oaat",
        help="oaat: One at a time, aaat: All at a time, rs: random sample",
    )
    parser.add_argument("--batch-size", "--bs", type=int)
    parser.add_argument("--n-epochs", "-n", default=5, type=int)
    parser.add_argument("--n-samples", default=2, type=int)
    return parser.parse_args()


def main(data_name, dl_type, batch_size, n_epochs, n_samples):
    ray_func = partial(
        train_with_ray,
        data_name=data_name,
        dl_type=dl_type,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )
    search_space = {
        "lr": tune.loguniform(1e-3, 1e-1),
        "momentum": tune.uniform(0.1, 0.9),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
    }

    # Uncomment this to enable distributed execution
    # `ray.init(address="auto")`

    tuner = tune.Tuner(
        ray_func,
        tune_config=tune.TuneConfig(
            num_samples=n_samples,
            scheduler=ASHAScheduler(metric="rmse", mode="min"),
            max_concurrent_trials=1,
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
    args = parse_arguments()
    main(
        args.data_name,
        args.train_dl_type,
        args.batch_size,
        args.n_epochs,
        args.n_samples,
    )
