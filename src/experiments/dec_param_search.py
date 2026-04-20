import argparse

from rich import print
import numpy as np
from dotenv import dotenv_values
import mlflow

from src.experiments.decentralized_mf import run
from src.data_utils import DL_TYPES, DataName
from src.graphs import GRAPH_TYPES
from src.name_utils import generate_random_name
from src.models.configs import MODEL_NAMES


def main(
    data_name: str,
    train_loader_type: str,
    val_loader_type: str,
    batch_size,
    n_epochs: int = 5,
    num_samples=1,
    graph_seed=1,
    userprop=None,
    model="",
    n_factors=30,
    graph_type="",
):
    rng = np.random.default_rng()
    if model == "gmf":
        lr_params = np.array([1e-5, 1e-2])
        wd_params = np.array([1e-5, 1e-1])
        mom_params = np.array([0.01, 0.9])
    elif model == "umf" and data_name == "ML100K":
        lr_params = np.array([1e-3, 5e-1])
        wd_params = np.array([1e-3, 1])
        mom_params = np.array([0.3, 0.95])

    else:
        lr_params = np.array([1e-3, 5e-1])
        wd_params = np.array([1e-5, 1e-1])
        mom_params = np.array([0.5, 0.95])

    def log_uniform_sample(x):
        log_x = np.log10(x)
        return 10 ** (rng.uniform(log_x[0], log_x[1]))

    best_params = {}
    best_val_loss = np.inf

    run_name = generate_random_name()

    experiment_name = f"param-search-{train_loader_type}-{data_name}-graph{graph_seed}"
    exp = mlflow.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(experiment_name)
    print(f"Experiment: {experiment_name}\tRun name: {run_name}")
    print(f"Ranges | lr: {lr_params} | wd: {wd_params} | mom: {mom_params}")

    for i in range(1, num_samples + 1):
        lr = log_uniform_sample(lr_params)
        wd = log_uniform_sample(wd_params)
        mom = rng.uniform(mom_params[0], mom_params[1])

        print(f"Trial: {i}\nParameters: {lr=:.04f}, {wd=:.04f}, {mom=:.04f}")
        tags = {"lr_range": lr_params, "wd_range": wd_params, "mom_range": mom_params}
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags=tags):
            try:
                train_losses, val_losses, test_loss, time_per_epoch = run(
                    data_name=data_name,
                    train_loader_type=train_loader_type,
                    val_loader_type=val_loader_type,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    lr=lr,
                    weight_decay=wd,
                    mom=mom,
                    graph_seed=graph_seed,
                    userprop=userprop,
                    model=model,
                    n_factors=n_factors,
                    graph_type=graph_type,
                )
                mlflow.log_metric("Test RMSE", test_loss)
            except ValueError as e:
                print(
                    f"ValueError received: {e}\nStopping current run and moving to next run."
                )
                continue
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_params = {"lr": lr, "wd": wd, "mom": mom}
    print(f"Best val loss: {best_val_loss}\nBest paras: {best_params}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", "-d", choices=DataName.__members__.keys())
    parser.add_argument(
        "--train-loader",
        "-t",
        choices=DL_TYPES,
        default="oaat",
        help="oaat: One at a time, aaat: All at a time, rs: random sample, centralized: single model, urs: user stratified sample",
    )
    parser.add_argument(
        "--val-loader",
        "-v",
        choices=DL_TYPES,
        default="oaat",
        help="oaat: One at a time (recommended), aaat: All at a time, rs: random sample, centralized: single model, urs: user stratified sample",
    )
    parser.add_argument("--batch-size", "--bs", type=int)
    parser.add_argument("--n-epochs", "-n", default=5, type=int)
    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("--userprop", default=None, type=float)
    parser.add_argument("--model", default="umf", choices=MODEL_NAMES)
    parser.add_argument("--n-factors", default=30, type=int)
    parser.add_argument("--graph-type", default="random_5_out", choices=GRAPH_TYPES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        args.data_name,
        args.train_loader,
        args.val_loader,
        args.batch_size,
        args.n_epochs,
        args.num_samples,
        userprop=args.userprop,
        model=args.model,
        n_factors=args.n_factors,
        graph_type=args.graph_type,
    )
