import argparse

import mlflow
import numpy as np
from rich import print

from src.data_utils import DataName
from src.experiments.centralized_mf import run
from src.name_utils import generate_random_name


def main(
    data_name: str,
    batch_size,
    n_epochs: int = 5,
    num_samples=1,
    n_factors=30,
):
    rng = np.random.default_rng()
    if data_name == "ML100K":
        lr_params = np.array([1e-3, 3e-2])
        # wd_params = np.array([1e-5, 1e-1])
        wd_params = np.array([1e-7, 1e-3])
        mom_params = np.array([0.01, 0.9])
    elif data_name == "HM":
        lr_params = np.array([1e-3, 2e-1])
        wd_params = np.array([1e-6, 1e-3])
        mom_params = np.array([0.01, 0.9])

    def log_uniform_sample(x):
        log_x = np.log10(x)
        return 10 ** (rng.uniform(log_x[0], log_x[1]))

    best_params = {}
    best_val_loss = np.inf

    run_name = generate_random_name()

    experiment_name = "centralized-param-search"
    exp = mlflow.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(experiment_name)
    print(f"Experiment: {experiment_name}\tRun name: {run_name}")

    for i in range(1, num_samples + 1):
        lr = log_uniform_sample(lr_params)
        wd = log_uniform_sample(wd_params)
        mom = rng.uniform(mom_params[0], mom_params[1])

        print(f"Trial: {i}\nParameters: {lr=:.04f}, {wd=:.04f}, {mom=:.04f}")
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
            try:
                val_losses, test_loss = run(
                    data_name=data_name,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    lr=lr,
                    wd=wd,
                    mom=mom,
                    n_factors=n_factors,
                )
            except ValueError as e:
                print(f"ValueError received: {e}\nStopping current run and moving to next run.")
                continue
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_params = {"lr": lr, "wd": wd, "mom": mom}
    print(f"Best val loss: {best_val_loss}\nBest paras: {best_params}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", "-d", choices=DataName.__members__.keys())
    parser.add_argument("--batch-size", "--bs", type=int)
    parser.add_argument("--n-epochs", "-n", default=5, type=int)
    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("--n-factors", default=30, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        data_name=args.data_name,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        num_samples=args.num_samples,
        n_factors=args.n_factors,
    )
