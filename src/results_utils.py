import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from mlflow.entities import ViewType
from typing_extensions import Self

HM_EXPERIMENT_ID = "481672132388513185"
ML100K_EXPERIMENT_ID = "894020621187641387"


COL_NAME_MAP = {
    "metrics.Train RMSE": "Train RMSE",
    "metrics.Validation RMSE": "Validation RMSE",
    "metrics.Test RMSE": "Test RMSE",
    "metrics.epochs_started": "epochs_started",
    "params.train_dl": "train_dl",
    "params.lr": "lr",
    "params.userprop": "p",
    "params.wd": "wd",
    "params.n_epochs": "n_epochs",
    "params.mom": "mom",
    "params.batch_size": "batch_size",
    "params.n_factors": "n_factors",
    "params.layer_size": "layer_size",
    "tags.mlflow.runName": "run_name",
    "Time taken": "Time (s)",
}
LATEX_COL_NAME_MAP = {
    "train_dl": "Train Type",
    "batch_size": "Batch",
    "lr": "LR",
    "wd": "WD",
    "mom": "MOM",
    "n_factors": "k",
    "layer_size": "Layer Size",
    "epochs_started": "Epochs",
    "n_epochs": "Max Epochs",
    "Time (s)": "Time (s)",
    "Validation RMSE": "Val",
    "Test RMSE": "Test",
}

PS_COL_NAME_MAP = {key: val for key, val in COL_NAME_MAP.items() if key != "metrics.Test RMSE"}

PS_LATEX_COL_NAME_MAP = {
    key: val for key, val in LATEX_COL_NAME_MAP.items() if key not in ["Test RMSE", "n_epochs"]
}

VALIDATION_PLOT_COL_NAME_MAP = {**COL_NAME_MAP, "run_id": "run_id"}

FLOAT_COLUMNS = ["Train RMSE", "Validation RMSE", "Test RMSE", "lr", "p", "wd", "mom", "Time (s)"]
INTLIKE_COLUMNS = ["n_epochs", "batch_size", "layer_size"]
NON_NUMERICAL_COLS = ["train_dl", "run_name"]


def mlflow_search_runs(experiment_ids, filter_string: str):
    results = mlflow.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    results["Time taken"] = results["end_time"] - results["start_time"]
    return results


class ResultFormatter:
    def __init__(self, experiment_ids, filter_string: str = "", run_names = None):
        df = mlflow_search_runs(experiment_ids, filter_string)
        if run_names:
            df = df[df["tags.mlflow.runName"].isin(run_names)]
        self.og_results: pd.DataFrame = df
        self.results: pd.DataFrame = self.og_results.copy()

    def filter_cols(self, superset_of_col_names) -> Self:
        shared_cols = [col for col in self.results.columns if col in superset_of_col_names]
        self.results = self.results[shared_cols]
        return self

    def rename_cols(self, col_map) -> Self:
        self.results = self.results.rename(columns=col_map)
        return self

    def format_batch_size(self, contains_userprop: bool) -> Self:
        if "p" in self.results.columns:
        # if contains_userprop:
            p_is_not_none = ~(self.results["p"].isna() | (self.results["p"] == "None"))
            self.results.loc[p_is_not_none, "batch_size"] = self.results.loc[p_is_not_none, "p"]
            self.results = self.results.drop(columns=["p"])
        if "train_dl" in self.results.columns:
            self.results.loc[self.results["train_dl"] == "oaat", "batch_size"] = 1
        return self

    def format_epochs(self, col_name="epochs_started") -> Self:
        if col_name in self.results.columns:
            self.results[col_name] = self.results[col_name].astype(str).str.split(".").str[0]
        return self

    # def format_n_epochs(self):
    #     if "epochs_started" not in

    def convert_time_to_seconds(self, col_name="Time (s)") -> Self:
        if col_name in self.results.columns:
            self.results[col_name] = self.results[col_name].dt.total_seconds()
        return self

    def convert_float_columns(self, float_columns=FLOAT_COLUMNS) -> Self:
        shared_float_cols = [col for col in self.results.columns if col in float_columns]
        self.results[shared_float_cols] = self.results[shared_float_cols].apply(
            pd.to_numeric, axis=1
        )
        return self

    def print_latex_table(
        self, col_name_map, drop_cols=["run_name"], num_decimal_places: int = 3
    ) -> None:
        latex_df = (
            self.results.sort_values("Validation RMSE")
            .rename(columns=col_name_map)
            .drop(columns=drop_cols)
        )
        # print(latex_df.to_latex(float_format=f"%.{num_decimal_places}f", index=False))
        print(
            latex_df[[col for col in col_name_map.values() if col in latex_df.columns]].to_latex(
                float_format=f"%.{num_decimal_places}f", index=False
            )
        )

    def format_and_print_latex_table(
        self, col_name_map, latex_col_name_map, contains_userprop=True
    ) -> None:
        (
            self.filter_cols(col_name_map)
            .rename_cols(col_name_map)
            .format_batch_size(contains_userprop=contains_userprop)
            .format_epochs()
            .convert_time_to_seconds()
            .convert_float_columns()
            .print_latex_table(latex_col_name_map)
        )


class FuncResultFormatter:
    def __init__(self, experiment_ids, filter_string: str = ""):
        self.results: pd.DataFrame = mlflow_search_runs(experiment_ids, filter_string)

    def filter_cols(self, superset_of_col_names) -> pd.DataFrame:
        shared_cols = [col for col in self.results.columns if col in superset_of_col_names]
        return self.results[shared_cols]

    def rename_cols(self, col_map) -> Self:
        return self.results.rename(columns=col_map)

    def format_batch_size(self, contains_userprop: bool) -> Self:
        if contains_userprop:
            p_is_not_none = ~(self.results["p"].isna() | (self.results["p"] == "None"))
            self.results.loc[p_is_not_none, "batch_size"] = self.results.loc[p_is_not_none, "p"]
            self.results = self.results.drop(columns=["p"])
        self.results.loc[self.results["train_dl"] == "oaat", "batch_size"] = 1
        return self

    def format_epochs(self, col_name="epochs_started") -> Self:
        self.results[col_name] = self.results[col_name].astype(str).str.split(".").str[0]
        return self

    # def format_n_epochs(self):
    #     if "epochs_started" not in

    def convert_time_to_seconds(self, col_name="Time (s)") -> Self:
        self.results[col_name] = self.results[col_name].dt.total_seconds()
        return self

    def convert_float_columns(self, float_columns=FLOAT_COLUMNS) -> Self:
        shared_float_cols = [col for col in self.results.columns if col in float_columns]
        self.results[shared_float_cols] = self.results[shared_float_cols].apply(
            pd.to_numeric, axis=1
        )
        return self

    def print_latex_table(
        self, col_name_map, drop_cols=["run_name"], num_decimal_places: int = 3
    ) -> None:
        latex_df = (
            self.results.sort_values("Validation RMSE")
            .rename(columns=col_name_map)
            .drop(columns=drop_cols)
        )
        # print(latex_df.to_latex(float_format=f"%.{num_decimal_places}f", index=False))
        print(
            latex_df[[col for col in col_name_map.values() if col in latex_df.columns]].to_latex(
                float_format=f"%.{num_decimal_places}f", index=False
            )
        )

    def format_to_latex_table(self, col_name_map, latex_col_name_map) -> None:
        (
            self.filter_cols(col_name_map)
            .rename_cols(col_name_map)
            .format_batch_size(contains_userprop=True)
            .format_epochs()
            .convert_time_to_seconds()
            .convert_float_columns()
            .print_latex_table(latex_col_name_map)
        )


def format_results(results, col_name_map, includes_userprop=False):
    shared_cols = [col for col in results.columns if col in col_name_map.keys()]
    df = results[shared_cols].rename(columns=col_name_map)
    if includes_userprop:
        p_is_not_none = ~(df["p"].isna() | (df["p"] == "None"))
        df.loc[p_is_not_none, "batch_size"] = df.loc[p_is_not_none, "p"]
        df = df.drop(columns=["p"])
    df.loc[df["train_dl"] == "oaat", "batch_size"] = 1
    df["Time (s)"] = df["Time (s)"].dt.total_seconds()
    print(df)
    float_cols = ["lr", "wd", "mom", "Time (s)", "Validation RMSE", "Test RMSE"]
    df[float_cols] = df[float_cols].apply(pd.to_numeric, axis=1)
    return df


def print_latex_table(df, col_name_map):
    print(
        df.sort_values("Validation RMSE")
        .rename(columns=col_name_map)
        .drop(columns=["run_name"])
        .to_latex(float_format="%.3f", index=False)
    )


def get_validation_rmse_data(df):
    df = df.copy()
    df["val_rmse_history"] = df.run_id.apply(client.get_metric_history, key="Validation RMSE")
    df = df[["train_dl", "val_rmse_history"]].explode("val_rmse_history")
    df["timestamp"] = df.val_rmse_history.apply(lambda x: x.timestamp)
    df["RMSE"] = df.val_rmse_history.apply(lambda x: x.value)
    df["Seconds"] = df.groupby(df.index)["timestamp"].transform(lambda x: (x - x.min()) / 1000)
    df = df.rename(columns={"train_dl": "Train Type"})
    return df
