from pathlib import Path

import pytest

from visualization.plot_experiments_result import makespan_histogram_pd, read_data_to_df
from exp_scripts.run_base_scheduling_exps import (
    ExperimentSettings,
    start_cluster,
    run_exps,
)
import pandas as pd


def test_makespan_histogram_csv(tmp_path: Path):
    try:
        df: pd.DataFrame = pd.read_parquet("tests/data/test_data.parquet")
    except FileNotFoundError:
        df: pd.DataFrame = pd.read_parquet("data/test_data.parquet")

    makespan_histogram_pd(
        df=df,
        save_path=tmp_path,
    )

    assert len(list(tmp_path.glob("*.png"))) >= 1


@pytest.mark.skip("takes too long")
def test_makespan_histogram(tmp_path: Path):
    exp_settings = ExperimentSettings(dist_seed=2, schedule_seed=3, exp_folder=tmp_path)
    try:
        client = start_cluster(exp_settings)
        run_exps(client=client, exp_settings=exp_settings)
    finally:
        client.shutdown()

    path_with_files = exp_settings.exp_folder
    assert path_with_files.is_dir()

    files = list(path_with_files.glob("*.json"))

    df = read_data_to_df(
        files, {"initial_makespan": -1, "final_makespan": -1, "FAIL": False}
    )

    df.to_parquet(dump_file := tmp_path.joinpath("test_data.parquet"))
    df2 = pd.read_parquet(dump_file)

    assert len(df.get("initial_makespan")) == len(df2.get("initial_makespan"))

    # for c1, c2 in zip(df.get("initial_makespan"), df2.get("initial_makespan")):
    #     assert len(c1) == len(c2)
    #     for i, _ in enumerate(c1):
    #         if c1[i] is None or c1[i] is np.nan:
    #             assert c2[i] is None or np.isnan(c2[i])
    #         else:
    #             assert c1[i] == c2[i]

    makespan_histogram_pd(
        df=df,
        save_path=path_with_files,
    )

    assert len(list(path_with_files.glob("*.png"))) >= 1
