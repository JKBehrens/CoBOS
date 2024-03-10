from pathlib import Path
from visualization.plot_experiments_result import makespan_histogram_pd
from exp_scripts.run_base_scheduling_exps import ExperimentSettings, start_cluster, run_exps
from dask.distributed import performance_report


def test_makespan_histogram(tmp_path: Path):
    exp_settings = ExperimentSettings(dist_seed=2, schedule_seed=10, exp_folder=tmp_path)
    try:
        client = start_cluster(exp_settings)
        run_exps(client=client, exp_settings=exp_settings)
    finally:
        client.shutdown()

    path_with_files = exp_settings.exp_folder
    assert path_with_files.is_dir()

    files = list(path_with_files.glob("*.json"))

    makespan_histogram_pd(
        extracted_files=files,
        # folder_path=path_with_files.__str__(),
        save_path=path_with_files,
    )
    #
    # makespan_histogram(
    #     extracted_files=files,
    #     # folder_path=path_with_files.__str__(),
    #     save_path=path_with_files.__str__(),
    # )


