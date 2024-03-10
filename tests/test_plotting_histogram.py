from pathlib import Path
from visualization.plot_experiments_result import makespan_histogram_pd
from exp_scripts.run_base_scheduling_exps import ExperimentSettings, start_cluster, run_exps


def test_makespan_histogram():
    exp_settings = ExperimentSettings(dist_seed=2, schedule_seed=10)
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


