from pathlib import Path
from visualization.plot_experiments_result import makespan_histogram, makespan_histogram_pd


def test_makespan_histogram():
    path_with_files = Path(
        "~/repos/rss24_sched/scheduling/exp_scripts/experiments/base_sched"
    ).expanduser()
    assert path_with_files.is_dir()

    files = list(path_with_files.glob("*.json"))

    makespan_histogram_pd(        
        extracted_files=files,
        # folder_path=path_with_files.__str__(),
        save_path=path_with_files,
    )

    makespan_histogram(
        extracted_files=files,
        # folder_path=path_with_files.__str__(),
        save_path=path_with_files.__str__(),
    )


