from pathlib import Path
from visualization.plot_experiments_result import makaspan_histogram


def test_makespan_hustogram():
    path_with_files = Path(
        "~/repos/rss24_sched/scheduling/exp_scripts/experiments/base_sched"
    ).expanduser()
    assert path_with_files.is_dir()

    files = list(path_with_files.glob("*.json"))

    makaspan_histogram(
        extracted_files=files,
        # folder_path=path_with_files.__str__(),
        save_path=path_with_files.__str__(),
    )


