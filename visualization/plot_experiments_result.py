import os
from pathlib import Path
import re
import json
import zipfile
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import numpy as np
import argparse

cases = [1, 2, 3, 4, 5, 6]



def set_density(ax, data, first=False):
    n_bins = 30
    # mode = ["Schedule method with soft constraints", "Schedule method without soft constraints", "Baseline method"]
    if first:
        mode = ["Prediction with constant processing time",
                "Prediction with processing time from distribution",
                "Simulation"] #, ""]
    else:
        mode = ["Prediction with constant processing time",
                "Prediction with processing time from distribution",
                "Simulation"] #, ""]

    # color = ['lightsteelblue', 'cornflowerblue', 'royalblue']
    color = ['royalblue', 'lightsteelblue', 'cornflowerblue'] if len(data) == 3 else ['royalblue', 'lightsteelblue']

    xs = 0
    ax.hist(data, n_bins, density=True, histtype='bar', color=color)
    xs = np.linspace(min(min(makespan) for makespan in data), max(max(makespan) for makespan in data), 20)

    for i, case in enumerate(data):
        density_re = gaussian_kde(case)
        density_re.covariance_factor = lambda: .50
        density_re._compute_covariance()
        ax.plot(xs, density_re(xs), "--", color=color[i])

    return ax


def makaspan_histogram(extracted_files: list[Path], all_together: bool=False, save_path: Path|None=None):
    folder_path = extracted_files[0].parent
    if all_together:
        makespan = np.array([])
        makespan_same_seed = np.array([])
        makespan_different_seed = np.array([])
        for file_name in extracted_files:
            with open(folder_path+'/'+file_name) as json_file:
                data = json.load(json_file)
            if 'schedule_0' in file_name:
                makespan_same_seed = np.append(makespan_same_seed, data['statistics']['makespan'][1])
            else:
                makespan_different_seed = np.append(makespan_different_seed, data['statistics']['makespan'][1])
            makespan = np.append(makespan, data['statistics']['makespan'][0])
        fig, ax = plt.subplots()  # nrows=1, ncols=1, figsize=(4, 4))
        set_density(ax, [makespan, makespan_same_seed, makespan_different_seed])
        plt.legend(['Original', 'Sim. with same seed', 'Sim. with different seed'], loc='upper left')

    else:
        makespan = [[] for _ in range(6)]
        makespan_same_seed = [[] for _ in range(6)]
        makespan_different_seed = [[] for _ in range(6)]
        for file_name in extracted_files:
            data = get_data_from_file(file_name)
            if not data:
                continue

            case_number, schedule_seed, _, _ = get_experiment_info(file_name.stem)
            if schedule_seed == 0:
                try:
                    makespan_same_seed[case_number-1].append(data['statistics']['makespan'][1])
                except KeyError as e:
                    print(e)
                    print(file_name)
                    continue
            else:
                makespan_different_seed[case_number-1].append(data['statistics']['makespan'][1])
            makespan[case_number-1].append(data['statistics']['makespan'][0])

        fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
        axes= [ax0, ax1, ax2, ax3, ax4, ax5]
        for case, ax in zip(cases,axes):
            set_density(ax, [makespan[case-1], makespan_different_seed[case-1]])
            ax.set_title(f'Case {case}')

        ax2.legend(['Original', 'Simulation'], loc='upper right')
    # Set common labels
    fig.supxlabel('Makespan [s]')
    fig.supylabel('Density')
    fig.tight_layout()
    if isinstance(save_path, Path):
        plt.savefig(Path.joinpath(save_path,'makaspan_histogram.png'))
    plt.show()


def get_data_from_file(file_name: Path):
    try:
        with open(file_name) as json_file:
            data = json.load(json_file)
    except json.decoder.JSONDecodeError as e:
        print(e)
        print(file_name)
        return None
    return data


def get_experiment_info(file_stem: str):
    # Use regular expression to extract the number of the case
    case_number = int(re.search(r'case_(\d+)', file_stem).group(1))
    schedule_seed = int(re.search(r'schedule_(\d+)', file_stem).group(1))
    dist_seed = int(re.search(r'dist_(\d+)', file_stem).group(1))
    sim_seed = int(re.search(r'sim_(\d+)', file_stem).group(1))
    return case_number, schedule_seed, dist_seed, sim_seed


def extract_files():
    # Path to the zip file containing JSON files
    zip_file_path = 'base_sched.zip'
    extracted_dir = 'extracted_json_files'

    # Create a directory to extract the files
    os.makedirs(extracted_dir, exist_ok=True)

    # Extract JSON files from the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)

    # Get a list of extracted JSON file paths
    extracted_files = [os.path.join(extracted_dir, '', file) for file in zip_ref.namelist()]
    extracted_files = [file for file in extracted_files if '.json' in file and '.ipynb' not in file]
    return extracted_files


def get_mean_calculation_time(files: list[Path], save_path: Path|None=None):
    simulation_time = [[] for _ in range(6)]
    decision_making_duration = [[] for _ in range(6)]
    solving_time = [[] for _ in range(6)]
    for file_name in files:
        data = get_data_from_file(file_name)
        if not data:
            continue

        case_number, schedule_seed, dist_seed, sim_seed = get_experiment_info(file_name.stem)
        simulation_time[case_number-1].append(data['statistics']['sim_time'])
        decision_making_duration[case_number-1].append(np.mean(data['statistics']['decision_making_duration']))
        solving = [time[3] for time in data['statistics']['solver'][0]]
        evaluation = [time[2] for time in data['statistics']['solver'][1]]
        solving_time[case_number-1].append(np.mean(solving+evaluation))

    output = {}
    for case in cases:
        output[case] = {'sim_mean': np.mean(simulation_time[case-1]),
                        'decision_mean': np.mean(decision_making_duration[case-1]),
                        'solving_time': np.mean(solving_time[case-1])}

    with open(Path.joinpath(save_path, 'statistics.json'), 'w') as f:
        json.dump(output, f, indent=4)


def rejection_and_rescheduling_correlation(files):
    rejection_time = np.empty((0, 2), float)
    rescheduling_time = np.empty((0, 2), float)
    for i, file_name in enumerate(files):
        data = get_data_from_file(file_name)
        if not data:
            continue

        for task in data['statistics']['rejection tasks']:
            rejection_time = np.append(rejection_time, [[task[1], i]], axis=0)
        for time in data['statistics']['solver'][0]:
            rescheduling_time = np.append(rescheduling_time, [[time[0], i]], axis=0)

    fig, ax = plt.subplots()  # nrows=1, ncols=1, figsize=(4, 4))
    ax.plot(rescheduling_time[:, 0], rescheduling_time[:, 1], 'bo', alpha=0.7)
    ax.plot(rejection_time[:, 0], rejection_time[:, 1], 'r+', alpha=1)
    plt.legend(['Rescheduling', 'Task rejection'], loc='upper left',  bbox_to_anchor=(0.66, 1.156), fontsize="11")
    plt.xlabel('Time [s]')
    plt.ylabel('Experiment [-]')
    plt.show()


if __name__ == '__main__':
    folder_path = 'extracted_json_files/base_sched/'
    save_path = ''

    parser = argparse.ArgumentParser()
    parser.add_argument('--from_zip', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.from_zip:
        extracted_files = extract_files()
    else:
        extracted_files = os.listdir(folder_path)
        extracted_files = [Path(folder_path+file) for file in extracted_files if '.json' in file and '.ipynb' not in file]

    makaspan_histogram(extracted_files, save_path=Path(save_path))

    # get_mean_calculation_time(extracted_files, save_path=Path(''))

    # rejection_and_rescheduling_correlation(extracted_files)
