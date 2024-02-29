import os
import json
import zipfile
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import numpy as np


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
    color = ['royalblue', 'lightsteelblue', 'cornflowerblue']
    # color = ['royalblue', 'lightsteelblue']
    xs = 0
    ax.hist(data, n_bins, density=True, histtype='bar', color=color)
    xs = np.linspace(min(min(case) for case in data), max(max(case) for case in data), 20)

    for i, case in enumerate(data):
        density_re = gaussian_kde(case)
        density_re.covariance_factor = lambda: .50
        density_re._compute_covariance()
        ax.plot(xs, density_re(xs), "--", color=color[i])

    return ax


def makaspan_histogram(extracted_files):
    makespan = np.array([])
    makespan_same_seed = np.array([])
    makespan_different_seed = np.array([])
    for file_name in extracted_files:
        with open(file_name) as json_file:
            data = json.load(json_file)
        if 'schedule_0' in file_name:
            makespan_same_seed = np.append(makespan_same_seed, data['statistics']['makespan'][1])
        else:
            makespan_different_seed = np.append(makespan_different_seed, data['statistics']['makespan'][1])
        makespan = np.append(makespan, data['statistics']['makespan'][0])

    fig, ax = plt.subplots() #nrows=1, ncols=1, figsize=(4, 4))
    set_density(ax, [makespan, makespan_same_seed, makespan_different_seed])
    plt.legend(['Original', 'Sim. with same seed', 'Sim. with different seed'],  loc='upper left')
    plt.xlabel('Makespan [s]')
    plt.ylabel('Density')
    plt.show()


def extract_files():
    # Path to the zip file containing JSON files
    zip_file_path = 'base_sched_100.zip'
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


def get_mean_calculation_time(files):
    calculation_time = np.array([])
    calculation_time_estimation = np.array([])
    for file_name in files:
        with open(file_name) as json_file:
            data = json.load(json_file)
        calculation_time = np.append(calculation_time, [time[3] for time in data['statistics']['solver'][0]])
        calculation_time_estimation = np.append(calculation_time_estimation, [time[2] for time in data['statistics']['solver'][1]])
    print(f'Mean of general calculation time {np.mean(np.append(calculation_time, calculation_time_estimation))} \n'
          f'Mean of rescheduling calculation time {np.mean(calculation_time)} \n'
          f'Mean of rescheduling estimation calculation time {np.mean(calculation_time_estimation)}')


def rejection_and_rescheduling_correlation(files):
    rejection_time = np.empty((0, 2), float)
    rescheduling_time = np.empty((0, 2), float)
    for i, file_name in enumerate(files):
        with open(file_name) as json_file:
            data = json.load(json_file)
        for task in data['statistics']['rejection tasks']:
            rejection_time = np.append(rejection_time, [[task[1], i]], axis=0)
        for time in data['statistics']['solver'][0]:
            rescheduling_time = np.append(rescheduling_time, [[time[0], i]], axis=0)

    fig, ax = plt.subplots()  # nrows=1, ncols=1, figsize=(4, 4))
    ax.plot(rescheduling_time[:,0], rescheduling_time[:,1], 'bo', alpha=0.7)
    ax.plot(rejection_time[:,0], rejection_time[:,1], 'r+', alpha=1)
    plt.legend(['Rescheduling', 'Task rejection'], loc='upper left',  bbox_to_anchor=(0.66, 1.156), fontsize="11")
    plt.xlabel('Time [s]')
    plt.ylabel('Experiment [-]')
    plt.show()

if __name__ == '__main__':
    extracted_files = extract_files()
    # makaspan_histogram(extracted_files)

    get_mean_calculation_time(extracted_files)

    # rejection_and_rescheduling_correlation(extracted_files)
