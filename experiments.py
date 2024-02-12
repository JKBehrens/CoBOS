"""
This file runs the simulation several time with different parameters and save statistics.
The parameters of the experiments can be set in the experiment_config.json file.
@author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
@contact: marina.ionova@cvut.cz
"""
import argparse
import logging
from datetime import datetime

from control.control_logic import ControlLogic
from simulation import sim_param_path as sim_config_path
import json


def run_test_with_config(params):

    for case in params["cases"]:
        for fail in params["fail_probability"]:
            weights = params["weights_1"] if case in ['0', '1', '2'] else params["weights_2"]
            for human_chose in params["human_chose_prob"]:
                DATA_FILE_NAME = f"case_{case}_fail_{fail[0]}_chose_{human_chose[0]}.json"
                STATISTICS_FILE_NAME = f"case_{case}_fail_{fail[0]}_chose_{human_chose[0]}.json"

                data = checking_file_existence(params["DATA_PATH"] + DATA_FILE_NAME)
                statistics = checking_file_existence(params["STATISTICS_PATH"] + STATISTICS_FILE_NAME)

                for weight in weights:
                    # continue experiments from the last
                    if str(weight[2]) in statistics.keys():
                        last_experiment = len(statistics[str(weight[2])])
                    else:
                        data[str(weight[2])] = []
                        statistics[str(weight[2])] = []
                        last_experiment = 0
                    logging.warning(f'last_experiment={last_experiment}')
                    iteration_name = f"case_{case} fail_prob {params['fail_probability'].index(fail) + 1}/{len(params['fail_probability'])}, " \
                                     f"chose prob {params['human_chose_prob'].index(human_chose) + 1}/{len(params['human_chose_prob'])}, " \
                                     f"weights {weights.index(weight) + 1}/{len(weights)}"

                    with open(sim_config_path, 'r+') as sim_config_file:
                        sim_params = json.load(sim_config_file)
                        sim_params["Allocation weights"] = weight
                        sim_params["Human chose prob"] = human_chose
                        sim_params["Fail probability"] = fail
                        sim_config_file.seek(0)
                        json.dump(sim_params, sim_config_file, indent=4)
                        sim_config_file.truncate()

                    for i in range(last_experiment, params["experiment_number"], 1):
                        with open(sim_config_path, 'r+') as sim_config_file:
                            sim_params = json.load(sim_config_file)
                            sim_params["Seed"] = i
                            sim_config_file.seek(0)
                            json.dump(sim_params, sim_config_file, indent=4)
                            sim_config_file.truncate()
                        if i % 10 == 0:
                            logging.warning(
                                f"Time: {datetime.now()} {iteration_name}, iter{i}/{params['experiment_number']}")

                        execute_job = ControlLogic(case)
                        schedule, stat = execute_job.run(experiments=True)
                        if schedule:
                            data[str(weight[2])].append(schedule)
                            statistics[str(weight[2])].append(stat)
                        else:
                            logging.error(f'SIM PARAM: seed {i}, human_chose {human_chose},'
                                          f'weight {weight}, fail {fail}')
                            logging.error('Scheduling failed. Next measurement.')

                    with open(params["DATA_PATH"] + DATA_FILE_NAME, 'w') as outfile:
                        json.dump(data, outfile)
                    with open(params["STATISTICS_PATH"] + STATISTICS_FILE_NAME, 'w') as outfile:
                        json.dump(statistics, outfile)


def run_test_with_random():
    PATH = 'new_experiments/same_distribution_same_seed.json'
    output = []
    for case in range(1, 6):
        for j in range(5): # distrib seed for job generation
            for i in range(5): # seed for scheduling and simulation
                execute_job = ControlLogic(str(case), distribution_seed=j, sim_seed=i, schedule_seed=i)
                schedule, stat = execute_job.run(experiments=True)
                if schedule:
                    output.append([{'case': case, 'distribution_seed': j, 'sim_seed': i, 'schedule_seed': i}, stat])
                    logging.warning(f'Iteration case: {case}, {j}/5, {i}/5')
                else:
                    logging.error(f'SIM PARAM: seed {i}')
                    logging.error('Scheduling failed. Next measurement.')
    with open(PATH, 'w') as outfile:
        json.dump(output, outfile)


def checking_file_existence(path):
    try:
        with open(path, "r+") as json_file:
            content = json.load(json_file)
            logging.warning(f"File with PATH={path} is already exist. Loading content form file...")
    except FileNotFoundError:
        content = {}
        logging.warning(f"File with PATH={path} does not exist.")
    return content


def get_parameters():
    with open('./experiments_config.json', 'r') as f:
        parameters = json.load(f)
    return parameters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    lvl = logging.WARNING
    logging.basicConfig(level=lvl,
                        format=f"%(levelname)-8s: %(filename)s %(funcName)s %(lineno)s - %(message)s")
    logging.getLogger("mylogger")

    if args.config:
        params = get_parameters()
        run_test_with_config(params)
    else:
        run_test_with_random()
