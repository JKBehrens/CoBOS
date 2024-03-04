import copy

from methods.scheduling_split_tasks import Schedule, schedule_as_dict
from methods import OverlapSchedule, NoOverlapSchedule, RandomAllocation, MaxDuration
from control.control_logic import ControlLogic
from control.agents import Agent
from visualization.json_2_video import video_parser
from visualization import schedule_save_file_name, Vis, comparison_save_file_name
from control.jobs import Job
import numpy as np
import argparse
import logging
import json

METHOD = OverlapSchedule
case = 4
solver_seed = 4
dist_seed = 7
sim_seed = 3


if __name__ == '__main__':
    cases = ['1', '2', '3', '4', '5', '6', '0']

    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=str, help='Choose one of this: 1, 2, 3, 4, 5, 6')
    parser.add_argument("file_name", nargs="?", help="File name (required if -f is set)")
    parser.add_argument('--only_schedule', action=argparse.BooleanOptionalAction)
    parser.add_argument('--change_job', action=argparse.BooleanOptionalAction)
    parser.add_argument('--offline', action=argparse.BooleanOptionalAction)
    parser.add_argument('--log_error', action=argparse.BooleanOptionalAction)
    parser.add_argument('--log_debug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--comparison', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    if args.log_error:
        lvl = logging.ERROR
    elif args.log_debug:
        lvl = logging.DEBUG
    else:
        lvl = logging.INFO
    logging.basicConfig(level=lvl,
                        format=f"%(levelname)-8s: - %(message)s")
    logging.getLogger("mylogger")

    if args.case in cases:
        case = int(args.case)
    else:
        logging.error("The case does not exist")
        raise SystemExit(1)

    job = Job(case, seed=dist_seed)

    agent_names = ["Human", "Robot"]
    agents: list[Agent] = []
    for agent_name in agent_names:
        agents.append(
            Agent(
                name=agent_name,
                job=copy.deepcopy(job),
                seed=sim_seed,
                answer_seed=sim_seed,
            )
        )

    solving_method = METHOD(job=job, seed=solver_seed)
    solving_method.prepare()

    if args.comparison:
        job = Job(case, seed=0)
        if args.change_job:
            execute_job = ControlLogic(method=METHOD, case=case, job=copy.deepcopy(job), schedule_seed=0, sim_seed=0, answer_seed=None)
            schedule1, stat1 = execute_job.run(animation=False, experiments=True)
            seed = 9
            nameList = [True, False]
            rand = np.random.default_rng(seed=seed)
            answers = []
            for task in job.task_sequence:
                answer = rand.choice(nameList, p=(1 - task.rejection_prob, task.rejection_prob), size=1)
                answers.append(answer)
                if not answer and task.universal:
                    task.universal = False
                    task.agent = 'Robot'
            execute_job = ControlLogic(method=METHOD, case=case, job=copy.deepcopy(job), schedule_seed=0, sim_seed=0,  answer_seed=seed)
            schedule2, stat2 = execute_job.run(animation=False, experiments=True)
        else:
            execute_job = ControlLogic(method=OverlapSchedule, case=case, job=copy.deepcopy(job), schedule_seed=0, sim_seed=0)
            schedule1, stat1 = execute_job.run(animation=False, experiments=True)
            execute_job = ControlLogic(method=MaxDuration, case=case, job=copy.deepcopy(job), schedule_seed=0, sim_seed=0)
            schedule2, stat2 = execute_job.run(animation=False, experiments=True)

        file_name = args.file_name if args.file_name else comparison_save_file_name

        # save original schedule, simulation with the same seed as in schedule, simulation with the another seed
        # statistics is saved for each simulation:
        # - predicted makespan, final makespan,
        # - list of the rejection tasks with time, when it was rejected
        # - list with the rescheduling info: when, status of solver, makespan, calculation time
        # - list with the rescheduling evaluation info: when, makespan, calculation time

        try:
            with open(file_name, "w") as outfile:
                json.dump({'schedule': schedule1+[schedule2[1]], 'statistics': [stat1, stat2]}, outfile)
                logging.info(f'Save data to {file_name}')
        except IndexError:
            with open(file_name, "w") as outfile:
                json.dump({'schedule': schedule1+schedule2, 'statistics': [stat1, stat2]}, outfile)
                logging.info(f'Save data to {file_name}')

    elif not args.only_schedule:
        control_logic = ControlLogic(job=job, agents=agents, method=solving_method)
        if args.offline:
            schedule, statistics = control_logic.run(experiments=True, save2file=True)
        else:
            schedule, statistics = control_logic.run(experiments=True)

    else:
        job = Job(case, seed=0)
        schedule_model = OverlapSchedule(job, seed=0)
        output = schedule_model.prepare()
        with open(schedule_save_file_name, "w") as outfile:
            json.dump(schedule_as_dict(output), outfile)
            logging.info(f'Save data to {schedule_save_file_name}')
        save_file_name = 'schedule.png'

        gantt = Vis(data=schedule_as_dict(output), from_file=True)
        gantt.plot_schedule(save_file_name)
        logging.info(f'Save picture to ./img/{save_file_name}')



