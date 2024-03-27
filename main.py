import copy
from methods import OverlapSchedule, RandomAllocation, MaxDuration, DynamicAllocation
from control.control_logic import ControlLogic
from control.agents import Agent
from inputs import RandomCase
from visualization import Vis, comparison_save_file_name
from control.jobs import Job
from pathlib import Path

import numpy as np
import argparse
import logging
import json

METHOD = OverlapSchedule    # RandomAllocation, MaxDuration, DynamicAllocation
solver_seed = 42
dist_seed = 7
sim_seed = 42
answer_seed = 42
det_job = True

if __name__ == '__main__':
    cases = ['1', '2', '3', '4', '5', '6', '7', '8', '0']

    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=str, help='Choose one of this: 1, 2, 3, 4, 5, 6')
    parser.add_argument('--only_schedule', action=argparse.BooleanOptionalAction)
    parser.add_argument('--change_job', action=argparse.BooleanOptionalAction)
    parser.add_argument('--save_gt', action=argparse.BooleanOptionalAction)
    parser.add_argument('--save2file', action=argparse.BooleanOptionalAction)
    parser.add_argument('--offline_video', action=argparse.BooleanOptionalAction)
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

    det_job = True if args.change_job else False
    video = True if args.offline_video else False
    save2file = True if args.save2file else False
    save_gt = True if args.save_gt else False

    if args.case in cases:
        case = int(args.case)
    else:
        logging.error("The case does not exist")
        raise SystemExit(1)

    # create job description
    if case == 7:
        job = Job(int(case), seed=dist_seed, random_case_param=RandomCase(agent_number=4, task_number=15, condition_number=10))
    else:
        job = Job(int(case), seed=dist_seed)

    # create agents description
    agent_names = ["Human", "Robot"]
    agents: list[Agent] = []
    for agent_name in agent_names:
        agents.append(
            Agent(
                name=agent_name,
                job=copy.deepcopy(job),
                seed=sim_seed,
                answer_seed=answer_seed,
            )
        )

    if det_job:
        job = agents[0]._get_deterministic_job()

    # init solving method
    solving_method = METHOD(job=job, seed=solver_seed)
    solving_method.prepare()

    if args.comparison:
        if args.change_job:
            control_logic = ControlLogic(job=copy.deepcopy(job), agents=agents, method=solving_method)
            schedule1, stat1 = control_logic.run(experiments=True, save2file=save2file,
                                                 animation=save_gt, offline_video=video)
            seed = sim_seed
            nameList = [True, False]
            rand = np.random.default_rng(seed=seed)
            answers = []
            for task in job.task_sequence:
                answer = rand.choice(nameList, p=(1 - task.rejection_prob, task.rejection_prob), size=1)
                answers.append(answer)
                if not answer and task.universal:
                    task.universal = False
                    task.agent = ['Robot']
            control_logic = ControlLogic(job=copy.deepcopy(job), agents=agents, method=solving_method)
            schedule2, stat2 = control_logic.run(experiments=True, save2file=save2file,
                                                 animation=save_gt, offline_video=video)
        else:
            control_logic = ControlLogic(job=job, agents=agents, method=solving_method)
            schedule1, stat1 = control_logic.run(experiments=True, save2file=save2file,
                                                 animation=save_gt, offline_video=video)

            if case == 8:
                job = Job(int(case), seed=dist_seed,
                          random_case_param=RandomCase(agent_number=4, task_number=15, condition_number=10))
            else:
                job = Job(int(case), seed=dist_seed)
            solving_method = METHOD(job=job, seed=1)
            solving_method.prepare()
            control_logic = ControlLogic(job=job, agents=agents, method=solving_method)
            schedule2, stat2 = control_logic.run(experiments=True, save2file=save2file,
                                                 animation=save_gt, offline_video=video)

        file_name = args.file_name if args.file_name else comparison_save_file_name

        try:
            with open(file_name, "w") as outfile:
                json.dump({'schedule': schedule1+[schedule2[1]], 'statistics': [stat1, stat2]}, outfile,  indent=4)
                logging.info(f'Save data to {file_name}')
        except IndexError:
            with open(file_name, "w") as outfile:
                json.dump({'schedule': schedule1+schedule2, 'statistics': [stat1, stat2]}, outfile,  indent=4)
                logging.info(f'Save data to {file_name}')

    elif args.only_schedule:
        control_logic = ControlLogic(job=job, agents=agents, method=solving_method)
        control_logic.solving_method.decide(control_logic.get_observation_data(), 0)
        assert job.validate()
        perfect_schedule = copy.deepcopy(control_logic.schedule_as_dict(hierarchy=True))
        if save_gt:
            gantt = Vis(data=perfect_schedule, from_file=False)
            gantt.plot_schedule(file_name=f'schedule_case_{case}.pdf', case=case)
        job.__str__()

    else:
        control_logic = ControlLogic(job=job, agents=agents, method=solving_method)
        schedule, statistics = control_logic.run(experiments=True, save2file=save2file,
                                                 animation=save_gt, offline_video=video)

        assert job.validate()
        assert job.progress() == 100



