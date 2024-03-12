from itertools import combinations
from pathlib import Path
from methods import OverlapSchedule, MaxDuration, RandomAllocation, DynamicAllocation
from control.control_logic import ControlLogic
from inputs.data_generator import RandomCase
from control.agents import Agent
from control.jobs import Job
import copy

from visualization.graphs import Vis


def test_refactoring_control_logic(tmp_path: Path):
    methods = [OverlapSchedule, MaxDuration, RandomAllocation, DynamicAllocation]
    cases = [0,1,2,3,4,5,6,7,8]
    dist_seed = 42
    sim_seed = 0
    schedule_seed = 0
    answer_seed = 1

    for case in cases:
        jobs: list[Job] = []
        det_jobs: list[Job] = []
        for method in methods:
            if case == 8:
                job = Job(case, seed=dist_seed, random_case_param=RandomCase(agent_number=4, task_number=15, condition_number=10))
            else:
                job = Job(case, seed=dist_seed)

            agent_names = ["Human", "Robot"]
            agents: list[Agent] = []

            for agent_name in agent_names:
                agents.append(Agent(name=agent_name, job=copy.deepcopy(job), seed=sim_seed, answer_seed=answer_seed))

            det_job = agents[0]._get_deterministic_job()
            det_jobs.append(copy.deepcopy(det_job))
            jobs.append(copy.deepcopy(job))

            solving_method = method(job=job, seed=schedule_seed)
            solving_method.prepare()

            control_logic = ControlLogic(job=job, agents=agents, method=solving_method)
            schedule, statistics = control_logic.run(experiments=True)

            assert job.validate(), f'method {method}, case {case}'
            assert job.progress() == 100,  f'method {method}, case {case}'

            assert control_logic.job == job
            assert control_logic.agents[0] == agents[0]

            gantt = Vis(data=schedule, from_file=False)
            gantt.plot_schedule(tmp_path.joinpath(f"simulation_case_{case}_method_{method.name()}.png").__str__())

        assert all([job1.job_description == job2.job_description for job1, job2 in combinations(jobs, 2)] )
        assert all([job1.job_description == job2.job_description for job1, job2 in combinations(det_jobs, 2)] )

        



