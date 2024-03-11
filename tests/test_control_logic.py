from methods import OverlapSchedule, MaxDuration, RandomAllocation
from control.control_logic import ControlLogic
from inputs.data_generator import RandomCase
from control.agents import Agent
from control.jobs import Job
import copy


def test_refactoring_control_logic():
    methods = [OverlapSchedule, MaxDuration, RandomAllocation]
    cases = [0,1,2,3,4,5,6,7,8]
    seed = 42
    sim_seed = 0
    schedule_seed = 0
    answer_seed = 1

    for case in cases:
        for method in methods:
            if case == 8:
                job = Job(case, seed=seed, random_case_param=RandomCase(agent_number=4, task_number=15, condition_number=10))
            else:
                job = Job(case, seed=seed)

            agent_names = ["Human", "Robot"]
            agents: list[Agent] = []

            for agent_name in agent_names:
                agents.append(Agent(name=agent_name, job=copy.deepcopy(job), seed=sim_seed, answer_seed=answer_seed))

            solving_method = method(job=job, seed=schedule_seed)
            solving_method.prepare()

            control_logic = ControlLogic(job=job, agents=agents, method=solving_method)
            schedule, statistics = control_logic.run(experiments=False)

            assert job.validate(), f'method {method}, case {case}'
            assert job.progress() == 100,  f'method {method}, case {case}'

            assert control_logic.job == job
            assert control_logic.agents[0] == agents[0]



