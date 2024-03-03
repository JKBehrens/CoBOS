from control.control_logic import ControlLogic
from control.jobs import Job
from control.agents import Agent
from methods.overlap_method import OverlapSchedule
import copy
from exp_scripts.run_base_scheduling_exps import run_exp


def test_model_invalid():
    METHOD = OverlapSchedule
    case = 0
    distribution_seed = 3
    sim_seed = 0
    schedule_seed = 0
    answer_seed = 1
    job = Job(case, seed=distribution_seed)

    agent_names = ["Human", "Robot"]
    agents: list(Agent) = []
    for agent_name in agent_names:
        agents.append(
            Agent(
                name=agent_name,
                job=copy.deepcopy(job),
                seed=sim_seed,
                answer_seed=answer_seed,
            )
        )

    solving_method = METHOD(job=job, seed=schedule_seed)
    solving_method.prepare()

    execute_job = ControlLogic(job=job, agents=agents, method=solving_method)
    execute_job.run(animation=True)


def test_case():
    METHOD = OverlapSchedule
    case = 5
    distribution_seed = 3
    sim_seed = 0
    schedule_seed = 0
    answer_seed = 1
    job = Job(case, seed=distribution_seed)

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

    solving_method = METHOD(job=job, seed=schedule_seed)
    solving_method.prepare()

    execute_job = ControlLogic(job=job, agents=agents, method=solving_method)
    execute_job.run(animation=True)

    distribution_seed = 1
    job = Job(case, seed=distribution_seed)

    agents: list(Agent) = []
    for agent_name in agent_names:
        agents.append(
            Agent(
                name=agent_name,
                job=copy.deepcopy(job),
                seed=sim_seed,
                answer_seed=answer_seed,
            )
        )

    solving_method = METHOD(job=job, seed=schedule_seed)
    solving_method.prepare()

    execute_job = ControlLogic(job=job, agents=agents, method=solving_method)
    execute_job.run(animation=True)



