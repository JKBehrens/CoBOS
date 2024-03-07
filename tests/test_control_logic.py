from methods import OverlapSchedule
from control.control_logic import ControlLogic
from control.agents import Agent
from control.jobs import Job
import copy

def test_refactoring_control_logic():
    method = OverlapSchedule
    case = 5
    seed = 42
    sim_seed = 0
    schedule_seed = 0
    answer_seed = 1
    job = Job(case, seed=seed)

    agent_names = ["Human", "Robot"]
    agents: list[Agent] = []
    for agent_name in agent_names:
        agents.append(Agent(name=agent_name, job=copy.deepcopy(job), seed=sim_seed, answer_seed=answer_seed))

    solving_method = method(job=job, seed=schedule_seed)
    solving_method.prepare()

    control_logic = ControlLogic(job=job, agents=agents, method=solving_method)
    schedule, statistics = control_logic.run(experiments=True)

    assert job.validate()
    assert job.progress() == 100

    assert control_logic.job == job
    assert control_logic.agents[0] == agents[0]



