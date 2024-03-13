from pathlib import Path
from control.control_logic import ControlLogic
from control.jobs import Job
from control.agents import Agent
from control.representation import JobDescription
from methods.overlap_method import OverlapSchedule
import copy
from exp_scripts.run_base_scheduling_exps import run_exp
from visualization.graphs import Vis

from hypothesis import Verbosity, example, given, strategies as st
from hypothesis import settings


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
    execute_job.run(animation=False)


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
    execute_job.run(animation=False)

    assert job.validate()

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


#     case 3, solver_seed 44, dist_seed 0 sim_seed 0
#     case 3, solver_seed 57, dist_seed 3 sim_seed 0
#     case 3, solver_seed 66, dist_seed 7 sim_seed 0
@settings(deadline=10000.0, max_examples=100, verbosity=Verbosity.verbose)
@given(st.integers(min_value=0, max_value=8), st.lists(st.integers(min_value=0, max_value=2000), min_size=4, max_size=4), st.booleans())
@example(3, [0,0,44,0], True)
@example(3, [3,0,57,0], True)
@example(3, [7,0,66,0], True)
@example(4, [1,0,0,0], True)
@example(1, [1280,0,773,0], False) # takes twice 10 secs for scheduling


def test_case_x(case: int, data: list[int], det_job: bool):
    distribution_seed, sim_seed, schedule_seed, answer_seed = data
    METHOD = OverlapSchedule

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

    if det_job:
        job = agents[0]._get_deterministic_job()

    solving_method = METHOD(job=job, seed=schedule_seed)
    solving_method.prepare()

    execute_job = ControlLogic(job=job, agents=agents, method=solving_method)
    schedule, stats = execute_job.run(animation=False, experiments=True)

    assert job.validate()

    # if det_job and sim_seed == schedule_seed == answer_seed:
    #     assert stats["initial_makespan"] == stats["final_makespan"]

    


    pass


def test_case_4_solver_seed_4_dist_seed_7_sim_seed_3():
    # ERROR:root:case 4, solver_seed 4, dist_seed 7
    # sim_seed 3

    case = 4
    solver_seed = 4
    dist_seed = 7
    sim_seed = 3
    schedule, stats = run_exp(
        OverlapSchedule,
        case=case,
        dist_seed=dist_seed,
        schedule_seed=solver_seed,
        sim_seed=sim_seed,
        answer_seed=sim_seed,
        det_job = False
    )

    print(schedule)
    # job = JobDescription.from_schedule(schedule)

    # assert job.validate()

    assert "FAIL" not in stats or stats["FAIL"] == False

    #     ERROR:root:
    # ERROR:root:No solution found. case 4
    # solver_seed 1, dist_seed 9
    # ERROR:root:case 4, solver_seed 1, dist_seed 9
    # sim_seed 7
    # WARNING:root:Duration of task 13 is wrong.
    # ERROR:root:
    # ERROR:root:No solution found. case 4
    # solver_seed 1, dist_seed 9
    # ERROR:root:case 4, solver_seed 1, dist_seed 9
    # sim_seed 2
    # ERROR:root:
    # ERROR:root:No solution found. case 4
    # solver_seed 1, dist_seed 9
    # ERROR:root:case 4, solver_seed 1, dist_seed 9
    # sim_seed 5

    #     ERROR:root:Something is wrong in FIND_TASK, status INFEASIBLE and the log of the solve
    # ERROR:root:ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='a4' Place='A3', conditions: [6], universal: True, start: 147, finish: [165, 9, 2, 7]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='h7' Place='A4', conditions: [7], universal: False, start: 121, finish: [147, 5, 6, 15]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='h1' Place='B2', conditions: [9], universal: False, start: 58, finish: [71, 6, 3, 4]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='a5' Place='B3', conditions: [10], universal: True, start: 91, finish: [104, 1, 1, 11]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='h8' Place='C1', conditions: [12], universal: False, start: 40, finish: [58, 10, 2, 9]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='a8' Place='D1', conditions: [], universal: True, start: 1, finish: [24, 17, 5, 1]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='a3' Place='D2', conditions: [], universal: True, start: 0, finish: [41, 8, 8, 25]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='h4' Place='D4', conditions: [], universal: False, start: 32, finish: [37, 1, 7, 5]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='a1' Place='A1', conditions: [4], universal: True, start: 159, finish: [172, 4, 1, 8]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='a6' Place='A2', conditions: [5], universal: True, start: 136, finish: [147, 1, 3, 7]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='r1' Place='B1', conditions: [8], universal: False, start: 103, finish: [119, 5, 4, 7]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='r5' Place='B4', conditions: [11], universal: False, start: 126, finish: [131, 1, 2, 2]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='r4' Place='C2', conditions: [13], universal: False, start: 41, finish: [62, 1, 11, 2]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='r6' Place='C3', conditions: [14], universal: False, start: 60, finish: [88, 19, 2, 7]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='r7' Place='C4', conditions: [15], universal: False, start: 68, finish: [97, 6, 8, 15]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='a2' Place='D3', conditions: [], universal: True, start: 62, finish: [71, 11, 11, 10]

    # ERROR:root:
    # ERROR:root:Something is wrong, status OPTIMAL and the log of the solve
    # ERROR:root:case 5, solver_seed 6, dist_seed 7
    # sim_seed 7
    # 2024-03-03 07:27:25,225 - distributed.worker - WARNING - Compute Failed
    # Key:       run_exp-313878089f66750bb91a9e30afe7b905
    # Function:  run_exp
    # args:      (<class 'methods.overlap_method.OverlapSchedule'>, 5, 7, 6, 7, 7)
    # kwargs:    {}
    # Exception: "AssertionError('dependency graph violation for task 3.')"

    #     ERROR:root:
    # ERROR:root:No solution found. case 5
    # solver_seed 3, dist_seed 9
    # ERROR:root:case 5, solver_seed 3, dist_seed 9
    # sim_seed 9
    # 2024-03-03 07:36:35,739 - distributed.worker - WARNING - Compute Failed
    # Key:       run_exp-d8b8e3bac5acfe5abbc604762d7817dc
    # Function:  run_exp
    # args:      (<class 'methods.overlap_method.OverlapSchedule'>, 5, 9, 3, 9, 9)
    # kwargs:    {}
    # Exception: "AssertionError('dependency graph violation for task 5.')"

    #     RROR:root:Something is wrong in FIND_TASK, status INFEASIBLE and the log of the solve
    # ERROR:root:ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='a1' Place='A1', conditions: [4, 5], universal: True, start: 155, finish: [173, 2, 15, 1]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='a4' Place='A3', conditions: [5, 6, 7], universal: True, start: 111, finish: [141, 18, 2, 10]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='h7' Place='A4', conditions: [6, 7], universal: False, start: 141, finish: [155, 4, 4, 6]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='h1' Place='B2', conditions: [8, 9, 10], universal: False, start: 90, finish: [103, 8, 2, 3]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='a5' Place='B3', conditions: [9, 10, 11], universal: True, start: 104, finish: [111, 1, 1, 5]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='h8' Place='C1', conditions: [12, 13], universal: False, start: 46, finish: [71, 10, 2, 9]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='a2' Place='D3', conditions: [], universal: True, start: 0, finish: [44, 12, 12, 10]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='h4' Place='D4', conditions: [], universal: False, start: 35, finish: [46, 1, 7, 5]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='a6' Place='A2', conditions: [4, 5, 6], universal: True, start: 133, finish: [161, 1, 3, 24]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='r1' Place='B1', conditions: [8, 9], universal: False, start: 122, finish: [133, 4, 6, 1]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='r5' Place='B4', conditions: [10, 11], universal: False, start: 102, finish: [122, 1, 12, 7]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='r4' Place='C2', conditions: [12, 13, 14], universal: False, start: 67, finish: [79, 1, 8, 3]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='r6' Place='C3', conditions: [13, 14, 15], universal: False, start: 79, finish: [85, 1, 2, 3]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='r7' Place='C4', conditions: [14, 15], universal: False, start: 85, finish: [98, 7, 2, 4]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='a8' Place='D1', conditions: [], universal: True, start: 0, finish: [23, 17, 5, 1]
    # ID: {self.id}, agent: {self.agent}, status: {TaskState(self.state).name}, task action: Object='a3' Place='D2', conditions: [], universal: True, start: 44, finish: [71, 23, 7, 13]

    # ERROR:root:
    # ERROR:root:Something is wrong, status OPTIMAL and the log of the solve
    # ERROR:root:case 6, solver_seed 3, dist_seed 7
    # sim_seed 7


def test_det_job_perf(tmp_path: Path):
    # ERROR:root:case 4, solver_seed 4, dist_seed 7
    # sim_seed 3

    case = 4
    solver_seed = 4
    dist_seed = 0
    sim_seed = 4
    schedule, stats = run_exp(
        OverlapSchedule,
        case=case,
        dist_seed=dist_seed,
        schedule_seed=solver_seed,
        sim_seed=sim_seed,
        answer_seed=sim_seed,
        det_job = True
    )

    print(schedule)

    gantt = Vis(data=schedule, from_file=False)
    gantt.plot_schedule(tmp_path.joinpath("simulation.png").__str__())

    assert "FAIL" not in stats or stats["FAIL"] == False
    assert stats["initial_makespan"] == stats["final_makespan"]

