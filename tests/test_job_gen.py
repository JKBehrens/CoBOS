from itertools import combinations
import numpy as np
from control.agents import Agent
from control.jobs import Action, Job, Task
from inputs.data_generator import RandomCase


def test_job_gen():
    case = 5
    seed = 42
    job = Job(case, seed=seed)

    assert isinstance(job.__str__(), str)
    assert 0.0 == job.progress()

    for task in job.task_sequence:
        assert isinstance(task, Task)
        assert isinstance(task.action, Action)


    
def test_random_job_gen():
    case = 8
    seed = 42
    job = Job(case, seed=seed, random_case_param=RandomCase(agent_number=4, task_number=15, condition_number=10))
    job2 = Job(case, seed=seed, random_case_param=RandomCase(agent_number=4, task_number=15, condition_number=10))
    jobs = [job, job2]
    assert all([job1.job_description == job2.job_description for job1, job2 in combinations(jobs, 2)] )

    assert isinstance(job.__str__(), str)
    assert 0.0 == job.progress()

    det_jobs: list[Job] = []
    agent = Agent(name="Human", job=job, seed=seed)
    det_jobs.append(agent._get_deterministic_job())

    agent = Agent(name="Human", job=job2, seed=seed)
    det_jobs.append(agent._get_deterministic_job())


    for task in job.task_sequence:
        assert isinstance(task, Task)
        assert isinstance(task.action, Action)

    assert all([job1.job_description == job2.job_description for job1, job2 in combinations(det_jobs, 2)] )
    assert all([job1.job_description == job2.job_description for job1, job2 in combinations(jobs, 2)] )



def test_job_sampling():
    case = 5
    seed = 42
    sim_seed = 0
    answer_seed = 1
    job = Job(case, seed=seed)

    agent_name = "Human"
    agent = Agent(name=agent_name, job=job, seed=sim_seed, answer_seed=answer_seed)

    job2 = agent._get_deterministic_job()

    assert isinstance(job2, Job)
    assert all(
        [task.rejection_prob == 0.0 for task in job2.task_sequence if task.universal]
    )

    agent2 = Agent(name=agent_name, job=job2, seed=sim_seed, answer_seed=answer_seed)

    assert agent.task_duration == agent2.task_duration

    job3 = agent2._get_deterministic_job()

    assert all(
        [
            task1.agent == task2.agent
            for task1, task2 in zip(job2.task_sequence, job3.task_sequence)
        ]
    )
    assert all(
        [
            np.allclose(task1.distribution, task2.distribution)
            for task1, task2 in zip(job2.task_sequence, job3.task_sequence)
        ]
    )

    assert job.get_universal_task_number() >= job2.get_universal_task_number()


    agent_name = "Robot"
    agent_robot = Agent(name=agent_name, job=job, seed=sim_seed, answer_seed=answer_seed)
    job2_rob = agent_robot._get_deterministic_job()

    assert all(
        [
            task1.agent == task2.agent
            for task1, task2 in zip(job2.task_sequence, job2_rob.task_sequence)
        ]
    )
    assert all(
        [
            np.allclose(task1.distribution, task2.distribution)
            for task1, task2 in zip(job2.task_sequence, job2_rob.task_sequence)
        ]
    )

    assert job.get_universal_task_number() >= job2_rob.get_universal_task_number()

