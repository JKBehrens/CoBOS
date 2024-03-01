import numpy as np
from control.agents import Agent
from control.jobs import Job


def test_job_gen():
    case = 5
    seed = 42
    job = Job(case, seed=seed)

    assert isinstance(job.__str__(), str)
    assert 0.0 == job.progress()


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
