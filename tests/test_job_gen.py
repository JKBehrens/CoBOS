

from control.jobs import Job


def test_job_gen():
    case = 5
    seed = 42
    job = Job(case, seed=seed)

    assert isinstance(job.__str__(), str)
    assert 0.0 == job.progress()

    # job.sample()
