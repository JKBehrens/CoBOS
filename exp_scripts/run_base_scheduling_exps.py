from itertools import product
from pathlib import Path
from typing import Any
from dask_jobqueue.slurm import SLURMCluster
from distributed import Client, Future
from methods.overlap_method import OverlapSchedule

from utils.file_handler import EnhancedJSONEncoder

import copy
import json
import logging
from control.agents import Agent
from control.control_logic import ControlLogic
from control.jobs import Job
from methods.solver import Solver

from pydantic import BaseModel


class ExperimentSettings(BaseModel):
    run_on_cluster: bool = False
    num_workers: int = 2

    exp_folder: Path = Path(__file__).parent.joinpath("experiments/base_sched/")
    cases: list[int] = []


# settings = ExperimentSettings()
# s = json.dumps(settings.dict(), cls=EnhancedJSONEncoder)
# print(s)
# pass


def start_cluster(exp_settings: ExperimentSettings) -> Client:
    if exp_settings.run_on_cluster:
        cluster = SLURMCluster(
            cores=8,
            processes=1,
            memory="16GiB",
            # account="behrejan",
            walltime="01:00:00",
            # partition="compute",
            queue="compute",
            env_extra=["#SBATCH --partition=compute"],
            # worker_extra_args=["--partition='compute'"]
        )

        cluster.scale(n=exp_settings.num_workers)
        client = Client(cluster)

        return client
    else:
        return Client()


def run_exp(
    method: Solver.__class__,
    case: int,
    dist_seed: int,
    schedule_seed: int,
    sim_seed: int,
    answer_seed: int,
) -> tuple[list[Any], dict[str, Any]]:

    method = method
    case = case
    dist_seed = dist_seed
    sim_seed = sim_seed
    schedule_seed = schedule_seed
    answer_seed = answer_seed
    job = Job(case, seed=dist_seed)

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

    solving_method = method(job=job, seed=schedule_seed)
    solving_method.prepare()

    control_logic = ControlLogic(job=job, agents=agents, method=solving_method)
    schedule, statistics = control_logic.run(experiments=True)


    if not job.validate() or not job.progress() == 100:
        statistics["FAIL"] = True

    return schedule, statistics


def run_exps(client: Client, exp_settings: ExperimentSettings):

    exp_folder = exp_settings.exp_folder
    exp_folder.mkdir(parents=True, exist_ok=True)

    methods = [OverlapSchedule]

    # dist_seed, schedule_seed, sim_seed, answer_seed
    

    
    for case in range(1, 7):
        print(f"case: {case}")
        for method in methods:
            futures: dict[str, Future] = {}
            seed_iter = product(range(10), range(10), range(10))
            for dist_seed, schedule_seed, sim_seed in seed_iter:
                answer_seed = sim_seed
                # job = Job(case, seed=dist_seed)
                fname = f"sched_case_{case}_method_{method.name()}_dist_{dist_seed}_sim_{sim_seed}_schedule_{schedule_seed}.json"
                if exp_folder.joinpath(fname).is_file():
                    continue

                # res = client.submit(run_exp, job, 0, 0)
                res: Future = client.submit(run_exp, method, case, dist_seed, schedule_seed, sim_seed, answer_seed)
                futures[fname] = res

            for fname, res in futures.items():
                with open(exp_folder.joinpath(fname), "w") as outfile:
                    schedule, stats = res.result()
                    json.dump({"schedule": schedule, "statistics": stats}, outfile)
                    logging.info(f"Save data to {fname}")

    # save original schedule, simulation with the same seed as in schedule, simulation with the another seed
    # statistics is saved for each simulation:
    # - predicted makespan, final makespan,
    # - list of the rejection tasks with time, when it was rejected
    # - list with the rescheduling info: when, status of solver, makespan, calculation time
    # - list with the rescheduling evaluation info: when, makespan, calculation time

    

    client.shutdown()

    # files = list(Path(exp_folder).glob("*.json"))

    # make_all_the_magic(files, result_folder: Path)


if __name__ == "__main__":
    exp_settings = ExperimentSettings()
    try:
        client = start_cluster(exp_settings)
        run_exps(client=client, exp_settings=exp_settings)
    finally:
        client.shutdown()
