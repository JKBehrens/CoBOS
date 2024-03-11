from itertools import product
from pathlib import Path
from typing import Any, Iterable
from dask_jobqueue.slurm import SLURMCluster
from distributed import Client, Future
from dask.distributed import wait
import tqdm

from methods import OverlapSchedule, RandomAllocation, MaxDuration


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

    exp_folder: Path = Path(__file__).parent.joinpath("experiments/base_sched")
    cases: list[int] = []

    dist_seed: int = 10 
    schedule_seed: int = 100
    sim_seed: int = 1

    det_job: list[bool] = [True, False]

    variation_int_var_lst: list[str] = ["dist_seed", "schedule_seed", "sim_seed"]
    variation_list_var_lst: list[str] = ["det_job"]
    

    def get_iter(self) -> Iterable:

        attr_list = [getattr(self, attr) for attr in self.variation_int_var_lst]
        range_lst = list(map(
            range, attr_list
        ))
        lst_attr_list = [getattr(self, attr) for attr in self.variation_list_var_lst]
        seed_iter = product(*range_lst, *lst_attr_list)

        return seed_iter

    # seed_iter = product(range(10), range(10), range(10))
    #         for dist_seed, schedule_seed, sim_seed in seed_iter:


class ExperimentRun(BaseModel):
    case: int

    dist_seed: int
    schedule_seed: int
    sim_seed: int
    answer_seed: int
    method_name: str

    det_job: bool

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
    # case: int,
    # dist_seed: int,
    # schedule_seed: int,
    # sim_seed: int,
    # answer_seed: int,
    # det_job: bool,
    ** kwargs
) -> tuple[list[Any], dict[str, Any]]:
    
    case: int = kwargs["case"]
    dist_seed: int = kwargs["dist_seed"]
    schedule_seed: int = kwargs["schedule_seed"]
    sim_seed: int = kwargs["sim_seed"]
    answer_seed: int = kwargs["answer_seed"]
    det_job: bool = kwargs["det_job"]



    method = method
    # case = case
    # dist_seed = dist_seed
    # sim_seed = sim_seed
    # schedule_seed = schedule_seed
    # answer_seed = answer_seed
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
    if det_job:
        job = agents[0]._get_deterministic_job()

    solving_method = method(job=job, seed=schedule_seed)
    solving_method.prepare()

    control_logic = ControlLogic(job=job, agents=agents, method=solving_method)
    schedule, statistics = control_logic.run(experiments=True)


    if not job.validate() or not job.progress() == 100:
        statistics["FAIL"] = True

    return schedule, statistics


def save_data(exp_folder: Path, fname: str, schedule_stats: tuple[Any, Any], settings: dict[str, Any]):
    schedule, stats = schedule_stats
    with open(exp_folder.joinpath(fname), "w") as outfile:
        # schedule, stats = res
        # settings = run_settings[fname]
        json.dump(out:={"schedule": schedule, "statistics": stats, "settings": settings}, outfile)
        logging.info(f"Save data to {fname}")

    return out


def run_exps(client: Client, exp_settings: ExperimentSettings):

    exp_folder = exp_settings.exp_folder
    exp_folder.mkdir(parents=True, exist_ok=True)

    methods = [OverlapSchedule, RandomAllocation, MaxDuration]

    # dist_seed, schedule_seed, sim_seed, answer_seed
    

    
    for case in range(1, 7):
        print(f"case: {case}")
        for method in methods:
            futures: dict[str, Future] = {}
            run_settings: dict[str, ExperimentRun] = {}

            # seed_iter = product(range(10), range(10), range(10))
            seed_iter = exp_settings.get_iter()
            for seeds_tpl in seed_iter:
                settings = dict(zip(exp_settings.variation_int_var_lst + exp_settings.variation_list_var_lst, seeds_tpl))
                settings["answer_seed"] = settings["sim_seed"]
                # job = Job(case, seed=dist_seed)
                #TODO: make sure that names are sorted
                fname = f"sched_case_{case}_method_{method.name()}"
                for name, seed in settings.items():
                    fname += f"_{name}_{seed}"
                fname += ".json"

                settings["method_name"] = method.name()
                settings["case"] = case

                settings = ExperimentRun(**settings)
                
                # fname = f"sched_case_{case}_method_{method.name()}_dist_{dist_seed}_sim_{sim_seed}_schedule_{schedule_seed}.json"
                if exp_folder.joinpath(fname).is_file():
                    continue

                # res = run_exp(method=method, **settings)
                res: Future = client.submit(run_exp, method, **settings.dict())
                
                future_saving = client.submit(save_data, exp_folder, fname, res, settings.dict())

                # res: Future = client.submit(run_exp, method, case, dist_seed, schedule_seed, sim_seed, answer_seed)
                futures[fname] = future_saving
                run_settings[fname] = settings

            print("progress")
            # progress(futures.values())
            batch_completion = 0
            done, not_done = wait(list(futures.values()), return_when='FIRST_COMPLETED')
            # for fn, future in futures.items():
            batch_size = len(done) + len(not_done)
            batch_completion += len(done)

            del futures
            del run_settings
            with tqdm.tqdm(total=batch_size) as pbar:
                while len(not_done) > 0:
                    done, not_done = wait(not_done, return_when='FIRST_COMPLETED')
                    batch_completion += len(done)
                    # cur_perc = batch_completion / batch_size

                    pbar.update(len(done))
                    # print(f"completed work: {batch_completion} / {batch_size}")
                
            
            print(f"batch: case {case}, method {method} completed.")

           




            # for fname, res in futures.items():
            #     schedule, stats = res.result()
            #     with open(exp_folder.joinpath(fname), "w") as outfile:
            #         # schedule, stats = res
            #         settings = run_settings[fname]
            #         json.dump({"schedule": schedule, "statistics": stats, "settings": settings.dict()}, outfile)
            #         logging.info(f"Save data to {fname}")

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
