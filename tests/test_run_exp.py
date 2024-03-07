


import json
from exp_scripts.run_base_scheduling_exps import ExperimentRun, run_exp
from methods.max_duration import MaxDuration
from methods.overlap_method import OverlapSchedule
from methods.random_task_allocation import RandomAllocation


def test_run_exp():

    methods = [OverlapSchedule, RandomAllocation, MaxDuration]

    # settings = {'dist_seed': 0, 'schedule_seed': 24, 'sim_seed': 0, 'det_job': False, 'answer_seed': 0, 'method_name': 'overlapschedule', 'case': 1}
    settings = {'case': 1, 'dist_seed': 0, 'schedule_seed': 13, 'sim_seed': 0, 'answer_seed': 0, 'method_name': 'overlapschedule', 'det_job': False}
    settings = {'case': 1, 'dist_seed': 1, 'schedule_seed': 80, 'sim_seed': 0, 'answer_seed': 0, 'method_name': 'overlapschedule', 'det_job': True}
    settings = {'case': 4, 'dist_seed': 0, 'schedule_seed': 26, 'sim_seed': 0, 'answer_seed': 0, 'method_name': 'randomallocation', 'det_job': True}
    settings = {'case': 6, 'dist_seed': 2, 'schedule_seed': 41, 'sim_seed': 0, 'answer_seed': 0, 'method_name': 'overlapschedule', 'det_job': False}
    settings = {'case': 6, 'schedule_seed': 53, 'dist_seed': 5, 'sim_seed': 0, 'answer_seed': 0, 'method_name': 'overlapschedule', "det_job": True}
    # settings = {'case': 6, 'schedule_seed': 53, 'dist_seed': 5, 'sim_seed': 0, 'answer_seed': 0, 'method_name': 'overlapschedule', "det_job": False}
    
    settings = ExperimentRun(**settings)

    for method in methods:
        schedule, statistics = run_exp(method=method, **settings.dict())



def test_run_to_json():
    settings = {'case': 1, 'dist_seed': 0, 'schedule_seed': 13, 'sim_seed': 0, 'answer_seed': 0, 'method_name': 'overlapschedule', 'det_job': False}
    settings = ExperimentRun(**settings)

    s = settings.json()
    s2 = json.dumps(settings.dict())


    data = json.loads(s)
    data2 = json.loads(s2)
    
    new_settings = ExperimentRun(**data)
    settings2 = ExperimentRun(**data2)

    assert new_settings == settings
    assert settings2 == settings