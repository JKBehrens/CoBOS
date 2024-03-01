from control.control_logic import ControlLogic
from methods.overlap_method import OverlapSchedule



def test_model_invalid():
    METHOD = OverlapSchedule
    case = '0'
    execute_job = ControlLogic(method=METHOD, case=case, distribution_seed=3, schedule_seed=0, sim_seed=0)
    execute_job.run(animation=True)



def test_case():
    METHOD = OverlapSchedule
    case = '5'
    execute_job = ControlLogic(method=METHOD, case=case, distribution_seed=3, schedule_seed=0, sim_seed=0)
    execute_job.run(animation=True)

    execute_job = ControlLogic(method=METHOD, case=case, distribution_seed=1, schedule_seed=0, sim_seed=0)
    execute_job.run(animation=True)



