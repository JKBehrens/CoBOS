from control.control_logic import ControlLogic



def test_model_invalid():
    case = '0'
    execute_job = ControlLogic(case, distribution_seed=3, schedule_seed=0, sim_seed=0)
    execute_job.run(animation=True)



def test_case():
    case = '5'
    execute_job = ControlLogic(case, distribution_seed=3, schedule_seed=0, sim_seed=0)
    execute_job.run(animation=True)

    execute_job = ControlLogic(case, distribution_seed=1, schedule_seed=0, sim_seed=0)
    execute_job.run(animation=True)



