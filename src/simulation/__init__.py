import os

from simulation.sim import Sim
from simulation.task_execution_time_const import get_approximated_task_duration
sim_param_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
