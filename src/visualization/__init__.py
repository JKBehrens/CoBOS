import os

from visualization.graphs import Vis
from visualization.web_visualization import Web_vis

initial_and_final_schedule_save_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'data_for_visualization/initial_and_final_schedule.json')
schedule_save_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'data_for_visualization/schedule.json')

comparison_save_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'data_for_visualization/comparison.json')


allocation_method_vis_save_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'data_for_visualization/allocation_method_visualization')
