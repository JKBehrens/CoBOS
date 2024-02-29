import argparse
import json
import time

import numpy as np

from visualization import Vis, initial_and_final_schedule_save_file_name, schedule_save_file_name, video_parser, comparison_save_file_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str,
                        help='Select the simulation you want to render: sim_vis, comparison, plot_schedule')
    args = parser.parse_args()

    save_file_name = ''
    if args.mode == "video":
        video_parser()
    elif args.mode == "sim_vis":
        with open(initial_and_final_schedule_save_file_name, "r+") as json_file:
            data = json.load(json_file)
        save_file_name = 'simulation1.png'
    elif args.mode == "comparison":
        with open(comparison_save_file_name, "r+") as json_file:
            data = json.load(json_file)
        save_file_name = 'comparison.png'
    else:
        with open(initial_and_final_schedule_save_file_name, "r+") as json_file:
            data = json.load(json_file)
            if len(data) == 2:
                data = data[1]
            elif len(data) == 1:
                data = data[0]
        save_file_name = 'schedule.png'

    gantt = Vis(data=data, from_file=True)
    gantt.plot_schedule(save_file_name)
