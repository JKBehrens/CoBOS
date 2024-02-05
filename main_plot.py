import argparse
import json
import time

import numpy as np

from visualization import Vis, initial_and_final_schedule, schedule, video_parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str,
                        help='Select the simulation you want to render: sim_vis or plot_schedule')
    args = parser.parse_args()

    save_file_name = ''
    if args.mode == "video":
        video_parser()
    elif args.mode == "sim_vis":
        with open(initial_and_final_schedule, "r+") as json_file:
            data = json.load(json_file)
        save_file_name = 'simulation1.png'
    else:
        with open(schedule, "r+") as json_file:
            data = json.load(json_file)
        save_file_name = 'schedule.png'

    gantt = Vis(data=data, from_file=True)
    gantt.plot_schedule(save_file_name)
