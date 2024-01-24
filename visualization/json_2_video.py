import json
import logging
import os
import matplotlib.animation as animation
from visualization.graphs import Vis


def video_parser():
    # animation function
    data_length = 0
    def animate(i):
        logging.info(f'Parsing to video. Progress {round(100*(i/data_length),2)}%')
        plot.data = data[str(i)]["Schedule"]
        plot.current_time = data[str(i)]["Time"]
        plot.plot_schedule(video=True)

    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'data_for_visualization/sim_2_video.json')
    with open(filename, "r+") as json_file:
        data = json.load(json_file)
    plot = Vis()
    data_length = len(data)

    # calling the animation function
    anim = animation.FuncAnimation(plot.fig, animate, interval=200, frames=len(data))

    # saves the animation in our desktop
    anim.save('simulation_visualization.mp4', writer='ffmpeg', fps=1)


