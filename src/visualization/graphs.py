"""
    Visualization class for Gantt chart and dependency graph

    @author: Marina Ionova, student of Cybernetics and Robotics at the CTu in Prague
    @contact: marina.ionova@cvut.cz
"""
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import json
import networkx as nx
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
import streamlit as st
import altair as alt
import logging

# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width / len(orig_handle.colors) * i - handlebox.xdescent,
                                          -handlebox.ydescent],
                                         width / len(orig_handle.colors),
                                         height,
                                         facecolor=c,
                                         edgecolor='none'))

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch


def get_task_from_id(id, data):
    for agent in data:
        for task in data[agent]:
            if task['id'] == id:
                return task["action"]['place']
    return None


class Vis:
    def __init__(self, horizon=None, data=None, from_file=False):
        self.fig = plt.figure(figsize=(12, 10)) #, facecolor="#EDE7E4")
        self.data4video = 'src/visualization/data_for_visualization/sim_2_video.json'
        self.save_path: Path = Path("~/sched_video/img/").expanduser()
        self.gnt1 = None
        self.gnt2 = None
        self.data = data
        self.current_time = 0
        self.from_file = from_file
        self.horizon = horizon

        # positions
        self.y_pos_and_text = {True:
                                   {"Human": [7.5, 8.8, 6.6],
                                    "Robot": [4.5, 5.8, 3.6]},
                               False:
                                   {"Human": [10.5, 11.8, 9.6],
                                    "Robot": [1.5, 2.8, 0.6]}}
        self.color = {None: 'lightcoral', -1: 'lightcoral', 0: 'gold', 1: 'lightgreen', 2: 'silver'}

        widths = [3, 1]
        self.gs0 = self.fig.add_gridspec(1, 2, width_ratios=widths)
        self.gs00 = self.gs0[0].subgridspec(2, 1)
        self.legend = True

    def delete_existing_file(self):
        try:
            with open(self.data4video, "w") as json_file:
                json.dump({}, json_file)
        except FileNotFoundError:
            pass

    def set_plot_param(self, title, gs, lim=None, reschedule_num=False):
        self.gnt = self.fig.add_subplot(gs)  # 211
        self.gnt.set_title(title)

        if not reschedule_num:
            # setting Y-axis limits
            self.gnt.set_ylim(0, 13)
            # setting ticks on y-axis
            self.gnt.set_yticks([1.5, 4.5, 7.5, 10.5])
            # Labelling tickes of y-axis
            self.gnt.set_yticklabels(
                ['Robot', 'Allocatable\n for robot', 'Allocatable\n for human', 'Human'])
        else:
            # setting Y-axis limits
            self.gnt.set_ylim(0, 14.6)
            # setting ticks on y-axis
            self.gnt.set_yticks([1.5, 4.5, 7.5, 10.5, 14])
            # Labelling tickes of y-axis
            self.gnt.set_yticklabels(
                ['Robot', 'Allocatable\n for robot', 'Allocatable\n for human',
                 'Human', 'Rescheduling'])

        # setting X-axis limits
        if lim:
            self.gnt.set_xlim(0, lim + 5)

        # setting labels for x-axis and y-axis
        self.gnt.set_xlabel('Time [s]')

        # setting graph attribute
        self.gnt.grid(True)

        # ------ choose some colors
        colors1 = ['royalblue']  # 'lightsteelblue', 'cornflowerblue',['thistle', 'plum', 'violet']
        colors11 = [ 'mediumslateblue']
        colors2 = ['lightseagreen']  # 'paleturquoise', 'turquoise',
        colors5 = ['royalblue', 'lightseagreen', 'mediumslateblue']
        colors4 = ['cornflowerblue', 'turquoise', '#a993e3']
        colors3 = ['lightsteelblue', 'paleturquoise', 'thistle']

        # ------ get the legend-entries that are already attached to the axis
        self.h, self.l = self.gnt.get_legend_handles_labels()

        # ------ append the multicolor legend patches
        self.h.append(MulticolorPatch(colors1))
        self.l.append("Robot task")
        self.h.append(MulticolorPatch(colors11))
        self.l.append("Human task")

        self.h.append(MulticolorPatch(colors2))
        self.l.append("Allocatable task")

        self.h.append(MulticolorPatch(colors3))
        self.l.append("Preparation")

        self.h.append(MulticolorPatch(colors4))
        self.l.append("Execution")

        self.h.append(MulticolorPatch(colors5))
        self.l.append("Completion")


        #______________________________
        self.h.append(MulticolorPatch(['lightcoral']))
        self.l.append("Not available")

        self.h.append(MulticolorPatch(['gold']))
        self.l.append("Available")

        self.h.append(MulticolorPatch(['lightgreen']))
        self.l.append("In process")

        self.h.append(MulticolorPatch(['silver']))
        self.l.append("Completed")



        # self.labels = []
        # self.labels.append(mpatches.Patch(color='lightcoral', label='Not available'))
        # self.labels.append(mpatches.Patch(color='gold', label='Available'))
        # self.labels.append(mpatches.Patch(color='lightgreen', label='In process'))
        # self.labels.append(mpatches.Patch(color='silver', label='Completed'))

    def set_horizon(self, data):
        # flatten the nested dictionary to get all "finish" values
        finish_time = 0
        for schedule in data:
            for agent in schedule:
                for task in schedule[agent]:
                    try:
                        if task["finish"][0] > finish_time:
                            finish_time = task['finish'][0]
                    except TypeError:
                        break
        # find the maximum "finish" time
        return finish_time

    def plot_schedule(self, file_name:Path|str="", video=False, stat=None, case=None):
        self.fig = plt.figure(figsize=(12, 10)) #, facecolor="#EDE7E4")
        if not isinstance(file_name, Path):
            file_name = Path(file_name)
        if 'simulation' in file_name.stem:
            title = ['Gantt Chart: Perfect knowledge', 'Gantt Chart: Initial schedule']
            # title = ['Gantt Chart: initial', 'Gantt Chart']
            gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 2])
            positions = [[311, 312], [313]]
            # gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1])
            # positions = [[211, 212], [221]]
            local_data = self.data
        elif video:
            title = ['Gantt Chart: Initial schedule', 'Gantt Chart: Simulation']
            # title = ['Gantt Chart: Perfect', 'Gantt Chart: Initial']
            gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 2])
            positions = [[311, 312], [313]]
            local_data = self.data
        elif 'comparison' in file_name.stem:
            title = ['Gantt Chart: initial', 'Gantt Chart: final (same sampling seed)',
                     'Gantt Chart: final (different sampling seed)']
            gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 2])
            positions = [[311, 312, 313], [314]]
            # gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1])
            # positions = [[111, 112, 113], [212]]
            local_data = self.data['schedule']
            # local_data = self.data
        else:
            title = [f'Gantt Chart for case {case}']
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
            positions = [[211], [212]]
            local_data = [self.data]
        horizon = self.set_horizon(local_data)
        for i, position in enumerate(positions[0]):
            if 'comparison' in file_name.stem or (video and stat is not None):
                if i != 0:
                    self.set_plot_param(title[i], gs[i, :], lim=horizon, reschedule_num=True)
                    colors = {0: 'red', 1: 'blue'}
                    position = {0: 14, 1: 15.5}
                    try:
                        if stat is None:
                            for idx, solver_info in enumerate(self.data['statistics'][i - 1]['solver']):
                                for data in solver_info:
                                    self.gnt.broken_barh([(data[0], 0.5)], [position[idx] - 0.5, 1],
                                                         facecolors=colors[idx])
                        else:
                            # for idx, solver_info in enumerate(stat):
                            for data in stat[0]:
                                self.gnt.broken_barh([(data['current_time'], 0.5)], [position[0] - 0.5, 1],
                                                     facecolors=colors[0])
                    except TypeError:
                        pass
                else:
                    self.set_plot_param(title[i], gs[i, :], lim=horizon)
            else:
                self.set_plot_param(title[i], gs[i, :], lim=horizon)

            if local_data[i]['Robot'][0]['start'] is None:
                continue

            for agent in local_data[i]:
                for task in local_data[i][agent]:
                    position_y, task_name_y, action_y = self.y_pos_and_text[task["universal"]][agent]

                    # if self.from_file:
                    # if not video:
                    if task['universal']:
                        color = ['paleturquoise', 'turquoise', 'lightseagreen']
                    else:
                        if 'Human' in task['agent']:
                            color = ['thistle', '#a993e3', 'mediumslateblue'] # mediumpurple'
                        else:
                            color = ['lightsteelblue', 'cornflowerblue', 'royalblue']
                    # else:
                    #     color = [self.color[task['state']], self.color[task['state']], self.color[task['state']]]

                    if video and task['state'] == 1 and task['finish'][0] < self.current_time and i != 0:
                        task['finish'][0] = self.current_time
                        self.gnt.broken_barh([(task['start'], task['finish'][0]-task['start'])], [position_y - 1.2, 2.4],
                                             facecolors=color[0])
                    else:
                        preps_duration = task['finish'][0] - task['start'] - task['finish'][2] - task['finish'][3]
                        self.gnt.broken_barh([(task['start'], preps_duration)], [position_y - 1.2, 2.4],
                                             facecolors=color[0])
                        self.gnt.broken_barh([(task['start'] + preps_duration, task['finish'][2])],
                                             [position_y - 1.2, 2.4], facecolors=color[1])
                        self.gnt.broken_barh([(task['start'] + preps_duration + task['finish'][2], task["finish"][3])],
                                             [position_y - 1.2, 2.4], facecolors=color[2])
                        self.gnt.annotate("", xy=((task['finish'][0]), position_y - 1.3),
                                          xytext=((task['finish'][0]), position_y + 1.3),
                                          arrowprops=dict(arrowstyle="-", lw=1, color="black"))

                    self.gnt.text(task["start"] + 0.5, task_name_y, task['id'], fontsize=9,
                                    rotation='horizontal')

                    # else:
                    #     duration = task['finish'][0] - task['start']
                    #     color = self.color[task["state"]]
                    #     self.gnt.broken_barh([(task["start"], duration - 0.2)], [position_y - 1.2, 2.4],
                    #                          facecolors=color)
                    #     self.gnt.broken_barh([(task['start'] + duration - 0.2, 0.2)], [position_y - 1.2, 2.4],
                    #                          facecolors='black')

                    #     self.gnt.text(task["start"] + 0.5, task_name_y, task['id'], fontsize=9,
                    #                   rotation='horizontal')

            if video and i != 0:
                self.gnt.annotate("", xy=(self.current_time, 0), xytext=(self.current_time, 13),
                                  arrowprops=dict(arrowstyle="-", lw=2, color="red"))

        # if not video:
        try:
            if 'comparison' in file_name.stem:
                self.plot_dependency_graph(local_data[1], gs=gs[3:, :-1])
            else:
                self.plot_dependency_graph(local_data[1], gs=gs[2:, :-1], video=video)
        except IndexError:
            # pass
            self.plot_dependency_graph(local_data[0], gs=gs[1, :-1], video=video)

        # ------ create the legend
        plt.tight_layout()
        plt.legend(self.h, self.l, loc='center left', bbox_to_anchor=(1.35, 0.5), fontsize="15",
                    handler_map={MulticolorPatch: MulticolorPatchHandler()})
        if file_name:
            if '.' in file_name.__str__():
                plt.savefig(file_name)
                if not video:
                    logging.info(f'The plot was saved to {file_name}')
            else:
                plt.savefig(f'{file_name}.svg')
                logging.info(f'The plot was saved to {file_name}.svg')
        else:
            plt.show()

        plt.close()


    def online_plotting(self):
        data = pd.DataFrame({
            "state": ["Completed", "In progress", "Available", "Non available", "Completed",
                      "In progress", "Available"],
            "start": [0, 7, 14, 26, 0, 10, 17],
            "End": [7, 14, 23, 30, 10, 17, 26],
            "Agent": ["Human", "Human", "Human", "Human", "Robot", "Robot", "Robot"]
        })
        bar_chart = alt.Chart(data).mark_bar().encode(
            y="Agent:N",
            # x="sum(Time):O",
            x=alt.X('start:Q', title='Time'),
            x2='End:Q',
            color=alt.Color('state:N', title='state',
                            scale=alt.Scale(
                                domain=['Completed', 'In progress', 'Available', 'Non available'],
                                range=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
                            ))).properties(
            title='Gantt Chart',
            width=max(self.data['End'])
        )
        current_time_rule = alt.Chart(pd.DataFrame({'current_time': [self.current_time]})).mark_rule(
            color='red').encode(
            x='current_time',
            size=alt.value(2)
        )

        self.chart_placeholder.altair_chart(bar_chart + current_time_rule, use_container_width=True)

    def init_online_plotting(self):
        self.chart_placeholder = st.empty()
        # "Energy Costs By Month"

        # if st.button('say hello'):
        #     st.write('Why hello there')
        # else:
        #     st.write('Goodbye')
        # progress_bar.empty()

        # streamlit widgets automatically run the script from top to bottom. since
        # this button is not connected to any other logic, it just causes a plain
        # rerun.
        # st.button("Re-run")

    def save_data(self):
        try:
            with open(self.data4video, "r+") as json_file:
                data = json.load(json_file)
        except FileNotFoundError:
            self.data4video = './..' + self.data4video[1:]
            with open(self.data4video, "r+") as json_file:
                data = json.load(json_file)
        except Exception:
            data = {}

        data[len(data)] = {'Time': self.current_time, 'Schedule': self.data[1]}
        with open(self.data4video, 'w') as f:
            json.dump(data, f, indent=4)

    def plot_dependency_graph(self, local_data, gs, video=False):
        sub2 = self.fig.add_subplot(gs)
        sub2.set_title("Dependency graph")
        axis = plt.gca()
        # maybe smaller factors work as well, but 1.1 works fine for this minimal example
        axis.set_xlim([1.1 * x for x in axis.get_xlim()])
        axis.set_ylim([1.1 * y for y in axis.get_ylim()])

        G = nx.DiGraph()
        labels = {}
        states = {None: [], -1: [], 0: [], 1: [], 2: []}
        color_map = {'Human': 'rebeccapurple', 'Robot': 'royalblue', 'Universal': 'lightseagreen'}
        colors = {'mediumslateblue': [], 'royalblue': [], 'lightseagreen': []}
        allocability = {True: [], False: []}
        task_number = 0
        for agent in local_data:
            for task in local_data[agent]:
                G.add_node(task["action"]["place"])
                states[task["state"]].append(task["action"]["place"])
                labels[task["action"]["place"]] = task['id']
                allocability[task['universal']].append(task['action']['place'])
                if task['universal']:
                    colors['lightseagreen'].append(task['action']['place'])
                else:
                    if 'Human' in task['agent']:
                        colors['mediumslateblue'].append(task['action']['place'])
                    else:
                        colors['royalblue'].append(task['action']['place'])

                task_number += 1
        for agent in local_data:
            for task in local_data[agent]:
                if task["conditions"]:
                    for j in task["conditions"]:
                        G.add_edges_from([(get_task_from_id(j, local_data), task["action"]["place"])])

        if task_number == 16:
            pos = { '0': (0, 3), '4': (1, 3), '8': (2, 3), '12': (3, 3),
                    '1': (0, 2), '5': (1, 2), '9': (2, 2), '13': (3, 2),
                    '2': (0, 1), '6': (1, 1), '10': (2, 1), '14': (3, 1),
                    '3': (0, 0), '7': (1, 0), '11': (2, 0), '15': (3, 0)} # positions for all nodes
        elif task_number == 7:
            # for case 7
            pos = {'1': (0.5, 2.5), '0': (1.5, 2.5), '2': (2.5, 2.5),
                   "3": (1, 1.5), '8': (1.5, 1.5), '4': (2, 1.5),
                   "5": (1, 0.5), '7': (1.5, 0.5), '6': (2, 0.5)}
        else:
            # TODO: fix graph plotting
            nx.draw(G)
            axis = plt.gca()
            # maybe smaller factors work as well, but 1.1 works fine for this minimal example
            axis.set_xlim([-0.5, 3.5])
            axis.set_ylim([-0.5, 3.5])
            return None
        node_size = 1000
        linewidths = 5
        nx.draw_networkx_edges(G, pos, width=1.7, alpha=0.7, node_size=node_size)
        nx.draw_networkx_labels(G, pos, labels, font_size=14, font_color="whitesmoke")

        # if not video:
        #     nx.draw_networkx_nodes(G, pos, nodelist=allocability[False], node_color='royalblue', node_size=node_size)
        #     nx.draw_networkx_nodes(G, pos, nodelist=allocability[True], node_color='lightseagreen', node_size=node_size)
        # else:
            # for node in G.nodes:
            #     node.
            # nx.draw_networkx_nodes(G, pos, nodelist=colors['royalblue'], node_color='royalblue', node_size=node_size)
            # nx.draw_networkx_nodes(G, pos, nodelist=colors['violet'], node_color='violet', node_size=node_size)
            # nx.draw_networkx_nodes(G, pos, nodelist=colors['lightseagreen'], node_color='lightseagreen', node_size=node_size)

            # node_color = ["lightcoral", "lightcoral", "gold", "lightgreen", "silver"]
        state_color_map = {None: "lightcoral", -1: "lightcoral", 0: "gold", 1: "lightgreen", 2: "silver"}
        for color in colors.keys():
            for state in states.keys():
                nx.draw_networkx_nodes(G, pos, nodelist=list(set(colors[color]) & set(states[state])),
                                       node_color=color, edgecolors=state_color_map[state],
                                       node_size=node_size, linewidths=linewidths)


            # nx.draw_networkx_nodes(G, pos, nodelist=list(set(colors['violet']) & set(state[None])), node_color='violet',
            #                        edgecolors=node_color[0], node_size=node_size, linewidths=linewidths)
            # nx.draw_networkx_nodes(G, pos, nodelist=list(set(colors['violet']) & set(state[None])), node_color='violet',
            #                        edgecolors=node_color[0], node_size=node_size, linewidths=linewidths)
            # nx.draw_networkx_nodes(G, pos, nodelist=list(set(colors['violet']) & set(state[None])), node_color='violet',
            #                        edgecolors=node_color[0], node_size=node_size, linewidths=linewidths)
            # nx.draw_networkx_nodes(G, pos, nodelist=list(set(colors['violet']) & set(state[None])), node_color='violet',
            #                        edgecolors=node_color[0], node_size=node_size, linewidths=linewidths)
            # nx.draw_networkx_nodes(G, pos, nodelist=list(set(colors['violet']) & set(state[None])), node_color='violet',
            #                        edgecolors=node_color[0], node_size=node_size, linewidths=linewidths)
            # nx.draw_networkx_nodes(G, pos, nodelist=state[-1], edgecolors=node_color[1], node_size=node_size, linewidths=linewidths)
            # nx.draw_networkx_nodes(G, pos, nodelist=state[0], edgecolors=node_color[2], node_size=node_size, linewidths=linewidths)
            # nx.draw_networkx_nodes(G, pos, nodelist=state[1], edgecolors=node_color[3], node_size=node_size, linewidths=linewidths)
            # nx.draw_networkx_nodes(G, pos, nodelist=state[2], edgecolors=node_color[4], node_size=node_size, linewidths=linewidths)

        axis = plt.gca()
        # maybe smaller factors work as well, but 1.1 works fine for this minimal example
        axis.set_xlim([-0.5, 3.5])
        axis.set_ylim([-0.5, 3.5])
