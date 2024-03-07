"""
    Visualization class for Gantt chart and dependency graph

    @author: Marina Ionova, student of Cybernetics and Robotics at the CTu in Prague
    @contact: marina.ionova@cvut.cz
"""
import pandas as pd
from matplotlib import pyplot as plt
import json
import os
import networkx as nx
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
import pandas
import streamlit as st
import numpy as np
import altair as alt


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
        self.fig = plt.figure(figsize=(12, 10))
        self.data4video = './visualization/data_for_visualization/sim_2_video.json'
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
        except FileNotfoundError as e:
            self.data4video = './..' + self.data4video[1:]
            with open(self.data4video, "w") as json_file:
                json.dump({}, json_file)

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
            self.gnt.set_ylim(0, 16.5)
            # setting ticks on y-axis
            self.gnt.set_yticks([1.5, 4.5, 7.5, 10.5, 14, 15.5])
            # Labelling tickes of y-axis
            self.gnt.set_yticklabels(
                ['Robot', 'Allocatable\n for robot', 'Allocatable\n for human',
                 'Human', 'Rescheduling', 'Evaluation'])

        # setting X-axis limits
        if lim:
            self.gnt.set_xlim(0, lim + 5)

        # setting labels for x-axis and y-axis
        self.gnt.set_xlabel('Time [s]')

        # setting graph attribute
        self.gnt.grid(True)

        # ------ choose some colors
        colors1 = ['royalblue']  # 'lightsteelblue', 'cornflowerblue',
        colors2 = ['lightseagreen']  # 'paleturquoise', 'turquoise',
        colors5 = ['royalblue', 'lightseagreen']
        colors4 = ['cornflowerblue', 'turquoise']
        colors3 = ['lightsteelblue', 'paleturquoise']

        # ------ get the legend-entries that are already attached to the axis
        self.h, self.l = self.gnt.get_legend_handles_labels()

        # ------ append the multicolor legend patches
        self.h.append(MulticolorPatch(colors1))
        self.l.append("Non-allocatable")

        self.h.append(MulticolorPatch(colors2))
        self.l.append("Allocatable")

        self.h.append(MulticolorPatch(colors3))
        self.l.append("Preparation")

        self.h.append(MulticolorPatch(colors4))
        self.l.append("Execution")

        self.h.append(MulticolorPatch(colors5))
        self.l.append("Completion")

        # if reschedule_num:
        #     colors6 = ['blue']
        #     colors7 = ['red']
        #
        #     self.h.append(MulticolorPatch(colors6))
        #     self.l.append("Possible rescheduling")
        #     self.h.append(MulticolorPatch(colors7))
        #     self.l.append("Main rescheduling")

        self.labels = []
        self.labels.append(mpatches.Patch(color='lightcoral', label='Not available'))
        self.labels.append(mpatches.Patch(color='gold', label='Available'))
        self.labels.append(mpatches.Patch(color='lightgreen', label='In process'))
        self.labels.append(mpatches.Patch(color='silver', label='Completed'))

    def set_horizon(self, data):
        # flatten the nested dictionary to get all "finish" values
        finish_time = 0
        for schedule in data:
            for agent in schedule:
                for task in schedule[agent]:
                    if task["finish"][0] > finish_time:
                        finish_time = task['finish'][0]

        # find the maximum "finish" time
        return finish_time

    def plot_schedule(self, file_name='', video=False):
        if 'simulation' in file_name:
            title = ['Gantt Chart: initial', 'Gantt Chart: final']
            gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 2])
            positions = [[311, 312], [313]]
            local_data = self.data
        elif 'comparison' in file_name:
            title = ['Gantt Chart: initial', 'Gantt Chart: final (same sampling seed)',
                     'Gantt Chart: final (different sampling seed)']
            gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 2])
            positions = [[311, 312, 313], [314]]
            local_data = self.data['schedule']
        else:
            title = ['Gantt Chart']
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
            positions = [[211], [212]]
            local_data = [self.data]
        horizon = self.set_horizon(local_data)
        for i, position in enumerate(positions[0]):
            if 'comparison' in file_name:
                if i != 0:
                    self.set_plot_param(title[i], gs[i, :], lim=horizon, reschedule_num=True)
                    colors = {0: 'red', 1: 'blue'}
                    position = {0: 14, 1: 15.5}
                    try:
                        for idx, solver_info in enumerate(self.data['statistics'][i - 1]['solver']):
                            for data in solver_info:
                                self.gnt.broken_barh([(data[0], 0.5)], [position[idx] - 0.5, 1],
                                                     facecolors=colors[idx])
                    except TypeError:
                        pass
                else:
                    self.set_plot_param(title[i], gs[i, :], lim=horizon)
            else:
                self.set_plot_param(title[i], gs[i, :], lim=horizon)

            for agent in local_data[i]:
                for task in local_data[i][agent]:
                    position_y, task_name_y, action_y = self.y_pos_and_text[task["universal"]][agent]

                    if self.from_file:
                        if task['universal']:
                            color = ['paleturquoise', 'turquoise', 'lightseagreen']
                        else:
                            color = ['lightsteelblue', 'cornflowerblue', 'royalblue']

                        self.gnt.text(task["start"] + 0.5, task_name_y, task['id'], fontsize=9,
                                      rotation='horizontal')
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
                    else:
                        duration = task['finish'][0] - task['start']
                        color = self.color[task["state"]]
                        self.gnt.broken_barh([(task["start"], duration - 0.2)], [position_y - 1.2, 2.4],
                                             facecolors=color)
                        self.gnt.broken_barh([(task['start'] + duration - 0.2, 0.2)], [position_y - 1.2, 2.4],
                                             facecolors='black')

                        self.gnt.text(task["start"] + 0.5, task_name_y, task['id'], fontsize=9,
                                      rotation='horizontal')

            if not self.from_file:
                self.gnt.annotate("", xy=(self.current_time, 0), xytext=(self.current_time, 13),
                                  arrowprops=dict(arrowstyle="-", lw=2, color="red"))

        try:
            if 'comparison' in file_name:
                self.plot_dependency_graph(local_data[0], gs=gs[3:, :-1])
            else:
                self.plot_dependency_graph(local_data[0], gs=gs[2:, :-1])
        except IndexError:
            # pass
            self.plot_dependency_graph(local_data[0], gs=gs[1, :-1])

        # ------ create the legend

        plt.tight_layout()
        if self.from_file:
            plt.legend(self.h, self.l, loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize="15",
                       handler_map={MulticolorPatch: MulticolorPatchHandler()})
            if file_name:
                plt.savefig('./img/' + file_name)
            # plt.show()
        else:
            plt.legend(handles=self.labels, loc='center left', bbox_to_anchor=(1, 0.5),  # bbox_to_anchor=(0.5, -0.05),
                       fancybox=True, shadow=True, ncol=5)
            if not video:
                plt.show()

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
        except FileNotFoundError as e:
            self.data4video = './..' + self.data4video[1:]
            with open(self.data4video, "r+") as json_file:
                data = json.load(json_file)
        except Exception as e:
            data = {}

        data[len(data)] = {'Time': self.current_time, 'Schedule': self.data}
        with open(self.data4video, 'w') as f:
            json.dump(data, f, indent=4)

    def plot_dependency_graph(self, local_data, gs):
        sub2 = self.fig.add_subplot(gs)
        sub2.set_title("Dependency graph")
        axis = plt.gca()
        # maybe smaller factors work as well, but 1.1 works fine for this minimal example
        axis.set_xlim([1.1 * x for x in axis.get_xlim()])
        axis.set_ylim([1.1 * y for y in axis.get_ylim()])

        G = nx.DiGraph()
        labels = {}
        state = {None: [], -1: [], 0: [], 1: [], 2: []}
        allocability = {True: [], False: []}
        task_number = 0
        for agent in local_data:
            for task in local_data[agent]:
                G.add_node(task["action"]["place"])
                state[task["state"]].append(task["action"]["place"])
                labels[task["action"]["place"]] = task['id']
                allocability[task['universal']].append(task['action']['place'])
                task_number += 1
        for agent in local_data:
            for task in local_data[agent]:
                if task["conditions"]:
                    for j in task["conditions"]:
                        G.add_edges_from([(get_task_from_id(j, local_data), task["action"]["place"])])

        if task_number == 16:
            pos = {'A1': (0, 3), 'B1': (1, 3), 'C1': (2, 3), 'D1': (3, 3),
                   "A2": (0, 2), 'B2': (1, 2), 'C2': (2, 2), 'D2': (3, 2),
                   "A3": (0, 1), 'B3': (1, 1), 'C3': (2, 1), 'D3': (3, 1),
                   "A4": (0, 0), 'B4': (1, 0), 'C4': (2, 0), 'D4': (3, 0)}  # positions for all nodes
        elif task_number == 8:
            # for case 7
            pos = {'A2': (0.5, 2.5), 'A1': (1.5, 2.5), 'A3': (2.5, 2.5),
                   "A4": (0.5, 1.5), 'C1': (1.5, 1.5), 'B1': (2.5, 1.5),
                   "B2": (0.5, 0.5), 'B4': (1.5, 0.5), 'B3': (2.5, 0.5)}
        else:
            # TODO: fix graph plotting
            nx.draw(G)
            axis = plt.gca()
            # maybe smaller factors work as well, but 1.1 works fine for this minimal example
            axis.set_xlim([-0.5, 3.5])
            axis.set_ylim([-0.5, 3.5])
            return None
        node_size = 900
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, node_size=node_size)
        nx.draw_networkx_labels(G, pos, labels, font_size=14, font_color="whitesmoke")

        if self.from_file:
            nx.draw_networkx_nodes(G, pos, nodelist=allocability[False], node_color='royalblue', node_size=node_size)
            nx.draw_networkx_nodes(G, pos, nodelist=allocability[True], node_color='lightseagreen', node_size=node_size)
        else:
            node_color = ["lightcoral", "lightcoral", "gold", "lightgreen", "silver"]
            nx.draw_networkx_nodes(G, pos, nodelist=state[None], node_color=node_color[0], node_size=node_size)
            nx.draw_networkx_nodes(G, pos, nodelist=state[-1], node_color=node_color[1], node_size=node_size)
            nx.draw_networkx_nodes(G, pos, nodelist=state[0], node_color=node_color[2], node_size=node_size)
            nx.draw_networkx_nodes(G, pos, nodelist=state[1], node_color=node_color[3], node_size=node_size)
            nx.draw_networkx_nodes(G, pos, nodelist=state[2], node_color=node_color[4], node_size=node_size)

        axis = plt.gca()
        # maybe smaller factors work as well, but 1.1 works fine for this minimal example
        axis.set_xlim([-0.5, 3.5])
        axis.set_ylim([-0.5, 3.5])
