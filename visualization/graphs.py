"""
    Visualization class for Gantt chart and dependency graph

    @author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
    @contact: marina.ionova@cvut.cz
"""
import pandas as pd
from matplotlib import pyplot as plt
import json
import os
import networkx as nx
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import pandas
import streamlit as st
import numpy as np
from simulation.sim import set_task_time
import altair as alt


# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self,  legend, orig_handle, fontsize, handlebox):
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


def get_task_from_id(ID, data):
    for agent in data:
        for task in data[agent]:
            if task['ID'] == ID:
                return task["Action"]['Place']
    return None


class Vis:
    def __init__(self, horizon=None, data=None, from_file=False):
        self.fig = plt.figure(figsize=(12, 8))
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
            os.remove(self.data4video)
        except Exception as e:
            pass

    def set_plot_param(self, position, title):
        self.gnt = self.fig.add_subplot(position)  # 211
        self.gnt.set_title(title)

        # Setting Y-axis limits
        self.gnt.set_ylim(0, 13)

        # Setting X-axis limits
        if self.horizon:
            self.gnt.set_xlim(0, self.horizon + 50)
        self.gnt.set_xlim(0,150)

        # Setting labels for x-axis and y-axis
        self.gnt.set_xlabel('Time [s]')

        # Setting ticks on y-axis
        self.gnt.set_yticks([1.5, 4.5, 7.5, 10.5])

        # Labelling tickes of y-axis
        self.gnt.set_yticklabels(
            ['Robot', 'Allocatable\n assigned\n to robot', 'Allocatable\n assigned\n to human', 'Human'])

        # Setting graph attribute
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

        self.labels = []
        self.labels.append(mpatches.Patch(color='lightcoral', label='Not available'))
        self.labels.append(mpatches.Patch(color='gold', label='Available'))
        self.labels.append(mpatches.Patch(color='lightgreen', label='In process'))
        self.labels.append(mpatches.Patch(color='silver', label='Completed'))

    def plot_schedule(self, file_name=''):
        if file_name == 'simulation.png':
            title = ['Gantt Chart: initial', 'Gantt Chart: final']
            positions = [[311, 312], [313]]
            index_offset = 0
        else:
            title = ['Gantt Chart']
            positions = [[211], [212]]
            index_offset = 0
        local_data = self.data if self.from_file and file_name == 'simulation.png' else [self.data]
        for i, position in enumerate(positions[0]):
            self.set_plot_param(position, title[i])  # [0]
            for agent in local_data[i + index_offset]:
                for task in local_data[i + index_offset][agent]:
                    position_y, task_name_y, action_y = self.y_pos_and_text[task["Universal"]][agent]

                    if isinstance(task['Finish'], int):
                        actions = set_task_time(task, seed=0)
                        duration = task['Finish'] - task['Start']
                        preps_end = task['Start'] + actions[1]
                        execution_end = task['Start'] + actions[1] + actions[2]
                    else:
                        duration = task['Finish'][0] - task['Start']
                        preps_end = task['Finish'][0] - task['Finish'][2] - task['Finish'][3]
                        execution_end = task['Finish'][0] - task['Finish'][3]
                        actions = [duration,
                                   duration - task['Finish'][2] - task['Finish'][3], task['Finish'][2],
                                   task['Finish'][3]]

                    if self.from_file:
                        if task['Universal']:
                            color = ['paleturquoise', 'turquoise', 'lightseagreen']
                        else:
                            color = ['lightsteelblue', 'cornflowerblue', 'royalblue']

                        self.gnt.text(task["Start"] + 0.5, task_name_y, task["Action"]['Object'], fontsize=9,
                                      rotation='horizontal')
                        self.gnt.broken_barh([(task["Start"], actions[1])], [position_y - 1.2, 2.4],
                                             facecolors=color[0])
                        self.gnt.broken_barh([(preps_end, actions[2])], [position_y - 1.2, 2.4],
                                             facecolors=color[1])
                        self.gnt.broken_barh([(execution_end, actions[3])], [position_y - 1.2, 2.4],
                                             facecolors=color[2])
                        self.gnt.annotate("", xy=((execution_end + actions[3]), position_y - 1.3),
                                          xytext=((execution_end + actions[3]), position_y + 1.3),
                                          arrowprops=dict(arrowstyle="-", lw=1, color="black"))
                    else:
                        color = self.color[task["Status"]]
                        self.gnt.broken_barh([(task["Start"], duration - 0.2)], [position_y - 1.2, 2.4],
                                             facecolors=color)
                        self.gnt.broken_barh([(task['Start'] + duration - 0.2, 0.2)], [position_y - 1.2, 2.4],
                                             facecolors='black')

                        self.gnt.text(task["Start"] + 0.5, task_name_y, task["Action"]['Object'], fontsize=9,
                                      rotation='horizontal')

            self.gnt.annotate("", xy=(self.current_time, 0), xytext=(self.current_time, 13),
                              arrowprops=dict(arrowstyle="-", lw=2, color="red"))

        self.plot_dependency_graph(local_data[0], positions[1][0])
        # ------ create the legend

        plt.tight_layout()
        if self.from_file:
            plt.legend(self.h, self.l, loc='upper right',
                       handler_map={MulticolorPatch: MulticolorPatchHandler()})
            if file_name:
                plt.savefig('./img/' + file_name)
            # plt.show()
        else:
            plt.legend(handles=self.labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                        fancybox=True, shadow=True, ncol=5)
            plt.show()

    def online_plotting(self):
        data = pd.DataFrame({
            "Status": ["Completed", "In progress", "Available", "Non available", "Completed",
                       "In progress", "Available"],
            "Start": [0, 7, 14, 26, 0, 10, 17],
            "End": [7, 14, 23, 30, 10, 17, 26],
            "Agent": ["Human", "Human", "Human", "Human", "Robot", "Robot", "Robot"]
        })
        bar_chart = alt.Chart(data).mark_bar().encode(
            y="Agent:N",
            # x="sum(Time):O",
            x=alt.X('Start:Q', title='Time'),
            x2='End:Q',
            color=alt.Color('Status:N', title='Status',
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


        # if st.button('Say hello'):
        #     st.write('Why hello there')
        # else:
        #     st.write('Goodbye')
                # progress_bar.empty()

        # Streamlit widgets automatically run the script from top to bottom. Since
        # this button is not connected to any other logic, it just causes a plain
        # rerun.
        # st.button("Re-run")

    def save_data(self):
        try:
            with open(self.data4video, "r+") as json_file:
                data = json.load(json_file)
        except Exception as e:
            data = {}

        data[len(data)] = {'Time': self.current_time, 'Schedule': self.data}
        with open(self.data4video, 'w') as f:
            json.dump(data, f, indent=4)

    def plot_dependency_graph(self, local_data, position):
        sub2 = self.fig.add_subplot(position)
        sub2.set_title("Dependency graph")
        G = nx.DiGraph()
        labels = {}
        status = {None: [], -1: [], 0: [], 1: [], 2: []}
        allocability = {True: [], False: []}
        for agent in local_data:
            for task in local_data[agent]:
                G.add_node(task["Action"]["Place"])
                status[task["Status"]].append(task["Action"]["Place"])
                labels[task["Action"]["Place"]] = task["Action"]['Object']
                allocability[task['Universal']].append(task['Action']['Place'])
        for agent in local_data:
            for task in local_data[agent]:
                if task["Conditions"]:
                    for j in task["Conditions"]:
                        G.add_edges_from([(get_task_from_id(j, local_data), task["Action"]["Place"])])

        pos = {'A1': (0, 3), 'B1': (1, 3), 'C1': (2, 3), 'D1': (3, 3),
               "A2": (0, 2), 'B2': (1, 2), 'C2': (2, 2), 'D2': (3, 2),
               "A3": (0, 1), 'B3': (1, 1), 'C3': (2, 1), 'D3': (3, 1),
               "A4": (0, 0), 'B4': (1, 0), 'C4': (2, 0), 'D4': (3, 0)}  # positions for all nodes
        # pos = {'A1': (0, 0), 'B1': (0, 2), 'C1': (1, 0), 'D1': (1, 2),
        #        "A2": (0, 1), 'C2': (2, 0), 'D2': (2, 2),
        #        'C3': (3, 0.5), 'D3': (3, 1.5)}  # positions for all nodes

        node_size = 800
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, node_size=node_size)
        nx.draw_networkx_labels(G, pos, labels, font_size=14, font_color="whitesmoke")

        if self.from_file:
            nx.draw_networkx_nodes(G, pos, nodelist=allocability[False], node_color='royalblue', node_size=node_size)
            nx.draw_networkx_nodes(G, pos, nodelist=allocability[True], node_color='lightseagreen', node_size=node_size)
        else:
            node_color = ["lightcoral", "lightcoral", "gold", "lightgreen", "silver"]
            nx.draw_networkx_nodes(G, pos, nodelist=status[None], node_color=node_color[0], node_size=node_size)
            nx.draw_networkx_nodes(G, pos, nodelist=status[-1], node_color=node_color[1], node_size=node_size)
            nx.draw_networkx_nodes(G, pos, nodelist=status[0], node_color=node_color[2], node_size=node_size)
            nx.draw_networkx_nodes(G, pos, nodelist=status[1], node_color=node_color[3], node_size=node_size)
            nx.draw_networkx_nodes(G, pos, nodelist=status[2], node_color=node_color[4], node_size=node_size)
