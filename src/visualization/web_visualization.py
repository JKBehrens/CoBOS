import streamlit as st
import altair as alt
import pandas as pd
from graphviz import Digraph

class Web_vis:

    def __init__(self, data=None):
        self.robot_text = st.empty()
        self.robot_text.text("Robot: ")
        self.human_text = st.empty()
        self.human_text.text("Human: ")
        # Create an empty placeholder for the chart
        self.chart_placeholder = st.empty()
        # Create a directed graph
        self.graph_placeholder = st.empty()
        self.current_time = 0
        self.data = data
        # Mapping dictionary for Status values
        self.status_mapping = {
            2: "Completed",
            1: "In progress",
            0: "Available",
            -1: "Not available"
        }

        # Color mapping for Status
        self.status_colors = {
            'Completed': '#1f77b4',
            'In progress': '#2ca02c',
            'Available': '#ff7f0e',
            'Not available': '#d62728'
        }

    def update_agent(self, row):
        if row['Universal']:
            return f"Allacatable for {row['Agent']}"
        else:
            return row['Agent']

    def update_gantt_chart(self):
        pandas_data = pd.DataFrame({
            "Status": self.data['Status'],
            "Start": self.data['Start'],
            "End": self.data['End'],
            "Agent": self.data['Agent'],
            "ID": self.data['ID'],
            "Universal": self.data["Universal"]
        })

        pandas_data["Agent"] = pandas_data.apply(self.update_agent, axis=1)
        pandas_data["Status"] = pandas_data["Status"].map(self.status_mapping)
        bar_chart = alt.Chart(pandas_data).mark_bar().encode(
            y=alt.X('Agent:N', title='Agents'),
            x=alt.X('Start:Q', title='Time [s]'),
            x2='End:Q',
            color=alt.Color('Status:N', title='Status',
                            scale=alt.Scale(
                                domain=['Completed', 'In progress', 'Available', 'Not available'],
                                range=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
                            )))

        tick = alt.Chart(pandas_data).mark_tick(
            color='black',
            thickness=1,
            size=15,  # controls width of tick.
        ).encode(
            x='End:Q',
            y='Agent:N'
        )

        text = alt.Chart(pandas_data).mark_text(dx=7, dy=0, color='white').encode(
            x='Start',
            y='Agent',
            # detail='site:N',
            text=alt.Text('ID')
        )

        current_time_rule = alt.Chart(pd.DataFrame({'current_time': [self.current_time]})).mark_rule(
            color='red').encode(
            x='current_time',
            size=alt.value(2)
        )

        try:
            self.chart_placeholder.altair_chart(bar_chart + current_time_rule + tick + text, use_container_width=True)
        except RuntimeError:
            pass

    def update_dependency_graph(self):
        pandas_data = pd.DataFrame({
            "Status": self.data['Status'],
            "ID": self.data['ID'],
            "Conditions": self.data['Conditions']})
        pandas_data["Status"] = pandas_data["Status"].map(self.status_mapping)
        graph = Digraph('Dependency Graph')
        # Add nodes to the graph with colors based on 'Status'
        for index, row in pandas_data.iterrows():
            node_color = self.status_colors.get(row["Status"], "gray")
            graph.node(str(row["ID"]), color=node_color, style="filled")

        # Add edges between tasks based on 'Conditions'
        for index, row in pandas_data.iterrows():
            for condition in row["Conditions"]:
                graph.edge(str(condition), str(row["ID"]))

        # Display the Graphviz graph using Streamlit
        self.graph_placeholder.graphviz_chart(graph)

    def update_info(self, agent, start=False):
        if start:
            string = f"{agent.name}: place {agent.current_task.action['Object']} to {agent.current_task.action['Place']}.   " \
                     f"Task ID: {agent.current_task.id}"
        else:
            string = f"{agent.name}: waiting for task"
        if agent.name == 'Robot':
            self.robot_text.text(string)
        else:
            self.human_text.text(string)
