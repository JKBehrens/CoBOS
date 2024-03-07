import streamlit as st

import pandas as pd
import altair as alt
import time
from graphviz import Digraph
from datetime import datetime

# Sample data
data = pd.DataFrame({
    "ID": [0, 1, 2, 3, 4, 5, 6],
    "Conditions": [[], [], [4], [1, 4], [], [0], [0, 2]],
    "Status": ["Completed", "In progress", "Available", "Non available", "Completed", "In progress", "Available"],
    "Start": [0, 7, 14, 23, 0, 10, 17],
    "End": [7, 14, 23, 30, 10, 17, 26],
    "Agent": ["Human", "Human", "Human", "Human", "Robot", "Robot", "Robot"]
})

# Define the initial current_time
current_time = 0


def graph(data):
    # Create a directed graph
    graph = Digraph('Dependency Graph')

    # Color mapping for Status
    status_colors = {
        'Completed': '#1f77b4',
        'In progress': '#2ca02c',
        'Available': '#ff7f0e',
        'Non available': '#d62728'
    }

    # Add nodes to the graph with colors based on 'Status'
    for index, row in data.iterrows():
        node_color = status_colors.get(row["Status"], "gray")
        graph.node(str(row["ID"]), color=node_color, style="filled")

    # Add edges between tasks based on 'Conditions'
    for index, row in data.iterrows():
        for condition in row["Conditions"]:
            graph.edge(str(condition), str(row["ID"]))

    # Display the Graphviz graph using Streamlit
    st.graphviz_chart(graph)


def encode(data, current_time):
    chart = alt.Chart(data).mark_bar().encode(
        y=alt.Y('Agent:N', title='Agent'),
        x=alt.X('Start:Q', title='Time'),
        x2='End:Q',
        color=alt.Color('Status:N', legend=alt.Legend(title='Status'), scale=alt.Scale(scheme='category20b')),
        tooltip=['Agent:N', 'Start:Q', 'End:Q', 'Status:N']
    ).properties(
        title='Gantt Chart of Task Status',
        width=600
    )

    # Add a vertical line for the current time
    current_time_rule = alt.Chart(pd.DataFrame({'current_time': [current_time]})).mark_rule(color='red').encode(
        x='current_time',
        size=alt.value(2)
    )

    return (chart + current_time_rule)


# Create a function to continuously update the current_time
def update_current_time():
    while True:
        global current_time
        current_time += 1
        time.sleep(1)  # Wait for 1 second before updating again


# Start a separate thread to continuously update the current_time
import threading

# update_thread = threading.Thread(target=update_current_time)
# update_thread.daemon = True
# update_thread.start()
# global current_time
# current_time = 5
# Create an empty placeholder for the chart
chart_placeholder = st.empty()
graph(data)
# while True:
#     # Create an Altair chart with the initial current_time
#     gantt_chart = encode(data, current_time)
#
#     # Display the chart (Note: You may need to run this code in a Jupyter Notebook or an appropriate environment to see the animation)
#     # gantt_chart.show()
#     chart_placeholder.altair_chart(gantt_chart, use_container_width=True)
#     current_time += 1
#     time.sleep(1)
