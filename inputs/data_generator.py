from typing import Optional
import networkx as nx
from pydantic import BaseModel
import matplotlib.pyplot as plt
import numpy as np
import json


class RandomCase(BaseModel):
    agent_number: int
    task_number: int
    condition_number: int


class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.int64):
                return int(o)
            return super().default(o)
        

rand = np.random.default_rng()

def get_data_str_from_graph(g: nx.DiGraph, fn):
    print(nx.adjacency_matrix(G=g))

    adj = nx.to_dict_of_lists(g)
    agent_dict = nx.get_node_attributes(g, "agent")
    task_durations = nx.get_node_attributes(g, "task_duration")

    data = {"dependencies": adj,
            "task_assignment": agent_dict,
            "task_durations": task_durations}

    with open(fn, "w") as f:
        # yaml.dump(data=data, stream=f)
        json.dump(data, f, cls=EnhancedJSONEncoder, indent=4)


def add_durations_to_graph(g: nx.DiGraph, seed:Optional[int]=None):
    if seed is not None:
        rand = np.random.default_rng(seed=seed)
    durations = rand.normal(loc=20, scale=10, size=len(g.nodes())).round(decimals=0)
    
    durations = np.max([durations, 3*np.ones(len(durations))], axis=0).astype(int).tolist()

    keys = list(g.nodes())
    prop_dict = dict(zip(keys, durations))
    nx.set_node_attributes(G=g, values=prop_dict, name="task_duration")

    return g


def gen_task_graph_mixed_cross_task_dependencies(param: RandomCase, rand_gen:  np.random.Generator | None = None):
    graph_size = param.task_number

    G = nx.gnr_graph(graph_size, 0.1)
    for node in G.nodes():
        G.nodes[node]["agent"] = np.sort(rand_gen.choice([i for i in range(param.agent_number)],
                                                      size=rand.choice([i+1 for i in range(param.agent_number)]),
                                                      replace=False))

    connections = rand_gen.integers(0, graph_size, size=(2, param.condition_number))
    for i in np.arange(connections.shape[-1]):
        u, v = connections[:, i]
        print(u, v+graph_size)
        G.add_edge(u, v)

        cyc = nx.simple_cycles(G=G)

        pass

    while True:
        # Check if the graph is acyclic
        if nx.is_directed_acyclic_graph(G):
            print("The graph is acyclic.")
            break
        else:
            print("The graph contains cycles.")
            # Find a cycle in the graph
            cycle = nx.find_cycle(G)
            # Choose an edge from the cycle to delete and remove it from graphe
            edge_to_delete = rand_gen.choice(cycle)
            G.remove_edge(*edge_to_delete)

    nx.draw(G)   # default spring_layout

    plt.show()

    return G


def gen_task_graph_mixed():

    G1 = nx.gnr_graph(15, 0.1)
    for node in G1.nodes():
        G1.nodes[node]["agent"] = rand.choice([0, 1, 2])

    G2 = nx.gnr_graph(15, 0.2)
    for node in G2.nodes():
        G2.nodes[node]["agent"] = rand.choice([0, 1, 2])
        
    G = nx.disjoint_union(G1, G2)

    subax1 = plt.subplot(111)
    nx.draw(G)   # default spring_layout

    plt.show()

    return G

def gen_task_graph_simple():
    G1 = nx.gnr_graph(15, 0.1)
    for node in G1.nodes():
        G1.nodes[node]["agent"] = 1

    G2 = nx.gnr_graph(15, 0.2)
    for node in G2.nodes():
        G2.nodes[node]["agent"] = 2

    G = nx.disjoint_union(G1, G2)

    subax1 = plt.subplot(111)
    nx.draw(G)   # default spring_layout

    plt.show()

    return G
# seed = 0
# rand_gen = np.random.default_rng(seed)
#
# g = gen_task_graph_mixed_cross_task_dependencies(RandomCase(agent_number=4, task_number=15, condition_number=10), rand_gen)
# g = add_durations_to_graph(g, seed=seed)
# get_data_str_from_graph(g, "t2.json")


