import networkx as networkx
import numpy as np
from utils.parameters.dataset_params import dataset_params


def graph_reader(dataset, relabel= True):
    graph = networkx.read_edgelist(dataset_params[dataset]['edge_path'], delimiter=dataset_params[dataset]['separator'])

    edge_types = {}
    for e in graph.edges:  # add edge types as an attribute
        edge_types[e] = {'type': ''.join(sorted(next(zip(*e))))}
    networkx.set_edge_attributes(graph, edge_types)

    if relabel:
        nodeList = np.array(list(graph))
        graph = networkx.relabel_nodes(graph, lambda x: np.where(nodeList == x)[0][0])  # giving nodes integer ids based on their first appearance

    # graph.remove_edges_from(networkx.selfloop_edges(graph))
    return graph


def nodes_to_idx(graph, nodes):
    nodeList = np.array(list(graph))
    return np.array([np.where(nodeList == x)[0][0] for x in nodes])





