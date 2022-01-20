import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle

from greedy_mixed_graph.generate import SimpleGraph


def plotGraph(graph, save=False, fname='graph.png'):
    if isinstance(graph, SimpleGraph):
        mg = graph.mg
    elif isinstance(graph, np.ndarray):
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("G must be a square matrix!")
        mg = graph
    else:
        raise ValueError("G must be a graph object or a matrix!")
    mg = np.abs(mg)
    np.fill_diagonal(mg, 0)
    mg[(mg < 100) & (mg > 0)] = 1
    mg[mg >= 100] = 100

    G = nx.Graph()
    p = mg.shape[0]
    labels = {}
    for i in range(p):
        G.add_node(i)
        labels[i] = i

    edge_d = np.argwhere(mg == 1)
    edge_b = np.argwhere(np.triu(mg) == 100)
    G.add_edges_from(edge_d)
    G.add_edges_from(edge_b)

    fig = plt.gcf()
    ax = plt.gca()
    pos = nx.circular_layout(G, center=(0, 0))
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', linewidths=1.0)
    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color="black")
    style1 = ArrowStyle('->')
    style2 = ArrowStyle('<->')
    nx.draw_networkx_edges(G, pos, edgelist=edge_d, arrows=True, arrowstyle=style1, edge_color='blue')
    nx.draw_networkx_edges(G, pos, edgelist=edge_b, arrows=True, arrowstyle=style2, edge_color='red')

    if save:
        plt.savefig(fname)
    plt.show()


def plotGraphs(graphs, save=False, fname='graphs.png'):
    if not isinstance(graphs, list):
        raise TypeError("The value of graphs must be a list!")

    if len(graphs) > 4:
        raise ValueError("Currently more than 4 graphs in the same plot is not supported!")
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax = axes.flatten()
    style1 = ArrowStyle('->')
    style2 = ArrowStyle('<->')

    for j in range(4):
        if j < len(graphs):
            graph = graphs[j]
            if isinstance(graph, SimpleGraph):
                mg = graph.mg
            elif isinstance(graph, np.ndarray):
                if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
                    raise ValueError("G must be a square matrix!")
                mg = graph
            else:
                raise ValueError("G must be a graph object or a matrix!")
            mg = np.abs(mg)
            np.fill_diagonal(mg, 0)
            mg[(mg < 100) & (mg > 0)] = 1
            mg[mg >= 100] = 100

            G = nx.Graph()
            p = mg.shape[0]
            labels = {}
            for i in range(p):
                G.add_node(i)
                labels[i] = i

            edge_d = np.argwhere(mg == 1)
            edge_b = np.argwhere(np.triu(mg) == 100)
            G.add_edges_from(edge_d)
            G.add_edges_from(edge_b)

            pos = nx.circular_layout(G, center=(0, 0))
            nx.draw_networkx_nodes(G, pos, node_color='lightgray', linewidths=1.0, ax=ax[j])
            nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color="black", ax=ax[j])
            style1 = ArrowStyle('->')
            style2 = ArrowStyle('<->')
            nx.draw_networkx_edges(G, pos, edgelist=edge_d, arrows=True, arrowstyle=style1, edge_color='blue', ax=ax[j])
            nx.draw_networkx_edges(G, pos, edgelist=edge_b, arrows=True, arrowstyle=style2, edge_color='red', ax=ax[j])

        ax[j].set_axis_off()

    if save:
        plt.savefig(fname)
    plt.show()
