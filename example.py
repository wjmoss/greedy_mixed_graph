import numpy as np

from greedy_mixed_graph.plot import plotGraph, plotGraphs
from greedy_mixed_graph.greedysearch import GreedySearch
from greedy_mixed_graph.generate import SimpleGraph, ParamedSimpleGraph, generate_params, generate_mixed_graph


if __name__ == '__main__':

    # random seed
    np.random.seed(19260817)

    # generate 10 random mixed graphs
    graphs = generate_mixed_graph(p=8, n=11, max_in_degree=2)

    # generate parameters and data for the 1st graph, graphs[0]
    g = ParamedSimpleGraph(graphs[0])
    params = generate_params(L=g.L, O=g.O)
    g.assignParams(params[0], params[1])
    g.generateData(1000)

    # construct a GreedySearch object, run greedy search 10 times from the 10 mixed graphs as the start point
    GS = GreedySearch(mg_start=graphs[1:], n=1000, cov_mat=np.cov(g.data.T), edge_penalty=1, bic='ext_bic')
    result = GS.greedy_search(n_restarts=10, mc_cores=2)

    # plot the true graph and the estimated graph
    plotGraph(g.mg)
    plotGraph(result.final_graph.mg)
    plotGraphs([g.mg, result.final_graph.mg])
