# Greedy Mixed Graph

Greedy Mixed Graph (greedy_mixed_graph) is a Python implemention of greedy equivalence search (GES) algorithm for learning Gaussian graphical model with hidden variables.

## Installation



```bash
python setup.py install
```

## Usage

```python
import numpy as np
from greedy_mixed_graph.plot import plotGraphs
from greedy_mixed_graph.greedysearch import GreedySearch
from greedy_mixed_graph.generate import SimpleGraph, ParamedSimpleGraph, generate_params, generate_mixed_graph


# generate 10 random mixed graphs
graphs = generate_mixed_graph(p=5, n=10, max_in_degree=2)

# generate parameters and data for the 1st graph, graphs[0]
g = ParamedSimpleGraph(graphs[0])
params = generate_params(L=g.L, O=g.O)
g.assignParams(params[0], params[1])
g.generateData(1000)

# construct a GreedySearch object, run greedy search 10 times from the 10 mixed graphs as the start point
# the greedy search finishes in 5 min
GS = GreedySearch(mg_start=graphs[1:], n=1000, cov_mat=np.cov(g.data.T), edge_penalty=1, bic='ext_bic')
result = GS.greedy_search(n_restarts=10, mc_cores=2)

# plot the true graph and the estimated graph
plotGraphs([g.mg, result.final_graph.mg])
```


## References
The setup and algorithm refers to
```
@inproceedings{amendola2020structure,
  title={Structure learning for cyclic linear causal models},
  author={Am{\'e}ndola, Carlos and Dettling, Philipp and Drton, Mathias and Onori, Federica and Wu, Jun},
  booktitle={Conference on Uncertainty in Artificial Intelligence},
  pages={999--1008},
  year={2020},
  organization={PMLR}
}
```
also heavily based on
```
@article{nowzohour2017distributional,
  title={Distributional equivalence and structure learning for bow-free acyclic path diagrams},
  author={Nowzohour, Christopher and Maathuis, Marloes H and Evans, Robin J and B{\"u}hlmann, Peter},
  journal={Electronic Journal of Statistics},
  volume={11},
  number={2},
  pages={5342--5374},
  year={2017},
  publisher={Institute of Mathematical Statistics and Bernoulli Society}
}
```