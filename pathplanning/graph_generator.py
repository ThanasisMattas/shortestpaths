# graph_geenrator.py is part of PathPlanning
#
# PathPlanning is free software; you may redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. You should have received a copy of the GNU
# General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.
#
# (C) 2020 Athanasios Mattas
# =======================================================================
"""Generates a pseudo-random graph, using a modified Gilbert version of the
Erdős-Rényi model.
"""

from itertools import combinations
import math
import random

import networkx as nx


def _edge_weight_bias(edge, n) -> float:
  """Penalizes edges that connect distant nodes.

  Args:
    edge (tuple)    : (tail, head)
    n (int) : used for normalization to 1

  Returns:
    bias (float)    : takes values in [0, 1]
  """
  # Bias will be capped with one of [0.1, 0.2, ..., 1.0], depending on with bin
  # it falls into, and then doubled.
  bias = abs(edge[0] - edge[1]) / n
  bias = (int(bias * 10) + 1) / 10 * 2
  return bias


def _edge_probability(edge, n, **kwargs) -> float:
  """Evaluates the probability an edge exists, by the "proximity" of its nodes.

  This probability will be used to build the Gilbert version of the Erdős-Rényi
  model.

  This function, working as a low-pass filter, regulates how dense the graph
  will be, to be used in different kinds of analysis. For visualization purpo-
  ses, by forcing a more sparse graph, we will get shortest-paths with more
  hops. For that reason, distant edges are penalized, by quickly driving the
  sigmoid curve to 1. Hence, the attributes gradient and center regulate how
  dense the random graph will be:

    center  zero       n/2         n
            ------------------------
    graph   sparse   neutral   dense

    gradient of the slope:
      > 1
        dense  -> slightly denser, adding the most distant edges
        sparse -> almost same density, but the balance will be swifted towards
                  the close edges
      < 1
        dense  -> slightly less dense, removing the most distant edges
        sparse -> almost same density, but the balance will be swifted towards
                  the distant edges

  Finally, the sigmoid distribution is inverted, by subtracting it from 1.


                                 1
                      -----------------------
                           - grand (x - cent)
                      1 + e

  1.0 |                               _ _ _ _ _
      |                           .
      |                        .
      |                      .
      |                     .
      |                    .
  0.5 |- - - - - - - - - -.
      |                  .:
      |                 . :
      |                .  :
      |               .   :
      |             .     :
  0.0 | _ _ _ _  .        :
      |___________________:_______________________
                        center   abs(head-tail) ->
  """
  gradient = kwargs.pop("gradient", 2),
  center_factor = kwargs.pop("center_factor", 0.3)

  center = center_factor * n
  sigmoid = 1 / (1 + math.exp(-gradient * (abs(edge[0] - edge[1]) - center)))
  return 1 - sigmoid


def _edge_weight(edge,
                 n,
                 weight_mode,
                 max_edge_weight,
                 edgewise_consistent_seed=True):
  """Calculates a relative to the proximity of the nodes random edge weight."""
  if weight_mode in ["edges", "edges-and-nodes"]:
    if edgewise_consistent_seed:
      # All edges will have consistent yet different seed.
      random.seed(edge[0] ** 2 + edge[1] ** 2 + edge[1])
    bias = _edge_weight_bias(edge, n)
    return round(bias * random.randint(0, max_edge_weight))
  elif weight_mode == "nodes":
    return 0
  elif weight_mode == "unweighted":
    return 1
  else:
    raise Exception(f"Unknown weight-mode: {weight_mode}")


def random_graph(n,
                 weighted=True,
                 weights_on="edges",
                 max_edge_weight=1000,
                 max_node_weight=1000,
                 random_seed=None,
                 **kwargs):
  """Generates a n-nodes random graph, using the Erdős-Rényi model.

  The graph is represented by its adjacency list. NetworkX is used only for
  plotting.

  NOTE:
    When nodes are weighted:
    1. The weight that each neighbor holds at the adjacency list (the edge we-
       ight) is increased by by its node-weight. Whereas this holds true for
       the adjacency list, it is not correct when adding the weighted edges to
       the nx.Graph object, but that's ok, because the nx.Graph object is used
       only for plotting.
    2. The undirected, simple graph is converted to a directed multigraph, sin-
       ce each undirected edge breaks into two edges with opposite directions
       and different weights. Namely,

                   {a, b}, weight_a, weight_b, weight_edge
                                becomes
                     (a, b), weight_edge + weight_b
                                   &
                     (b, a), weight_edge + weight_a

       This way each edge is related to one value, instead of three, therefore
       Dijkstra's algorithm can operate without modifications.

  Args:
    n (int)               : number of nodes
    weighted (bool)       : defaults to True
    weights_on (string)   : 'edges', 'nodes' or 'edges-and-nodes'
    max_edge_weight (int) : each edge has a random weight from 0 to
                            max_edge_weight (defaults to 100)
    max_node_weight (int) : each node has a random weight from 0 to
                            max_node_weight (defaults to 100)
    random_seed (int)     : in case of fixed random graph (defaults to None)

  Returns:
    adj_list (list)       : the adjacency list
                            each element is a list of tuples (neighbor, weight)
                            of the neighbors of each node
    G (Graph)             : used to plot the graph
  """
  weight_mode = "unweighted" if not weighted else weights_on
  random.seed(random_seed)

  nodes = list(range(1, n + 1))
  # edges is used only for the graph visualization via NetworkX.
  edges = set()
  adj_list = [set() for _ in range(n + 1)]

  if weight_mode in ["nodes", "edges-and-nodes"]:
    node_weights = random.choices(range(max_node_weight + 1), k=n + 1)

  # Iterate through all possible edges, randomly weight them and randomly desi-
  # de which to keep.
  for edge in combinations(nodes, 2):
    # The closer the nodes are, the more probable it is that they are connected
    # with an edge and the edge-weight is lower.
    # (This way, it is more realistic - edges of nearby nodes cost less - and
    # paths with too few nodes, that go straight to the sink, are avoided.)
    # Namely, distance (up) (down) edge_probability.
    edge_probability = _edge_probability(edge, n, *kwargs)

    random_probability = random.random()
    if edge_probability > random_probability:
      edge_weight = _edge_weight(edge,
                                 n,
                                 weight_mode,
                                 max_edge_weight,
                                 bool(random_seed))
      if weight_mode in ["nodes", "edges-and-nodes"]:
        tail_weight = node_weights[edge[0]]
        head_weight = node_weights[edge[1]]
        adj_list[edge[0]].add((edge[1], edge_weight + head_weight))
        adj_list[edge[1]].add((edge[0], edge_weight + tail_weight))
      else:
        adj_list[edge[0]].add((edge[1], edge_weight))
        adj_list[edge[1]].add((edge[0], edge_weight))
      edges.add((*edge, edge_weight))

  G = nx.Graph()
  G.add_nodes_from(nodes)
  G.add_weighted_edges_from(edges)

  return adj_list, G
