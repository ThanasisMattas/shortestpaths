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
Erdős–Rényi model.
"""

import random

import networkx as nx


def _edge_weight_bias(edge, num_nodes) -> float:
  """Penalizes edges that connect distant nodes.

  Args:
    edge (tuple)    : (tail, head)
    num_nodes (int) : used for normalization to 1

  Returns:
    bias (float)    : takes values in [0, 1]
  """
  # Bias will be capped with one of [0.1, 0.2, ..., 1.0], depending on with bin
  # it falls into, and then doubled.
  bias = abs(edge[0] - edge[1]) / num_nodes
  bias = (int(bias * 10) + 1) / 10 * 2
  return bias


def _edge_weight(edge,
                 num_nodes,
                 weight_mode,
                 max_edge_weight,
                 random_seed=None):
  """Calculates a relative to the proximity of the nodes ranodm edge weight."""
  bias = _edge_weight_bias(edge, num_nodes)

  if random_seed is not None:
    # Almost all edges will have consistent yet different form each other seed.
    random.seed(sum(edge))

  if weight_mode == "not-weighted":
    return 1
  elif weight_mode in ["edges", "edges-and-nodes"]:
    return round(bias * random.randint(0, max_edge_weight))
  elif weight_mode in ["nodes", "edges-and-nodes"]:
    return 0
  else:
    raise Exception(f"Unknown weight-mode: {weight_mode}")


def random_graph(num_nodes,
                 weighted=True,
                 weights_on="edges",
                 max_edge_weight=1000,
                 max_node_weight=1000,
                 random_seed=None):
  """Generates a random graph of <num_nodes>, using the Erdős–Rényi model.

  The graph is represented by its adjacency list. NetworkX is used only for
  plotting.

  NOTE:
    When nodes are weighted:
    1. The weight that each neighbor holds at the
       adjacency list (the edge weight) is increased by by its node-weight.
       Whereas this holds true in the case of the adjacency list, it is not
       correct when adding the weighted edges to the nx.Graph object, but
       that's ok, because the nx.Graph object is used only for plotting.
    2. The undirected, simple graph is converted to a directed multigraph,
       since each undirected edge breaks into two with opposite directions
       and different weights. Namely,

                   {a, b}, weight_a, weight_b, weight_edge
                                 becomes
                     (a, b), weight_edge + weight_b &
                     (b, a), weight_edge + weight_a

       This way each edge is related to one value, instead of three, hence
       the Dijkstra's algorithm can operate without modifications.

  Args:
    num_nodes (int)       : number of nodes
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
  weight_mode = "not-weighted" if not weighted else weights_on
  random.seed(random_seed)

  nodes = list(range(1, num_nodes + 1))
  # edges is used only for the graph visualization via networkx
  edges = set()
  adj_list = [set() for _ in range(num_nodes + 1)]

  if weight_mode in ["nodes", "edges-and-nodes"]:
    node_weights = random.choices(range(max_node_weight + 1), k=num_nodes + 1)

  # Iterate through all possible edges and randomly deside which to keep.
  for edge in combinations(nodes, 2):
    # The closer the nodes are, the more probable it is that they are connected
    # with an edge and the edge-weight is lower. (This way, it is more
    # realistic - edges of nearby nodes cost less - and paths with too few
    # nodes, that go straight to the end, are avoided.)
    # Namely, distance (up) (down) edge_probability.
    edge_probability = max(0, 1 - abs(edge[0] - edge[1]) / num_nodes - 0.6)
    random_probability = random.random()
    if edge_probability > random_probability:
      edge_weight = _edge_weight(edge,
                                 num_nodes,
                                 weight_mode,
                                 max_edge_weight,
                                 random_seed)
      if weight_mode in ["nodes", "edges-and-nodes"]:
        tail_node_weight = node_weights[edge[0]]
        head_node_weight = node_weights[edge[1]]
        adj_list[edge[0]].add((edge[1], edge_weight + head_node_weight))
        adj_list[edge[1]].add((edge[0], edge_weight + tail_node_weight))
      else:
        adj_list[edge[0]].add((edge[1], edge_weight))
        adj_list[edge[1]].add((edge[0], edge_weight))
      edges.add((*edge, edge_weight))

  G = nx.Graph()
  G.add_nodes_from(nodes)
  G.add_weighted_edges_from(edges)

  return adj_list, G