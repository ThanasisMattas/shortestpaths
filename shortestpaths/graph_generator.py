# graph_generator.py is part of ShortestPaths
#
# ShortestPaths is free software; you may redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version. You should have received a copy of the GNU General Pu-
# blic License along with this program. If not, see
# <https://www.gnu.org/licenses/>.
#
# (C) 2020 Athanasios Mattas
# ==========================================================================
"""Generates a pseudo-random graph, using a modified Gilbert version of the
Erdős-Rényi model.
"""

from itertools import combinations, permutations
import math
import random
from typing import Union

import networkx as nx

from shortestpaths.utils import time_this  # noqa: F401


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


def _edge_probability(edge,
                      gradient=0.5,
                      center=None,
                      p_0=0.8) -> float:
  """Evaluates the probability an edge exists, by the "proximity" of its nodes.

  Utilizing the numeric naming of the nodes, the distance between two nodes is
  taken as the absolute of the difference of their ids. For example, nodes 1
  and 5 will have distance 4.

  This probability will be used to build a modified Gilbert version of the
  Erdős-Rényi model, where instead of one coherent probability, each edge wiil
  have its own probability of existence.

  This function regulates the form and the density of the graph. By forcing a
  more sparse graph, we will get shortest-paths with more hops. thus aiding the
  visualization and the analysis. For that reason, distant edges are penalized,
  by getting low probability and big weights.

  The basis of the probability distribution is the sigmoid function, which out-
  puts values in the desired range 0-1, and provides simple parametric control.
  More specifically, working as a high-pass filter, cuts out the edge distances
  after its center.

  Control over the form and the density of the graph can be used in different
  kinds of analyses.

  The graph model parameters are:

    * center (c):
      controls the form and the density of the graph; values: (1, n-1)
        close to 0: filters out distant edges; graph becomes more sparse
        close to n: incorporates more distant edges; graph becomes more dense
    * gradient (λ):
      controls the form of the graph; values: (0, 1]
        close to 1: steep cut of probabilities after the center
        close to 0: smoother diffusion of the probability distribution near the
                    center
    * initial_probability (p_0):
      controls the density of the graph; values: (0, 1]
        p_0 (up) (up) density

  Note that the sigmoid distribution is inverted, by subtracting it from 1, to
  shift the lower probability to distant edges.

                             1
                      ----------------
                           - λ (x - c)
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

  Final probability equation:

              p(x) = p_0 (1 - sigmoid(x; λ, c))

  """
  exponent = -gradient * (abs(edge[0] - edge[1]) - center)
  if exponent >= 5:
    return p_0
  elif exponent <= -5:
    return 0
  else:
    sigmoid = 1 / (1 + math.exp(exponent))
    return p_0 * (1 - sigmoid)


def _edge_weight(edge,
                 n,
                 weight_mode,
                 edge_initial_weight):
  """Calculates a relative to the proximity of the nodes random edge weight."""
  if weight_mode in ["edges", "edges-and-nodes"]:
    bias = _edge_weight_bias(edge, n)
    return round(bias * edge_initial_weight)
  elif weight_mode == "nodes":
    return 0
  elif weight_mode == "unweighted":
    return 1
  else:
    raise Exception(f"Unknown weight-mode: {weight_mode}")


def graph_density(n, m, directed):
  """Evaluates the graph density as m / m_max."""
  if directed:
    m_max = n * (n - 1)
  else:
    m_max = n * (n - 1) / 2
  return m / m_max


# @time_this
def random_graph(n,
                 weighted=True,
                 directed=True,
                 weights_on="edges-and-nodes",
                 max_edge_weight=1000,
                 max_node_weight=50,
                 random_seed=None,
                 center_portion=0.3,
                 gradient=0.5,
                 p_0=0.6,
                 get_probability_distribution=False):
  """Generates a n-nodes random graph, using a modified Erdős-Rényi model.
  (see _edge_probability() docstring)

  The graph is represented by its adjacency list. NetworkX is used only for
  plotting.

  NOTE:
    When nodes are weighted:
    1. The weight that each neighbor holds at the adjacency list (the edge we-
       ight) is increased by by its node-weight. Whereas this holds true for
       the adjacency list, it is not correct when adding the weighted edges to
       the nx.Graph object, but that's ok, because the nx.Graph object is used
       only for plotting.
    2. The undirected graph is converted to a digraph, since each undirected
       edge breaks into two edges with opposite directions and different
       weights. For each hop, we pay the weight of the edge plus the weight of
       the head. Namely,

                 {a, b}, weight_a, weight_b, weight_edge
                                becomes
                     (a, b), weight_edge + weight_b
                                   &
                     (b, a), weight_edge + weight_a

       This way each edge is related to one value, instead of three, therefore
       Dijkstra's algorithm can operate without modifications.
    3. In case of a digraph, the edge_weight is increased by the head weight.

  Args:
    n (int)                : number of nodes
    weighted (bool)        : defaults to True
    weights_on (string)    : 'edges', 'nodes' or 'edges-and-nodes'
    max_edge_weight (int)  : each edge has a random weight from 0 to
                             max_edge_weight (defaults to 1000)
    max_node_weight (int)  : each node has a random weight from 0 to
                             max_node_weight (defaults to 1000)
    random_seed (int)      : in case of fixed random graph (defaults to None)
    center_portion (float) : defines the sigmoid center
    gradient (float)       : sigmoid exponent amplitude
    p_0 (float)            : initial probability
    get_probability_distribution (bool)
                           : flag to return graph data for meta-analysis

  Returns:
    adj_list (list)        : the adjacency list: [{(neighbor, weight),},]
    G (Graph)              : used to plot the graph
  """
  weight_mode = "unweighted" if not weighted else weights_on
  random.seed(random_seed)

  nodes = list(range(1, n + 1))
  # edges is used only for the graph visualization via NetworkX.
  edges = set()
  adj_list = [set() for _ in range(n + 1)]

  # Generate random weights for all nodes and edges.
  if weight_mode in ["nodes", "edges-and-nodes"]:
    node_weights = random.choices(range(max_node_weight + 1), k=n + 1)
  center = center_portion * n

  if get_probability_distribution:
    probs = []
    edge_lengths = []
    edge_lengths_true = []

  # Iterate through all possible edges, randomly weight them and randomly desi-
  # de which to keep.
  if directed:  # permutations: n (n - 1)
    edge_weights = random.choices(range(max_edge_weight + 1),
                                  k=n * (n - 1))
    for i, edge in enumerate(permutations(nodes, 2)):
      # The closer the nodes are, the more probable it is that they are connec-
      # ted with an edge and the edge-weight is lower. This way, it is more
      # realistic - edges of nearby nodes cost less - and paths with too few
      # nodes, that go straight to the sink, are avoided.
      # Namely, distance (up) (down) edge_probability.
      edge_probability = _edge_probability(edge,
                                           gradient=gradient,
                                           center=center,
                                           p_0=p_0)
      if get_probability_distribution:
        probs.append(edge_probability)
        edge_lengths.append(abs(edge[0] - edge[1]))

      edge_initial_weight = edge_weights[i]

      random_probability = random.random()
      if edge_probability > random_probability:

        if get_probability_distribution:
          edge_lengths_true.append(abs(edge[0] - edge[1]))

        edge_weight = _edge_weight(edge,
                                   n,
                                   weight_mode,
                                   edge_initial_weight)
        if weight_mode in ["nodes", "edges-and-nodes"]:
          head_weight = node_weights[edge[1]]
          adj_list[edge[0]].add((edge[1], edge_weight + head_weight))
        else:
          adj_list[edge[0]].add((edge[1], edge_weight))
        edges.add((*edge, edge_weight))
  else:  # combinations: n (n - 1) / 2
    edge_weights = random.choices(range(max_edge_weight + 1),
                                  k=n * (n - 1) // 2)
    for i, edge in enumerate(combinations(nodes, 2)):
      # The closer the nodes are, the more probable it is that they are connec-
      # ted with an edge and the edge-weight is lower. This way, it is more
      # realistic - edges of nearby nodes cost less - and paths with too few
      # nodes, that go straight to the sink, are avoided.
      # Namely, distance (up) (down) edge_probability.
      edge_probability = _edge_probability(edge,
                                           gradient=gradient,
                                           center=center,
                                           p_0=p_0)
      if get_probability_distribution:
        probs.append(edge_probability)
        edge_lengths.append(abs(edge[0] - edge[1]))

      edge_initial_weight = edge_weights[i]

      random_probability = random.random()
      if edge_probability > random_probability:

        if get_probability_distribution:
          edge_lengths_true.append(abs(edge[0] - edge[1]))

        edge_weight = _edge_weight(edge,
                                   n,
                                   weight_mode,
                                   edge_initial_weight)
        if weight_mode in ["nodes", "edges-and-nodes"]:
          tail_weight = node_weights[edge[0]]
          head_weight = node_weights[edge[1]]
          adj_list[edge[0]].add((edge[1], edge_weight + head_weight))
          adj_list[edge[1]].add((edge[0], edge_weight + tail_weight))
        else:
          adj_list[edge[0]].add((edge[1], edge_weight))
          adj_list[edge[1]].add((edge[0], edge_weight))
        edges.add((*edge, edge_weight))

  if directed:
    G = nx.DiGraph()
  else:
    G = nx.Graph()
  G.add_nodes_from(nodes)
  G.add_weighted_edges_from(edges)

  if get_probability_distribution:
    return G, probs, edge_lengths, edge_lengths_true
  return adj_list, G


def adj_list_reversed(adj_list):
  """Creates the adj_list of the inverted graph in O(n^2).

  The inverted graph is the same with the direct, but with inverted edges
  direction.
  """
  i_adj_list = [set() for _ in range(len(adj_list))]
  for node, neighbors in enumerate(adj_list):
    for neighbor in neighbors:
      i_adj_list[neighbor[0]].add((node, neighbor[1]))
  return i_adj_list


def nx_to_adj_list(G: Union[nx.Graph, nx.DiGraph]) -> tuple:
  """Converts NetworkX graph to an adj_list.

  - adj_list format:
      [
        {
          (neighbor, weight),
        },
      ]
  - G.nodes() are encoded to a numerical representation [1, n], in order to be
    used as indexes of adj_list.

  Args:
    G (Graph or Digraph)

  Returns:
    adj_list (list)
    decoder (dict) : maps the numerical representation [1, n] back to G.nodes()

  Raises:
    Exception       : if G is negatively weighted
  """

  if nx.is_negatively_weighted(G):
    raise Exception("Only non-negative weighted graphs are currently"
                    " supported.")

  n = G.number_of_nodes()
  encoder = dict(zip(G.nodes, range(1, n + 1)))
  decoder = dict(zip(range(1, n + 1), G.nodes))
  adj_list = [set() for _ in range(n + 1)]

  if nx.is_directed(G):
    if nx.is_weighted(G):
      for u, v, w in G.edges.data("weight"):
        adj_list[encoder[u]].add((encoder[v], w))
    else:
      for u, v in G.edges:
        adj_list[encoder[u]].add((encoder[v], 1))
  else:
    if nx.is_weighted(G):
      for u, v, w in G.edges.data("weight"):
        adj_list[encoder[u]].add((encoder[v], w))
        adj_list[encoder[v]].add((encoder[u], w))
    else:
      for u, v in G.edges:
        adj_list[encoder[u]].add((encoder[v], 1))
        adj_list[encoder[v]].add((encoder[u], 1))

  return adj_list, encoder, decoder


def decode_path_nodes(paths: list, decoder: dict) -> list:
  decoded_paths = []
  for p in paths:
    decoded_paths.append(
      (list(map(decoder.get, p[0])),) + p[1:]
    )
  return decoded_paths
