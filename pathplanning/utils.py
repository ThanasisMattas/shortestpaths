# PathPlanning is free software; you may redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. You should have received a copy of the GNU
# General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.
# ======================================================================
"""
info:
    file        :  utils.py
    author      :  Thanasis Mattas
    license     :  GNU General Public License v3
    description :  helper functions
"""

from datetime import datetime
from itertools import combinations
import random

import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(G, path, path_cost):
  """plots the graph in a spring_layout"""
  shortest_path_edge_list = list(zip(path, path[1:]))

  pos = nx.spring_layout(G)
  nx.draw_networkx(G, pos, node_size=450, width=0.3)
  nx.draw_networkx_edges(G, pos=pos, edgelist=shortest_path_edge_list,
                         edge_color='r', width=3)
  plt.title(
    "nodes: {0}    edges: {1}    path cost: {2}\npath: ".format(
      G.number_of_nodes(), G.number_of_edges(), path_cost) + str(path)
  )
  plt.show()


def _edge_weight_bias(edge, num_nodes):
  """penalizes edges that connect distant nodes

  Args:
    edge (tuple)    : (tail, head)
    num_nodes (int) : used for normalization to 1

  Returns:
    bias (float)    : takes values in [0, 1]
  """
  # In case that num_nodes is big and the edge connects close nodes, the bias
  # can become too small. Thus, discretized bin values are used, so as to cap
  # a bias to the upper limit of the bin that it lies into.
  bias_bins = [0.1 * i for i in range(11)]
  bias = abs(edge[0] - edge[1]) / num_nodes
  for b in bias_bins:
    if bias < b:
      bias = b
      break
  return bias


def random_graph(num_nodes,
                 weighted=True,
                 weights_on="edges",
                 max_edge_weight=100,
                 max_node_weight=1000):
  """Generates a random graph of num_nodes, using the Erdős–Rényi model

  Args:
    num_nodes (int)       : number of nodes
    weighted (bool)       : defaults to True
    weights_on (string)   : 'edges', 'nodes' or 'edges-and-nodes'
    max_edge_weight (int) : each edge has a random weight from 0 to
                            max_edge_weight (defaults to 100)
    max_node_weight (int) : each node has a random weight from 0 to
                            max_node_weight (defaults to 100)

  Returns:
    adj_list (list)       : the adjacency list
                            each element is a list of tuples (neighbor, weight)
                            of the neighbors of each node
    G (Graph)             : used to plot the graph
  """
  random.seed(datetime.now)
  nodes = [node for node in range(1, num_nodes + 1)]
  edges = set()
  adj_list = [set() for _ in range(num_nodes + 1)]
  G = nx.Graph()
  G.add_nodes_from(nodes)
  node_weights = random.choices(range(max_node_weight), k=num_nodes + 1)

  # When nodes have weights, the weight that each neighbor holds at the
  # adjacency list (the edge weight) is increased by by its node-weight.
  # Whereas this holds true in the case of the adjacency list, it is not
  # correct when adding the weighted edges to the Graph object, but that's ok,
  # because the Graph object is used only for plotting.
  weight_mode = "not-weighted" if not weighted else weights_on

  weight = {
    "not-weighted": lambda _, __: 1,
    "edges": lambda _, bias: round(bias * random.randint(0, max_edge_weight)),
    "nodes": lambda node_id, _: node_weights[node_id],
    "edges-and-nodes": lambda node_id, bias: (
      node_weights[node_id] + round(bias * random.randint(0, max_edge_weight))
    )
  }

  for edge in combinations(nodes, 2):
    # The closer the nodes are, the more probable it is that they are connected
    # with an edge and the weight is lower. (This way, it is more realistic and
    # paths with too few nodes are avoided)
    bias = _edge_weight_bias(edge, num_nodes)
    edge_probability = max(
      0, (num_nodes - abs(edge[0] - edge[1])) / num_nodes - 0.4
    )
    random_probability = random.random()
    if random_probability < edge_probability:
      edges.add((*edge, weight[weight_mode](edge[1], bias)))
      adj_list[edge[0]].add((edge[1], weight[weight_mode](edge[1], bias)))
      adj_list[edge[1]].add((edge[0], weight[weight_mode](edge[0], bias)))

  G.add_weighted_edges_from(edges)
  return adj_list, G
