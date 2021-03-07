# utils.py is part of PathPlanning
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
"""Houses some utility functions."""

import ast
from datetime import datetime, timedelta
from itertools import combinations
from functools import wraps
from math import sqrt
from operator import itemgetter
import os
import random
import time
from timeit import default_timer as timer
from typing import Iterable
import warnings

import click
import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(G,
               paths_data,
               disconnected_nodes,
               save_graph,
               show_graph,
               layout_seed=None):
  """Plots the graph and all the generated paths in spring_layout."""
  pos = nx.spring_layout(G, k=10 / sqrt(G.number_of_nodes()), seed=layout_seed)

  # Layouts
  # -------
  # circular_layout
  # spring_layout                <--
  # fruchterman_reingold_layout  <--
  # spiral_layout                <--

  # 1. Draw the graph
  nx.draw_networkx(G, pos, node_size=450, width=0.25, alpha=0.3,
                   with_labels=False)

  # 2. Draw the disconnected nodes
  nx.draw_networkx_nodes(G, pos=pos, nodelist=disconnected_nodes, node_color='r',
                         node_shape='x', node_size=800, linewidths=3)

  # colors = iter(['b', 'm', 'g', 'k', 'r', 'c', 'y', 'w'])
  colors = iter(['mediumblue', 'm', 'g', 'k', 'r', 'c', 'y', 'w'])
  # Accumulates all the nodes of all the paths for label drawing.
  paths_nodes = set()

  for path in paths_data:
    paths_nodes.update(path[0])

  # Change the font of the labels of the path nodes and restore alpha=None.
  for node, (x, y) in pos.items():
    if node in paths_nodes:
      plt.text(x, y, node, fontsize=14, ha='center', va='center')

  # 3. Draw the nodes of all the paths
  nx.draw_networkx_nodes(G, pos=pos, nodelist=paths_nodes, node_size=600,
                         edgecolors='k', node_color="deepskyblue")

  # 4. Draw the paths
  for i, path in enumerate(paths_data):
    color = next(colors)
    path_edges_sequence = list(zip(path[0], path[0][1:]))

    label = (f"path_{i + 1}: {str(path[0])}\ncost: {path[1]}    "
             f"disconnected nodes: {str(list(path[2]))}")

    # Draw the path
    nx.draw_networkx_edges(G, pos=pos, edgelist=path_edges_sequence,
                           edge_color=color,
                           width=len(paths_data) + 15 - 4.5 * i, label=label)

    # Mark the disconnceted node with an X.
    nx.draw_networkx_nodes(G, pos=pos, nodelist=path[2], node_color=color,
                           node_shape='x', node_size=800, linewidths=3)

  # Draw 'Start' & 'End' labels.
  # labels = {paths_data[0][0][0]: "Start", paths_data[0][0][-1]: "End"}
  # for node, (x, y) in pos.items():
  #   if node in labels.keys():
  #     plt.text(x + 50, y + 50, node, fontsize=14, ha='center', va='center')
  # nx.draw_networkx_labels(G, pos=pos, labels=labels,
  #                         font_color='k', font_size=20,
  #                         bbox=dict(boxstyle="square", fc='w', ec='k'))

  frame_title = (f"#nodes: {G.number_of_nodes()}    "
                 f"#edges: {G.number_of_edges()}")
  if disconnected_nodes:
    frame_title += f"\ndisconnected nodes: {list(disconnected_nodes)}"
  plt.title(frame_title)
  plt.legend()

  if save_graph:
    date_n_time = str(datetime.now())[:19]
    date_n_time = date_n_time.replace(':', '-').replace(' ', '_')
    file_name = f"graph_vis_{date_n_time}.png"
    plt.savefig(os.path.join(os.getcwd(), file_name))
  if show_graph:
    plt.show()


def plot_adaptive_dijkstra(G,
                           paths_data,
                           disconnected_nodes,
                           nodes_visited_sequence,
                           checkpoint_node):
  """Plots the adaptive Dijkstra's algorithms."""
  pos = nx.spring_layout(G, seed=1)

  # 1. Draw the graph
  nx.draw_networkx(G, pos, node_size=450, width=0.25, alpha=0.4,
                   with_labels=False)

  nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes_visited_sequence,
                         node_color='goldenrod', node_size=1250,
                         linewidths=3, label="nodes visited by the algorithm")
  retrieved_state = nodes_visited_sequence[
    :nodes_visited_sequence.index(checkpoint_node)
  ]
  nx.draw_networkx_nodes(G, pos=pos, nodelist=retrieved_state,
                         node_color='g', node_size=800,
                         linewidths=3, label="retrieved state")

  # 2. Draw the disconnected nodes
  nx.draw_networkx_nodes(G, pos=pos, nodelist=disconnected_nodes, node_color='r',
                         node_shape='x', node_size=800, linewidths=3)

  # colors = iter(['b', 'm', 'g', 'k', 'r', 'c', 'y', 'w'])
  colors = iter(['mediumblue', 'r', 'g', 'k', 'r', 'c', 'y', 'w'])
  # Accumulates all the nodes of all the paths for label drawing.
  paths_nodes = set()

  for path in paths_data:
    paths_nodes.update(path[0])

  # Change the font of the labels of the path nodes and restore alpha=None.
  for node, (x, y) in pos.items():
    if node in paths_nodes:
      plt.text(x, y, node, fontsize=14, ha='center', va='center')

  # 3. Draw the nodes of all the paths
  nx.draw_networkx_nodes(G, pos=pos, nodelist=paths_nodes, node_size=550,
                         edgecolors='k', node_color="deepskyblue")

  # 4. Draw the paths
  for i, path in enumerate(paths_data):
    color = next(colors)
    path_edges_sequence = list(zip(path[0], path[0][1:]))

    label = (f"path_{i + 1}: {str(path[0])}\ncost: {path[1]}    "
             f"disconnected nodes: {str(list(path[2]))}")

    # Draw the path
    nx.draw_networkx_edges(G, pos=pos, edgelist=path_edges_sequence,
                           edge_color=color,
                           width=len(paths_data) + 9 - 6 * i, label=label)

    # Mark the disconnceted node with an X.
    nx.draw_networkx_nodes(G, pos=pos, nodelist=path[2], node_color=color,
                           node_shape='x', node_size=800, linewidths=3)

  # Draw 'Start' & 'End' labels.
  # labels = {paths_data[0][0][0]: "Start", paths_data[0][0][-1]: "End"}
  # for node, (x, y) in pos.items():
  #   if node in labels.keys():
  #     plt.text(x + 50, y + 50, node, fontsize=14, ha='center', va='center')
  # nx.draw_networkx_labels(G, pos=pos, labels=labels,
  #                         font_color='k', font_size=20,
  #                         bbox=dict(boxstyle="square", fc='w', ec='k'))

  frame_title = (f"#nodes: {G.number_of_nodes()}    "
                 f"#edges: {G.number_of_edges()}")
  if disconnected_nodes:
    frame_title += f"\ndisconnected nodes: {list(disconnected_nodes)}"
  plt.title(frame_title)
  plt.legend()
  plt.show()


def _edge_weight_bias(edge, num_nodes) -> float:
  """Penalizes edges that connect distant nodes.

  Args:
    edge (tuple)    : (tail, head)
    num_nodes (int) : used for normalization to 1

  Returns:
    bias (float)    : takes values in [0, 1]
  """
  # Bias will be capped with one of [0.1, 0.2, ..., 1.0], depending on with bin
  # it falls into.
  bias_bins = [0.1 * i for i in range(1, 11)]
  bias = abs(edge[0] - edge[1]) / num_nodes
  for b in bias_bins:
    if bias < b:
      bias = b * 2
      break
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


def print_duration(start, end, process, time_type=None):
    """Prints the duration of a process.

    Args:
      start (float)    : the starting timestamp in seconds
      end (float)      : the ending timestamp in seconds
      process (string) : the process name
      time_type (string) : i.e. wall-clock, CPU or sys time (defaults to None)
    """
    if time_type is None:
      prefix = f"{process} time"
    else:
      prefix = f"{process:-<24}{time_type} time"
    duration = timedelta(seconds=end - start)
    print(f"{prefix:-<40}{duration}"[:51])


def time_this(wall_clock=None):
    """function timer decorator

    - Uses wraps to preserve the metadata of the decorated function
      (__name__ and __doc__)
    - prints wall and CPU (user+sys) time

    usage:
      @time_this
      def a_func(): pass

      @time_this()  # wall_clock=False
      def a_func(): pass

      @time_this(wall_clock=True)
      def a_func(): pass

    Args:
        f(funtion)      : the function to be decorated

    Returns:
        wrap (callable) : returns the result of the decorated function
    """
    def inner_decorator(f):
      if not callable(f):
        raise Exception(f"{f} is not a callable and, thus, it cannot be "
                        f"decorated with @time_this.")

      @wraps(f)
      def wrapper(*args, **kwargs):
        using_wall_clock = False
        if callable(wall_clock):
          # Meaning that @time_this is used without ()
          pass
        else:
          if wall_clock:
            using_wall_clock = True
            start_wall = timer()
        start_user_plus_sys = time.process_time()
        result = f(*args, **kwargs)
        end_user_plus_sys = time.process_time()
        if using_wall_clock:
          end_wall = timer()
          print_duration(start_wall,
                         end_wall,
                         f.__name__,
                         "Wall")
        print_duration(start_user_plus_sys,
                       end_user_plus_sys,
                       f.__name__,
                       "CPU")
        return result
      return wrapper

    if callable(wall_clock):
      return inner_decorator(wall_clock)
    return inner_decorator


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


def check_nodal_connection(nodes: Iterable,
                           adj_list: list,
                           disconnected_nodes: Iterable) -> Iterable:
  """Checks for connection status of important nodes, i.e. start node and,
  in case of a disconnected important node, it replaces it with the closest
  neighbor.
  """
  for i, node in enumerate(nodes):
    if node in disconnected_nodes:
      input(f"Node <{node}> is disconnected. Setting it to its closest "
            "first neighbor. Press ENTER to continue...")
      node_neighbors = sorted(adj_list[node], key=itemgetter(1))
      for neighbor in node_neighbors:
        if neighbor[1] not in disconnected_nodes:
          nodes[i] = neighbor[1]
          break
  return nodes


def extract_path(visited, start, goal, with_step_weights=False):
  """Dijkstra's method saves the shortest path cost for each node of the graph,
  as well as its previous node on the path, so as to retrieve the path by
  jumping through previous nodes, until the start node.

  Args:
    visited (2D list)               : each entry is a 2-list,
                                      [path_cost, prev_node_id]
    start, goal (any hashable type) : the ids of start and goal nodes
    with_step_weights (bool)        : - True : returns just the node_id's
                                      - False: returns 2-lists:
                                               (node_id, step_to_node_cost)

  Returns:
    path (list)                     : each entry is a node_id or a 2-list,
                                      [node_id, edge-cost] (depending on the
                                      with_step_weights argument), constituting
                                      the consecutive nodes of the path.
  """
  # At this stage, edge costs are comulative.
  path = [[goal, visited[goal][0]]]
  node = goal
  while node != start:
    prev_node = visited[node][1]
    prev_node_cost = visited[prev_node][0]
    # Offset the cost of node with prev_node_cost, because the corresponding
    # costs are comulative and not edge-costs.
    path[-1][1] -= prev_node_cost
    path.append([prev_node, prev_node_cost])
    if node == prev_node:
      # raise Exception(f"Start node ({start}) is not connected to the"
      #                 f" destination node ({goal}).")
      warnings.warn(f"Start node ({start}) is not connected to the"
                    f" destination node ({goal}).")
      return []
    node = prev_node
  path.reverse()

  if not with_step_weights:
    path = list(list(zip(*path))[0])
  return path
