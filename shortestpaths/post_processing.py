# post_processing.py is part of ShortestPaths
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
"""Handles the post processing."""

from datetime import datetime
import os
from typing import Literal
import warnings

import click
import matplotlib.pyplot as plt
import networkx as nx


COLORS = ['mediumblue', 'm', 'g', 'k', 'r', 'c', 'y', 'w']


def path_label(path, path_number, failing):
  if failing is None:
      failed_msg = ''
  elif path[2] is None:
    # then nothing failed, because this is the absolute shortest path
    failed_msg = f"failed {failing[:-1]}: -"
  elif (isinstance(path[2], tuple)) and (len(path[2]) == 2):
    # then it is a single edge
    failed_msg = f"failed {failing[:-1]}: {str(path[2])}"
  elif not hasattr(path[2], "__len__"):
    failed_msg = f"failed {failing[:-1]}: {str(path[2])}"
  else:
    # then it is a list of failed nodes or edges
    failed_msg = f"failed {failing}: {str(list(path[2]))}"
  label = (f"path_{path_number}: {str(path[0])}\ncost: {path[1]}"
           f"    {failed_msg}")
  return label


def plot_graph(G,
               paths_data,
               failing: Literal["edges", "nodes", None]="edges",  # noqa: E252
               save_graph=False,
               show_graph=True,
               layout_seed=None,
               draw_edge_weights=False):
  """Plots the graph and all the generated paths in spring_layout."""
  for i in range(len(paths_data)):
    if (not hasattr(paths_data[i], "__len__")) or (len(paths_data[i]) != 3):
      warnings.warn(f"An invalid path detected. Path_data: {paths_data[i]}")
      del paths_data[i]
  # , k=10 / sqrt(G.number_of_nodes())
  pos = nx.spring_layout(G, seed=layout_seed)

  # Layouts
  # -------
  # circular_layout
  # spring_layout                <--
  # fruchterman_reingold_layout  <--
  # spiral_layout                <--

  # 1. Draw the graph
  if G.number_of_nodes() < 400:
    node_size = 450
    path_node_size = 600
    failed_node_size = 800
  elif G.number_of_nodes() < 2200:
    node_size = 450 - G.number_of_nodes() // 5
    path_node_size = 600 - G.number_of_nodes() // 10
    failed_node_size = 800 - G.number_of_nodes() // 10
  else:
    node_size = 10
    path_node_size = 350
    failed_node_size = 550
  nx.draw_networkx(G, pos, node_size=node_size, width=0.25, alpha=0.3,
                   with_labels=False, arrows=False)
  if draw_edge_weights:
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

  # 2. Draw the nodes of all the paths
  all_paths_nodes = set()
  for path in paths_data:
    all_paths_nodes.update(path[0])
  # Change the font of the labels of the path nodes and restore alpha=None.
  for node, (x, y) in pos.items():
    if node in all_paths_nodes:
      plt.text(x, y, node, fontsize=14, ha='center', va='center')
  nx.draw_networkx_nodes(G, pos=pos, nodelist=all_paths_nodes, edgecolors='k',
                         node_size=path_node_size, node_color="deepskyblue")

  # 3. Draw the paths
  colors = iter(COLORS)
  for i, path in enumerate(paths_data):
    # Generate the legend label

    color = next(colors)
    label = path_label(path, i + 1, failing)
    path_edges_sequence = list(zip(path[0], path[0][1:]))
    # arrows=False, arrowsize=20, arrowstyle='fancy',
    # min_source_margin=1, min_target_margin=1,
    # from matplotlib.patches import ConnectionStyle
    # connectionstyle=ConnectionStyle("Arc3", rad=0.2),
    nx.draw_networkx_edges(G, pos=pos, edgelist=path_edges_sequence,
                           edge_color=color, arrows=False,
                           width=len(paths_data) + 15 - 4.5 * i, label=label)

    # Mark the disconnceted edge or node with an ×.
    if failing == "nodes":
      if path[2] not in [None, [None]]:
        if hasattr(path[2], "__len__"):
          nodelist = path[2]
        else:
          nodelist = [path[2]]
        nx.draw_networkx_nodes(G, pos=pos, nodelist=nodelist, node_color=color,
                               node_shape='x', node_size=failed_node_size,
                               linewidths=3)
    elif failing == "edges":
      # Check for the case of the absolute shortest path, where there is no
      # disconnected edge.
      if path[2] is not None:
        nx.draw_networkx_edge_labels(G, pos, edge_labels={path[2]: '×'},  # ✕×✗
                                     font_size=50, font_color=color,
                                     bbox=dict(alpha=0), rotate=False)
    elif failing is None:
      pass
    else:
      raise ValueError("failing should be 'edges', 'nodes' or None")

  # 4. Draw 'source' & 'Sink' labels.
  # labels = {paths_data[0][0][0]: "source", paths_data[0][0][-1]: "Sink"}
  # for node, (x, y) in pos.items():
  #   if node in labels.keys():
  #     plt.text(x + 50, y + 50, node, fontsize=14, ha='center', va='center')
  # nx.draw_networkx_labels(G, pos=pos, labels=labels,
  #                         font_color='k', font_size=20,
  #                         bbox=dict(boxstyle="square", fc='w', ec='k'))

  frame_title = (f"#nodes: {G.number_of_nodes()}   "
                 f"#edges: {G.number_of_edges()}   "
                 f"#paths: {len(paths_data)}")
  plt.title(frame_title)
  plt.legend()

  if save_graph:
    date_n_time = str(datetime.now())[:19]
    date_n_time = date_n_time.replace(':', '-').replace(' ', '_')
    file_name = f"graph_vis_{date_n_time}.png"
    plt.savefig(os.path.join(os.getcwd(), file_name))
  if show_graph:
    plt.show()


def print_paths(k_paths, failing=None):
  for k, path in enumerate(k_paths):
    msg = f"path {k + 1}: {path[0]}   cost: {path[1]}"
    if failing:
      msg += f"   failed {failing[-1]}: {path[2]}"
    click.echo(msg)


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
  all_paths_nodes = set()

  for path in paths_data:
    all_paths_nodes.update(path[0])

  # Change the font of the labels of the path nodes and restore alpha=None.
  for node, (x, y) in pos.items():
    if node in all_paths_nodes:
      plt.text(x, y, node, fontsize=14, ha='center', va='center')

  # 3. Draw the nodes of all the paths
  nx.draw_networkx_nodes(G, pos=pos, nodelist=all_paths_nodes, node_size=550,
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

  # Draw 'Source' & 'Sink' labels.
  # labels = {paths_data[0][0][0]: "Source", paths_data[0][0][-1]: "Sink"}
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
