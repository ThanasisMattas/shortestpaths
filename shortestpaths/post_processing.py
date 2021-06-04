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
from math import sqrt
import os

import click
import matplotlib.pyplot as plt
import networkx as nx

from shortestpaths import dijkstra, graph_generator, utils


COLORS = [
    # "dimgray",
    # "#990011FF"  # spacecherry
    "mediumblue",
    "#c71585",  # redviolet
    "aqua",
    'k',
    "r",
    "darkorange",
    "#bf00ff",  # electric purple
    "limegreen"
]

def path_label(path, path_number, failing):
  if (failing is None) or (len(path) == 2):
      failed_msg = ''
  elif path[3] is None:
    # then nothing failed, because this is the absolute shortest path
    failed_msg = f"failed {failing[:-1]}: -"
  elif (isinstance(path[3], tuple)) and (len(path[3]) == 2):
    # then it is a single edge
    failed_msg = f"failed {failing[:-1]}: {str(path[3])}"
  elif not hasattr(path[3], "__len__"):
    failed_msg = f"failed {failing[:-1]}: {str(path[3])}"
  else:
    # then it is a list of failed nodes or edges
    failed_msg = f"failed {failing}: {str(list(path[3]))}"
  label = (f"path_{path_number}: {str(path[0])}\ncost: {path[1]}"
           f"    {failed_msg}")
  return label


def _node_sizes(G):
  if G.number_of_nodes() < 400:
    node_size = 550
    path_node_size = 1200
    failed_node_size = 1600
  elif G.number_of_nodes() < 2200:
    node_size = 550 - G.number_of_nodes() // 5
    path_node_size = 1200 - G.number_of_nodes() // 10
    failed_node_size = 1600 - G.number_of_nodes() // 10
  else:
    node_size = 10
    path_node_size = 350
    failed_node_size = 550
  return node_size, path_node_size, failed_node_size


def visited_nodes(visited, source):
  """Returns a list of the visited nodes."""
  nodes = [source]
  for u in range(1, len(visited)):
    if dijkstra.was_visited(visited, u):
      nodes.append(u)
  return nodes


def _xylimits(pos):
  "Crops the inner margins of the frame."""
  x_values = [x for x, y in pos.values()]
  y_values = [y for x, y in pos.values()]
  xmin = 1.05 * min(x_values)
  xmax = 1.05 * max(x_values)
  ymin = 1.1 * min(y_values)
  ymax = 1.1 * max(y_values)
  return xmin, xmax, ymin, ymax


def plot_paths(paths_data,
               G,
               mode,
               save_graph=False,
               show_graph=True,
               layout_seed=None,
               draw_edge_weights=False):
  """Plots the graph and all the generated paths (up to 8) in spring_layout."""
  utils.verify_paths(paths_data)
  if save_graph:
    figsize = (10 * 1.8, 10)
    dpi = 200
    title_fontsize = 22
    legend_fontsize = 20
  else:
    figsize = (8 * 1.8, 8)
    dpi = 100
    title_fontsize = 18
    legend_fontsize = 16

  # path_nodes_pos = [
  #     (1, -1),
  #     (0.7, -0.1),
  #     (0.35, -0.5),
  #     (0.15, 0),
  #     (-0.1, -0.1),
  #     (-0.4, 0.3),
  #     (-0.6, -0.1),
  #     (-1, 1.1)
  # ]
  # path_nodes_pos = {paths_data[0][0][i]: path_nodes_pos[i] for i in range(8)}

  fig = plt.figure(figsize=figsize, dpi=dpi)
  # , k=10 / sqrt(G.number_of_nodes())  # the default spring coefficient
  pos = nx.spring_layout(
      G,
      seed=layout_seed,
      # pos=path_nodes_pos,
      k=10 / sqrt(G.number_of_nodes()),
      # fixed=path_nodes_pos.keys()
  )

  # Layouts
  # -------
  # circular_layout
  # spring_layout                <--
  # fruchterman_reingold_layout  <--
  # spiral_layout                <--

  # 1. Draw the graph
  node_size, path_node_size, failed_node_size = _node_sizes(G)
  nx.draw_networkx(G, pos, node_size=node_size, width=0.3, alpha=0.3,
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
      plt.text(x, y, node, fontsize=19, ha='center', va='center')
  nx.draw_networkx_nodes(G, pos=pos,
                         nodelist=all_paths_nodes, node_size=path_node_size,
                         edgecolors='k', node_color="deepskyblue", alpha=0.8)

  # 3. Draw the paths
  colors = iter(COLORS)
  width_step = 6.5
  last_path_width = 4
  first_path_width = max(last_path_width + (len(paths_data) - 1) * width_step,
                         8)
  for i, path in enumerate(paths_data):
    color = next(colors)
    label = path_label(path, i + 1, mode["failing"])
    path_edges_sequence = list(zip(path[0], path[0][1:]))
    # arrows=False, arrowsize=20, arrowstyle='fancy',
    # min_source_margin=1, min_target_margin=1,
    # from matplotlib.patches import ConnectionStyle
    # connectionstyle=ConnectionStyle("Arc3", rad=0.2),
    nx.draw_networkx_edges(G, pos=pos, edgelist=path_edges_sequence,
                           edge_color=color, alpha=0.8, arrows=False,
                           width=first_path_width - i * width_step,
                           label=label)

    # Mark the disconnceted edge or node with an ×.
    if mode["failing"] == "nodes":
      if (len(path) > 2) and (path[3] not in [None, [None]]):
        if hasattr(path[3], "__len__"):
          nodelist = path[3]
        else:
          nodelist = [path[3]]
        nx.draw_networkx_nodes(G, pos=pos, nodelist=nodelist,
                               node_color=color, node_shape='x',
                               node_size=failed_node_size, linewidths=5)
    elif mode["failing"] == "edges":
      # Check for the case of the absolute shortest path, where there is no
      # disconnected edge.
      if (len(path) > 2) and (path[3] is not None):  # ✕×✗
        nx.draw_networkx_edge_labels(G, pos, edge_labels={path[3]: '×'},
                                     font_size=60, font_color=color,
                                     bbox=dict(alpha=0), rotate=False)
    elif mode["failing"] is None:
      pass
    else:
      raise ValueError("failing should be 'edges', 'nodes' or None")

  if (len(path) > 2) and (path[3] is not None):
    online_status = "on-line" if mode.get('online') else "off-line"
    frame_title = ("Replacement-paths\n"
                   f"mode: {online_status} / failing {mode.get('failing')}")
  else:
    frame_title = "\nk-shortest paths"

  frame_title += (f"\n#nodes: {G.number_of_nodes()}   "
                  f"#edges: {G.number_of_edges()}   "
                  f"#paths: {len(paths_data)}")
  plt.title(frame_title, fontsize=title_fontsize)
  leg = plt.legend(fontsize=legend_fontsize)
  leg.get_frame().set_alpha(None)
  leg.get_frame().set_facecolor((1, 1, 1, 0.5))

  plt.tight_layout()
  xmin, xmax, ymin, ymax = _xylimits(pos)
  plt.xlim(xmin, xmax)
  plt.ylim(ymin, ymax)

  if save_graph:
    date_n_time = str(datetime.now())[:19]
    date_n_time = date_n_time.replace(':', '-').replace(' ', '_')
    file_name = f"graph_vis_{date_n_time}.png"
    plt.savefig(os.path.join(os.getcwd(), file_name), dpi=fig.dpi)
  if show_graph:
    plt.show()


def print_paths(paths, failing=None):
  path_str_len = 0
  cost_str_len = 0
  num_paths_str_len = len(str(len(paths)))
  for path in paths:
    path_str_len = max(path_str_len, len(str(path[0])))
    cost_str_len = max(cost_str_len, len(str(path[1])))

  for k, path in enumerate(paths):
    msg = (f"path {k + 1:>{num_paths_str_len}}:"
           f" {str(path[0]):{path_str_len}}   "
           f"cost: {path[1]:>{cost_str_len}}")
    if failing:
      msg += f"   failed {failing[:-1]}: {path[3]}"
    click.echo(msg)


def state_retrieval_vis(G,
                        paths_data,
                        visited_nodes_forward,
                        visited_nodes_reverse,
                        retrieved_nodes_forward,
                        retrieved_nodes_reverse,
                        meeting_edge: tuple,
                        mode,
                        layout_seed,
                        show_graph,
                        save_graph):
  if save_graph:
    figsize = (10 * 1.8, 10)
    dpi = 200
    title_fontsize = 23
    legend_fontsize = 21
  else:
    figsize = (8 * 1.8, 8)
    dpi = 100
    title_fontsize = 18
    legend_fontsize = 16

  fig = plt.figure(figsize=figsize, dpi=dpi)
  pos = nx.spring_layout(G,
                         seed=layout_seed,
                         k=20 / sqrt(G.number_of_nodes()))

  # 1. Draw the graph
  # Exclude the visited nodes
  nodelist = set(G.nodes()).difference(visited_nodes_forward).difference(visited_nodes_reverse)
  node_size, path_node_size, failed_node_size = _node_sizes(G)
  nx.draw_networkx(G, pos, node_size=node_size, width=0.3, alpha=0.3,
                   nodelist=nodelist, with_labels=False, arrows=False)

  # 2. Draw the forward state
  nx.draw_networkx_nodes(G, pos=pos, nodelist=visited_nodes_forward,
                         node_color='goldenrod', alpha=0.8, node_size=2800,
                         linewidths=0, label="forward visited")
  nx.draw_networkx_nodes(G, pos=pos, nodelist=retrieved_nodes_forward,
                         node_color='g', alpha=0.7, node_size=2000,
                         linewidths=0, label="forward retrieved")

  # 3. Draw the reverse state
  # background        : foreground
  # ------------------------------
  # 603F83FF 343148FF : C7D3D4FF
  # 192e5b            : 72a2c0
  # 00539CFF          : 9CC3D5FF B1624EFF A2A2A1FF
  nx.draw_networkx_nodes(G, pos=pos, nodelist=visited_nodes_reverse,
                         node_color='#132226', alpha=0.7, node_size=2800,
                         linewidths=0, label="reverse visited")

  nx.draw_networkx_nodes(G, pos=pos, nodelist=retrieved_nodes_reverse,
                         node_color='#f2eaed', alpha=0.7, node_size=2000,
                         linewidths=0, label="reverse retrieved")

  # 4. Draw the nodes of all the paths
  all_paths_nodes = set()
  for path in paths_data:
    all_paths_nodes.update(path[0])
  # Change the font of the labels of the path nodes and restore alpha=None.
  for node, (x, y) in pos.items():
    if node in all_paths_nodes:
      plt.text(x, y, node, fontsize=19, ha='center', va='center')
  nx.draw_networkx_nodes(G, pos=pos,
                         nodelist=all_paths_nodes, node_size=path_node_size,
                         edgecolors='k', node_color="deepskyblue")

  # 5. Draw the paths
  colors = iter(['k', 'r'])
  edgewidth_max = 14
  edgewidth_step = 8
  for i, path in enumerate(paths_data):

    color = next(colors)
    label = path_label(path, i + 1, mode["failing"])
    path_edges_sequence = list(zip(path[0], path[0][1:]))
    nx.draw_networkx_edges(G, pos=pos, edgelist=path_edges_sequence,
                           edge_color=color, alpha=0.8, arrows=False,
                           width=edgewidth_max - edgewidth_step * i,
                           label=label)

    # Mark the disconnceted edge or node with an ×.
    if mode["failing"] == "nodes":
      if (len(path) > 2) and (path[3] not in [None, [None]]):
        if hasattr(path[3], "__len__"):
          disconnected = path[3]
        else:
          disconnected = [path[3]]
        nx.draw_networkx_nodes(G, pos=pos, nodelist=disconnected,
                               node_color=color, node_shape='x',
                               node_size=failed_node_size, linewidths=5)
    elif mode["failing"] == "edges":
      # Check for the case of the absolute shortest path, where there is no
      # disconnected edge.
      if (len(path) > 2) and (path[3] is not None):  # ✕×✗
        nx.draw_networkx_edge_labels(G, pos, edge_labels={path[3]: '×'},
                                     font_size=70, font_color=color,
                                     bbox=dict(alpha=0), rotate=False)
    elif mode["failing"] is None:
      pass
    else:
      raise ValueError("failing should be 'edges', 'nodes' or None")

  # 6. Draw the meeting edge
  nx.draw_networkx_edges(G, pos=pos, edgelist=[meeting_edge],
                         edge_color="coral", alpha=0.8, arrows=False,
                         width=edgewidth_max - edgewidth_step,
                         label=f"meeting edge: {meeting_edge}")

  leg = plt.legend(fontsize=legend_fontsize)
  leg.get_frame().set_alpha(None)
  leg.get_frame().set_facecolor((1, 1, 1, 0.5))
  plt.title("State retrieval", fontsize=title_fontsize)

  plt.tight_layout()
  xmin, xmax, ymin, ymax = _xylimits(pos)
  plt.xlim(xmin, xmax)
  plt.ylim(ymin, ymax)

  if ((visited_nodes_forward == retrieved_nodes_forward)
          and (visited_nodes_reverse == retrieved_nodes_reverse)):
    failed_edge = "_me"
  else:
    failed_edge = ""

  if save_graph:
    file_name = f"state_retrieval_vis{failed_edge}.png"
    plt.savefig(os.path.join(os.getcwd(), file_name), dpi=fig.dpi)
  if show_graph:
    plt.show()


def state_vis(to_visit, visited, source, sink, layout_seed=1, G=None):
  """Generates a Dijkstra's algorithm state visualization."""
  to_visit_nodes = to_visit.keys()
  # visited_nodes_list = G.nodes.difference(to_visit_nodes)
  visited_nodes_list = visited_nodes(visited, source)
  visited_height = [0 for _ in range(1, len(visited))]
  to_visit_height = [0 for _ in range(1, len(visited))]
  sink_height = []
  for i, entry in enumerate(visited[1:]):
    node = i + 1
    if node == source:
      continue
    elif node == sink:
      try:
        sink_height.append(to_visit[sink][0])
      except KeyError:
        sink_height.append(visited[sink][0])
    elif entry[0] == 0:
      to_visit_height[node] = to_visit[node][0]
    else:
      visited_height[node] = entry[0]

  if G is None:
    fig, ax = plt.subplots(figsize=(8 * 1.618, 8))
    ax.bar(range(1, len(visited)), visited_height)
    ax.bar(range(1, len(visited)), to_visit_height)
    ax.bar([sink], sink_height, color='r')
  else:
    fig, ax = plt.subplots(1, 2, figsize=(8 * 1.618, 8))
    ax[0].bar(range(1, G.number_of_nodes() + 1), visited_height)
    ax[0].bar(range(1, G.number_of_nodes() + 1), to_visit_height)
    ax[0].bar([sink], sink_height, color='r')
    ax[0].set_xlabel("node id")
    ax[0].set_ylabel("path_to_node_cost")

    pos = nx.spring_layout(G, seed=layout_seed)
    node_size, path_node_size, failed_node_size = _node_sizes(G)
    nx.draw_networkx(G, pos, node_size=node_size, width=0.2, alpha=0.3,
                     arrows=False, with_labels=False, ax=ax[1])
    for node, (x, y) in pos.items():
      plt.text(x, y, node, fontsize=14, ha='center', va='center')

    # add visited
    nx.draw_networkx_nodes(G, pos=pos, nodelist=visited_nodes_list,
                           edgecolors='k', node_color="deepskyblue",
                           node_size=path_node_size,
                           label="visited")
    # add not visited
    nx.draw_networkx_nodes(G, pos=pos, nodelist=to_visit_nodes,
                           edgecolors='k', node_color="#ff7f0e",
                           node_size=path_node_size,
                           label="not visited")

  plt.tight_layout()
  fig.suptitle("State visualization", fontsize=16)
  plt.legend()
  plt.show()


def plot_search_sphere(G,
                       visited,
                       path,
                       show_graph=True,
                       save_graph=False,
                       layout_seed=0,
                       visited_reverse=None,
                       meeting_edge_head=None):
  """Plots the visited nodes of the uni/bi-directional search."""
  visited_nodes_forward = visited_nodes(visited, path[0])
  num_visited_forward = len(visited_nodes_forward)
  if visited_reverse is None:
    algorithm = "Dijkstra's algorithm"
    file_name = "dijkstra"
    path_forward = path
  else:
    algorithm = "Bidirectional Dijkstra's algorithm"
    file_name = "bidirectional"
    visited_nodes_reverse = visited_nodes(visited_reverse, path[-1])
    num_visited_reverse = len(visited_nodes_reverse)
    meeting_edge_head_idx = path.index(meeting_edge_head)
    path_forward = path[:meeting_edge_head_idx]
    path_reverse = path[meeting_edge_head_idx:]

  fig = plt.figure(figsize=(10 * 1.8, 10))
  pos = nx.spring_layout(G,
                         seed=layout_seed,
                         k=80 / sqrt(G.number_of_nodes()))

  # 1. Draw the graph
  node_size, path_node_size, failed_node_size = _node_sizes(G)
  nx.draw_networkx(G, pos, node_size=node_size, width=0.15, alpha=0.3,
                   with_labels=False, arrows=False)

  # 2. Draw the visited nodes
  nx.draw_networkx_nodes(G, pos=pos, nodelist=visited_nodes_forward,
                         node_color='goldenrod', node_size=650, linewidths=0,
                         label="forward search", alpha=0.6)
  if visited_reverse is not None:
    nx.draw_networkx_nodes(G, pos=pos, nodelist=visited_nodes_reverse,
                           node_color='g', node_size=650, linewidths=0,
                           label="reverse search", alpha=0.6)

  # 3. Draw the path-nodes
  for node, (x, y) in pos.items():
    if node in path:
      plt.text(x, y, node, fontsize=14, ha='center', va='center')

  nx.draw_networkx_nodes(G, pos=pos, nodelist=path_forward, edgecolors='k',
                         node_size=path_node_size, node_color="goldenrod")
  if visited_reverse is not None:
    nx.draw_networkx_nodes(G, pos=pos, nodelist=path_reverse, edgecolors='k',
                           node_size=path_node_size, node_color="g")

  # 4. Draw the path
  if visited_reverse:
    forward_edge_color = "darkgoldenrod"
    label = f"forward subpath: {path_forward}"
  else:
    forward_edge_color = 'k'
    label = f"path: {path_forward}"
  path_edges_sequence = list(zip(path_forward[:-1], path_forward[1:]))
  nx.draw_networkx_edges(G, pos=pos, edgelist=path_edges_sequence,
                         edge_color=forward_edge_color, arrows=False,
                         width=6, label=label)
  if visited_reverse is not None:
    # then, draw the reverse subpath
    path_edges_sequence = list(zip(path_reverse[:-1], path_reverse[1:]))
    nx.draw_networkx_edges(G, pos=pos, edgelist=path_edges_sequence,
                           edge_color='darkgreen', arrows=False,
                           width=6, label=f"reverse subpath: {path_reverse}")
    meeting_edge = [(path_forward[-1], path_reverse[0])]
    nx.draw_networkx_edges(G, pos=pos, edgelist=meeting_edge,
                           edge_color='k', arrows=False,
                           width=5, label=f"meeting edge: {meeting_edge[0]}")

  if visited_reverse is None:
    num_visited_total = num_visited_forward
  else:
    num_visited_total = num_visited_forward + num_visited_reverse
  frame_title = (f"{algorithm} search space\n"
                 f"n: {G.number_of_nodes()}"
                 f"   m: {G.number_of_edges()}"
                 f"   nodes visited: {num_visited_total}")
  plt.title(frame_title)
  plt.legend()

  if save_graph:
    date_n_time = str(datetime.now())[:19]
    date_n_time = date_n_time.replace(':', '-').replace(' ', '_')
    file_name = (f"search_sphere_{file_name}_{date_n_time}.png")
    plt.savefig(os.path.join(os.getcwd(), file_name), dpi=fig.dpi)
  if show_graph:
    plt.show()


def graph_density_contour(n, directed=True, weights_on="edges-and-nodes"):
  """Plots density(sigmoid_center, initial_probability)."""
  import numpy as np
  densities = [[0] * 10 for _ in range(10)]

  for c in range(1, 11):
    for p in range(1, 11):
      G, probs, edge_lengths, edge_lengths_true = graph_generator.random_graph(
          n,
          directed=directed,
          weights_on=weights_on,
          random_seed=1,
          center_portion=c / 10,
          gradient=1,
          p_0=p / 10,
          get_probability_distribution=True
      )
      densities[c - 1][p - 1] = graph_generator.graph_density(
          n,
          len(edge_lengths_true),
          directed
      )
  plt.figure(figsize=(10, 10))
  x = y = np.arange(0.1, 1.1, 0.1)
  X, Y = np.meshgrid(x, y)
  cm = plt.cm.get_cmap('viridis')
  levels = np.arange(0, 1.1, 0.1)

  cp = plt.contour(X, Y, densities, levels=levels, colors="black", linewidths=0.5)
  plt.clabel(cp, fmt="%1.1f", fontsize=20, rightside_up=False, manual=True)
  plt.contourf(X, Y, densities, cmap=cm, levels=levels)

  plt.xlabel("${p_0}$", fontsize=20)
  plt.ylabel("center_portion", fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  plt.title(f"Graph density   n: {n}", fontsize=22)
  plt.show()
