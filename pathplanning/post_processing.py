import datetime
from math import sqrt
import os

import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(G,
               paths_data,
               disconnected_nodes,
               save_graph,
               show_graph,
               layout_seed=None,
               draw_edge_weights=False):
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
  if draw_edge_weights:
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

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

  # Draw 'source' & 'Sink' labels.
  # labels = {paths_data[0][0][0]: "source", paths_data[0][0][-1]: "Sink"}
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
