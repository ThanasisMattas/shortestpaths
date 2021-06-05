#!/usr/bin/env python3
"""
Visualization for comparing the search space of uni- and bi- directional
variations of Dijkstra's algorithm.

python plot_search_sphere.py -b -s 6 --layout-seed 3 --save-graph 500
{
  max_edge_weight=1000,
  max_node_weight=20,
  center_portion=0.15,
  gradient=0.75,
  p_0=0.23,
}
"""
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import click

from shortestpaths import dijkstra, graph_generator, post_processing


@click.command()
@click.argument('n', type=click.INT)
@click.option('-b', "--bidirectional", is_flag=True)
@click.option('-s', "--random-seed", type=click.INT,
              default=0, show_default=True)
@click.option("--layout-seed", type=click.INT,
              default=0, show_default=True)
@click.option("--save-graph", is_flag=True)
def main(n, bidirectional, random_seed, layout_seed, save_graph):

  # source, sink = 100, n - 100
  source, sink = 1, n

  adj_list, G = graph_generator.random_graph(
      n,
      weighted=True,
      directed=False,
      weights_on="edges-and-nodes",
      max_edge_weight=1000,
      max_node_weight=50,
      random_seed=random_seed
  )

  if bidirectional:
    adj_list_reverse = graph_generator.adj_list_reversed(adj_list)
  else:
    adj_list_reverse = None

  forward_config, reverse_config = dijkstra.dijkstra_init(
      source,
      sink,
      adj_list,
      adj_list_reverse
  )

  if bidirectional:
    prospect = [0, 0, 0, 0]
    top_reverse = 0

    while forward_config["to_visit"] and reverse_config["to_visit"]:
      # Forward step
      visited, top_forward, prospect = dijkstra._dijkstra_step(
          **forward_config,
          opposite_visited=reverse_config["visited"],
          prospect=prospect,
          is_forward=True
      )
      if top_forward + top_reverse > prospect[0] != 0:
        break

      # Reverse step
      visited_reverse, top_reverse, prospect = dijkstra._dijkstra_step(
          **reverse_config,
          opposite_visited=forward_config["visited"],
          prospect=prospect,
          is_forward=False
      )
      if top_forward + top_reverse > prospect[0] != 0:
        break

    path, _ = dijkstra.extract_bidirectional_path(
        source,
        sink,
        prospect,
        {},
        visited=visited,
        visited_reverse=visited_reverse
    )
    meeting_edge_head = prospect[2]
  else:
    visited = dijkstra.dijkstra(**forward_config)
    path, _ = dijkstra.extract_path(source, sink, visited, {})
    visited_reverse = None
    meeting_edge_head = None

  # post_processing.state_vis(forward_config["to_visit"],
  #                           visited,
  #                           source,
  #                           sink,
  #                           G=G)

  post_processing.plot_search_sphere(G,
                                     visited,
                                     path,
                                     show_graph=True,
                                     save_graph=save_graph,
                                     layout_seed=layout_seed,
                                     visited_reverse=visited_reverse,
                                     meeting_edge_head=meeting_edge_head)


if __name__ == '__main__':
  main()
