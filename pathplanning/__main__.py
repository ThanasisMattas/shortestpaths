# __main__.py is part of PathPlanning
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
"""Produces a PathPlanning demo."""

import click

from pathplanning import dijkstra
from pathplanning import utils
from pathplanning.utils import PythonLiteralOption


@click.command()
@click.argument("num_nodes", type=click.INT)
@click.option("--weighted/--no-weighted", default=True, show_default=True)
@click.option("--weights-on", default="edges-and-nodes", show_default=True,
              type=click.Choice(["edges", "nodes", "edges-and-nodes"],
                                case_sensitive=False))
# @click.option("-p", "--probability",
#               default=0.5, show_default=True, type=click.FloatRange(0, 1),
#               help="the probability of edges exist (Erdős–Rényi model)")
@click.option("--max-edge-weight", default=1000, show_default=True,
              help="the max edge weight of the graph (defaults to 1000)")
@click.option("--max-node-weight", default=1000, show_default=True,
              help="the max nodal weight of the graph (defaults to 1000)")
@click.option("--num-paths", default=1, show_default=True,
              help="number of alternative paths to be generated")
@click.option("--adapted-path/--no-adapted-path",
              default=False, show_default=True,
              help="Generates a new path after disconnecting some nodes.")
@click.option("--disconnected-nodes",
              cls=PythonLiteralOption, default="[]", show_default=True,
              help=("Usage:\n\n"
                    "--disconnected-nodes '[id_1, id_2, ...]'"
                    "\n\n"
                    "An alternative path will be constructed, disregarding the"
                    " disconnected nodes."))
@click.option("--saving-states/--no-saving-states",
              default=True, show_default=True,
              help="Whether to use dynamic programming or not.")
@click.option('-s', "--seed", "random_seed", default=None, show_default=True,
              help="If provided, a fixed random graph will be generated")
@click.option("--layout-seed", default=1, show_default=True,
              help="Fixes the random initialization of the spirng_layout.")
@click.option("--show-graph/--no-show-graph", default=True, show_default=True)
@click.option("--save-graph/--no-save-graph", default=False, show_default=True)
def main(num_nodes,
         weighted,
         weights_on,
         max_edge_weight,
         max_node_weight,
         num_paths,
         adapted_path,
         disconnected_nodes,
         saving_states,
         random_seed,
         layout_seed,
         show_graph,
         save_graph):

  # 1. Preprocessing
  adj_list, G = utils.random_graph(num_nodes=num_nodes,
                                   weighted=weighted,
                                   weights_on=weights_on,
                                   max_edge_weight=max_edge_weight,
                                   max_node_weight=max_node_weight,
                                   random_seed=random_seed)
  disconnected_nodes = set(disconnected_nodes)

  # 2. Paths generation
  #
  # paths_data format:
  # [
  #  [path_1, path_1_cost, disconnected_nodes_1],
  #  [path_2, path_2_cost, disconnected_nodes_2],
  #  ...
  # ]
  paths_data = dijkstra.shortest_path(adj_list,
                                      num_nodes,
                                      start=1,
                                      goal=num_nodes,
                                      num_paths=num_paths,
                                      saving_states=saving_states,
                                      adapted_path=adapted_path,
                                      disconnected_nodes=disconnected_nodes,
                                      random_seed=random_seed)

  # 3. Post-processing
  click.echo(f"disconnected nodes: {disconnected_nodes}")
  for i, path in enumerate(paths_data):
    click.echo(f"path_{i}: {str(path[0])}")
    click.echo(f"path cost: {path[1]}")
    click.echo(f"avoided nodes: {str(path[2])}")
    click.echo()

  if save_graph or show_graph:
    utils.plot_graph(G,
                     paths_data,
                     disconnected_nodes,
                     save_graph,
                     show_graph,
                     layout_seed=layout_seed)

    # utils.plot_adaptive_dijkstra(G,
    #                              paths_data,
    #                              disconnected_nodes,
    #                              nodes_visited_sequence,
    #                              checkpoint_node)


if __name__ == "__main__":
  main()
