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


@click.command()
@click.argument("num_nodes", type=click.INT)
@click.option("--weighted/--no-weighted", "weighted",
              default=True, show_default=True)
@click.option("--weights-on", "weights_on",
              default="edges", show_default=True,
              type=click.Choice(["edges", "nodes", "edges-and-nodes"],
                                case_sensitive=False))
# @click.option("-p", "--probability",
#               default=0.5, show_default=True, type=click.FloatRange(0, 1),
#               help="the probability of edges exist (Erdős–Rényi model)")
@click.option("--max-edge-weight", "max_edge_weight",
              default=1000, show_default=True,
              help="the max edge weight of the graph (defaults to 100)")
@click.option("--max-node-weight", "max_node_weight",
              default=1000, show_default=True,
              help="the max nodal weight of the graph (defaults to 100)")
@click.option("--show-graph/--no-show-graph", "show_graph",
              default=True, show_default=True)
def main(num_nodes,
         weighted,
         weights_on,
         max_edge_weight,
         max_node_weight,
         show_graph):

  adj_list, G = utils.random_graph(num_nodes=num_nodes,
                                   weighted=weighted,
                                   weights_on=weights_on,
                                   max_edge_weight=max_edge_weight,
                                   max_node_weight=max_node_weight)

  path_cost, path = dijkstra.shortest_path(adj_list,
                                           num_nodes,
                                           start=1,
                                           goal=num_nodes)

  click.echo("path cost: {}".format(path_cost))
  click.echo("path: " + str(path))

  if show_graph:
    utils.plot_graph(G, path, path_cost)

if __name__ == '__main__':
  main()
