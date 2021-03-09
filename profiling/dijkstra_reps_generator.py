import gc
import os
from pathlib import Path
import sys

cwd = Path(os.getcwd())
home_dir = str(cwd.parent)
sys.path.insert(0, home_dir)

import click

from pathplanning import dijkstra
from pathplanning import utils


@click.command()
@click.argument("num_graphs", type=click.INT)
@click.argument("num_nodes", type=click.INT)
@click.option("--weighted/--no-weighted", default=True, show_default=True)
@click.option("--weights-on", default="edges-and-nodes", show_default=True,
              type=click.Choice(["edges", "nodes", "edges-and-nodes"],
                                case_sensitive=False))
@click.option("--max-edge-weight", default=1000, show_default=True,
              help="the max edge weight of the graph (defaults to 1000)")
@click.option("--max-node-weight", default=1000, show_default=True,
              help="the max nodal weight of the graph (defaults to 1000)")
@click.option("--num-paths", default=1, show_default=True,
              help="number of alternative paths to be generated")
@click.option("--adapted-path/--no-adapted-path",
              default=False, show_default=True,
              help="Generates a new path after disconnecting some nodes.")
@click.option("--saving-states/--no-saving-states",
              default=True, show_default=True,
              help="Whether to use dynamic programming or not.")
@click.option('-s', "--seed", "random_seed", default=None, show_default=True,
              help="If provided, a fixed random graph will be generated")
def main(num_graphs,
         num_nodes,
         weighted,
         weights_on,
         max_edge_weight,
         max_node_weight,
         num_paths,
         adapted_path,
         saving_states,
         random_seed):

  for i in range(num_graphs):
    print(f"graph #{i}")
    num_nodes += bool(i) * 5
    adj_list, G = utils.random_graph(num_nodes=num_nodes,
                                     weighted=weighted,
                                     weights_on=weights_on,
                                     max_edge_weight=max_edge_weight,
                                     max_node_weight=max_node_weight,
                                     random_seed=i)

    paths_data = dijkstra.shortest_path(adj_list=adj_list,
                                        num_nodes=num_nodes,
                                        start=1,
                                        goal=num_nodes,
                                        num_paths=num_paths,
                                        saving_states=saving_states,
                                        adapted_path=adapted_path,
                                        random_seed=i)

    gc.collect()


if __name__ == "__main__":
  main()
