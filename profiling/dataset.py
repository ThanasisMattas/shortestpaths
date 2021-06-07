#!/usr/bin/env python3
import os
from pathlib import Path
import sys

import click

cwd = Path(os.getcwd())
home_dir = str(cwd.parent)
sys.path.insert(0, home_dir)

from shortestpaths import graph_generator, io


@click.command()
@click.option('-f', "--filename", type=click.INT, default=None, show_default=True)
@click.option("--clear", is_flag=True, default=True, prompt="Delete dataset with the same name?")
@click.option('-n', "--initial-order", default=50, show_default=True,
              help="Initial graph order")
@click.option('-s', "--step", default=25, show_default=True,
              help="graph order increase step")
@click.option('-i', "--increases", default=31, show_default=True)
@click.option('-g', "--graphs-per-step", default=10, show_default=True)
@click.option('-c', "center_portion", default=0.15, show_default=True)
@click.option("--gradient", default=0.3, show_default=True)
@click.option("--directed", is_flag=True)
@click.option("--weights-on", default="edges-and-nodes", show_default=True,
              type=click.Choice(["edges", "nodes", "edges-and-nodes"],
                                case_sensitive=False))
@click.option("--max-edge-weight", default=1000, show_default=True)
@click.option("--max-node-weight", default=50, show_default=True)
def dataset(filename,
            clear,
            initial_order,
            step,
            increases,
            graphs_per_step,
            **kwargs):

  num_graphs = increases * graphs_per_step * 10
  if filename is None:
    filename = f"dataset_{num_graphs}.csv"

  if clear:
    if os.path.isfile(filename):
      os.remove(filename)

  graph_counter = 0
  for n in range(initial_order, initial_order + step * increases, step):
    for p in range(1, 11):
      graph_counter += 1
      for g in range(1, graphs_per_step + 1):
        print((f"Graph: {graph_counter:>3}/{num_graphs // graphs_per_step}"
               f"   Instance: {g:>2}/{graphs_per_step}"
               f"   nodes: {n:>4}   p\u2080: {p / 10:.1f}"),
              end='\r')
        adj_list, _ = graph_generator.random_graph(n=n,
                                                   p_0=p / 10,
                                                   random_seed=g,
                                                   **kwargs)
        io.append_graph_to_csv(filename, adj_list)
  print()


if __name__ == '__main__':
  dataset()
