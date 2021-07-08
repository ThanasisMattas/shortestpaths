#!/usr/bin/env python3
import os
from pathlib import Path
import pickle
import sys

import click

cwd = Path(os.getcwd())
home_dir = str(cwd.parent)
sys.path.insert(0, home_dir)

from shortestpaths import graph


def dump_graph(fileobj, n, p, g, **kwargs):
  """Creates and dumps a graph to the open pickle file.

  NOTE: It is important to create the graph here, in order to be garbage
        collected upon exit.
  """
  adj_list, _ = graph.random_graph(n=n,
                                   p_0=p / 10,
                                   random_seed=g,
                                   **kwargs)
  pickle.dump(adj_list, fileobj)
  # io.append_graph_to_csv(filename, adj_list)


@click.command()
@click.option('-f', "--filename", type=click.INT, default=None, show_default=True)
@click.option("--clear", is_flag=True, default=True, prompt="Delete dataset with the same name?")
@click.option('-n', "--n-init", default=500, show_default=True,
              help="Initial graph order")
@click.option('-s', "--step", default=250, show_default=True,
              help="graph order increase step")
@click.option("--p-step", default=2, show_default=True)
@click.option('-i', "--increases", default=30, show_default=True)
@click.option('-g', "--graphs-per-step", default=10, show_default=True)
@click.option('-c', "center_portion", default=0.15, show_default=True)
@click.option("--gradient", default=0.3, show_default=True)
@click.option("--directed", is_flag=True)
@click.option("--weights-on", default="edges-and-nodes", show_default=True,
              type=click.Choice(["edges", "nodes", "edges-and-nodes"],
                                case_sensitive=False))
@click.option("--max-edge-weight", default=1000, show_default=True)
@click.option("--max-node-weight", default=50, show_default=True)
def main(filename,
         clear,
         n_init,
         step,
         p_step,
         increases,
         graphs_per_step,
         **kwargs):

  directed = "directed" if kwargs["directed"] else "undirected"
  n_range = range(n_init, n_init + step * (increases + 1), step)
  p_range = range(1, 11, p_step)

  if filename is None:
    filename = (f"dataset_{len(n_range)}x{len(p_range)}"
                f"x{graphs_per_step}_npg_{directed}.dat")

  if clear:
    if os.path.isfile(filename):
      os.remove(filename)

  with open(filename, "wb") as f:
    for i, n in enumerate(n_range):
      for j, p in enumerate(p_range):
        for g in range(1, graphs_per_step + 1):
          print(
            (f"Graph: {i * len(p_range) + j + 1:>3}"
             f"/{len(p_range) * (increases + 1)}"
             f"   nodes: {n:>4}   p\u2080: {p / 10:.1f}"
             f"   Instance: {g:>2}/{graphs_per_step}"),
            end='\r'
          )
          dump_graph(f, n, p, g, **kwargs)
  print()


if __name__ == '__main__':
  main()
