import gc
import os
from pathlib import Path
import sys
from time import process_time

cwd = Path(os.getcwd())
home_dir = str(cwd.parent)
sys.path.insert(0, home_dir)

import click

from shortestpaths import core, graph_generator, post_processing


@click.command()
@click.argument("num_graphs", type=click.INT)
@click.argument("n", type=click.INT)
@click.option("--unweighted", "weighted", is_flag=True,
              default=True, show_default="weighted")
@click.option("--weights-on", default="edges-and-nodes", show_default=True,
              type=click.Choice(["edges", "nodes", "edges-and-nodes"],
                                case_sensitive=False))
@click.option("--max-edge-weight", default=1000, show_default=True,
              help="the max edge weight of the graph (defaults to 1000)")
@click.option("--max-node-weight", default=50, show_default=True,
              help="the max nodal weight of the graph (defaults to 1000)")
@click.option("-k", type=click.INT, default=1, show_default=True,
              help="number of alternative paths to be generated")
@click.option('-b', "--bidirectional", is_flag=True,
              help="bidirectional shortest path search (uses 2 processes)")
@click.option('-p', "--parallel", is_flag=True,
              help="whether to use multiprocessing or not")
@click.option('-m', "--memoize-states", is_flag=True,
              help="whether to use dynamic programming or not")
@click.option('-s', "--seed", "random_seed", type=click.INT,
              default=None, show_default=True,
              help="If provided, a fixed random graph will be generated.")
def main(num_graphs,
         n,
         weighted,
         weights_on,
         max_edge_weight,
         max_node_weight,
         k,
         bidirectional,
         parallel,
         memoize_states,
         random_seed):

  steps = 50
  un_time = [0. for _ in range(num_graphs)]
  bi_time = [0. for _ in range(num_graphs)]

  un_time_mean = [0. for _ in range(steps)]
  bi_time_mean = [0. for _ in range(steps)]
  ns = [n + i * 20 for i in range(steps)]

  for j in range(steps):
    for i in range(num_graphs):
      # print(f"graph #{i}")
      if random_seed is None:
        random_seed == i

      adj_list, G = graph_generator.random_graph(n=n,
                                                 weighted=weighted,
                                                 weights_on=weights_on,
                                                 max_edge_weight=max_edge_weight,
                                                 max_node_weight=max_node_weight,
                                                 random_seed=random_seed)

      un_start = process_time()
      k_paths = core.k_shortest_paths(adj_list=adj_list,
                                      source=1,
                                      sink=n,
                                      k=k,
                                      bidirectional=False,
                                      parallel=parallel,
                                      memoize_states=memoize_states,
                                      random_seed=random_seed)
      un_time[i] = process_time() - un_start

      bi_start = process_time()
      b_paths = core.k_shortest_paths(adj_list=adj_list,
                                      source=1,
                                      sink=n,
                                      k=k,
                                      bidirectional=True,
                                      parallel=parallel,
                                      memoize_states=memoize_states,
                                      random_seed=random_seed)
      bi_time[i] = process_time() - bi_start

      if k_paths[0][1] != b_paths[0][1]:
        print(f"graph: {i}")
        post_processing.print_paths(k_paths)
        post_processing.print_paths(b_paths)

      # print()
      gc.collect()
    un_time_mean[j] = sum(un_time) / num_graphs
    bi_time_mean[j] = sum(bi_time) / num_graphs
    n += 20

  import matplotlib.pyplot as plt

  plt.plot(ns, un_time_mean, color='r')
  plt.plot(ns, bi_time_mean, color='b')
  diff = [un_time_mean[ll] - bi_time_mean[ll] for ll in range(steps)]
  plt.plot(ns, diff, color='g')
  plt.savefig("uni-vs_bi-_directional.png")
  plt.show()

if __name__ == "__main__":
  main()
