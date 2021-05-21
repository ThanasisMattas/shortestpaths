"""
usage:
$ python shortestpaths_reps.py [OPTIONS] n

where n is the 1st graph size

python shortestpaths_reps.py -s 250 -i 5 -g 3 500
"""

import gc
import os
from pathlib import Path
import sys
from timeit import default_timer as timer


cwd = Path(os.getcwd())
home_dir = str(cwd.parent)
sys.path.insert(0, home_dir)

import click
import matplotlib.pyplot as plt

from shortestpaths import core, graph_generator


@click.command()
@click.argument('n', type=click.INT)
@click.option('-s', "--increase-step", default=25, show_default=True,
              help="number of nodes the graph size is increased by")
@click.option('-i', "--graph-increases", default=50, show_default=True,
              help="number of graph sizes formed by <increase-step>")
@click.option('-g', "graphs_per_step", default=5, show_default=True,
              help=("number of different but equal sized graphs to run at each"
                    "step (the time measure will be the average)"))
@click.option("--unweighted", "weighted", is_flag=True,
              default=True, show_default="weighted")
@click.option("--weights-on", default="edges-and-nodes", show_default=True,
              type=click.Choice(["edges", "nodes", "edges-and-nodes"],
                                case_sensitive=False))
@click.option("--max-edge-weight", default=1000, show_default=True,
              help="the max edge weight of the graph (defaults to 1000)")
@click.option("--max-node-weight", default=50, show_default=True,
              help="the max nodal weight of the graph (defaults to 1000)")
@click.option('-f', "--failing", default="nodes", show_default=True)
@click.option("--online", is_flag=True)
@click.option('-k', type=click.INT, default=1, show_default=True,
              help="number of alternative paths to be generated")
def main(n,
         increase_step,
         graph_increases,
         graphs_per_step,
         weighted,
         weights_on,
         max_edge_weight,
         max_node_weight,
         failing,
         online,
         k):

  mode = {
    # 'reference': [False, False, False],
    # 'parallel': [False, True, False],
    'bidirectional': [True, False, False],
    # 'bidirectional & parallel': [True, True, False],
    'dynamic': [True, False, True],
    # 'dynamic, bidirectional & parallel': [True, True, True]
  }

  times = {k: [] for k, v in mode.items()}

  times_mean = {k: [0. for _ in range(graph_increases)]
                for k, v in mode.items()}
  n_values = [n + i * increase_step for i in range(graph_increases)]

  for j in range(graph_increases):
    print(f"graph {j + 1}/{graph_increases}  nodes: {n}")
    for i in range(graphs_per_step):
      print(f"  {i + 1}/{graphs_per_step}")

      adj_list, G = graph_generator.random_graph(
        n=n,
        weighted=weighted,
        weights_on=weights_on,
        max_edge_weight=max_edge_weight,
        max_node_weight=max_node_weight,
        random_seed=i
      )

      for k, v in mode.items():
        start = timer()
        paths = core.replacement_paths(adj_list=adj_list,
                                       n=n,
                                       source=1,
                                       sink=n,
                                       failing=failing,
                                       bidirectional=v[0],
                                       parallel=v[1],
                                       dynamic=v[2],
                                       online=online)
        end = timer()
        times[k].append(end - start)

        gc.collect()

    for k, v in mode.items():
      times_mean[k][j] = sum(times[k]) / graphs_per_step
      del times[k][:]
    n += increase_step

  colors = iter(['b', 'm', 'g', 'k', 'c', 'y'])
  fig, ax = plt.subplots(figsize=(10.03, 6.2), dpi=200)
  for k, v in mode.items():
    color = next(colors)
    ax.plot(n_values, times_mean[k], color=color, label=k)
  ax.set_xlabel("graph size (nodes)")
  ax.set_ylabel("wall clock time (s)")
  ax.grid('major')
  ax.grid('minor', axis='y')
  fig.suptitle("replacement-paths profiling")
  fig.legend()
  plt.savefig("replacement_paths_profiling.png")
  plt.show()

if __name__ == "__main__":
  main()
