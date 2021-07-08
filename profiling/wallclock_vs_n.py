#!/usr/bin/env python3
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

from shortestpaths import core, graph


def measure(n,
            i,
            modes,
            problem,
            k,
            times,
            failing,
            online,
            **kwargs):
  """Replacement-paths search timer on the <i>th graph and for all the <modes>.

  It is important that upon function exit, the graphs are garbage collected.
  """
  adj_list, _ = graph.random_graph(n=n,
                                   random_seed=i,
                                   gradient=0.3,
                                   center_portion=0.15,
                                   p_0=0.3,
                                   **kwargs)

  init_config = {
    "adj_list": adj_list,
    "adj_list_reverse": graph.adj_list_reversed(adj_list),
    "source": 1,
    "sink": n
  }

  for solver, m in modes.items():
    mode = {
      "bidirectional": m[0],
      "parallel": m[1],
      "dynamic": m[2],
      "failing": failing,
      "online": online,
      "verbose": 0
    }
    start = timer()
    if problem == "k-shortest-paths":
      paths = core.k_shortest_paths(k, mode, init_config)
    else:
      paths = core.replacement_paths(mode, init_config)
    end = timer()
    times[solver].append(end - start)

  return times


@click.command()
@click.argument('n', type=click.INT)
@click.option('-s', "--step", default=25, show_default=True,
              help="number of nodes the graph size is increased by")
@click.option('-i', "--increases", default=50, show_default=True,
              help="number of graph sizes formed by <increase-step>")
@click.option('-g', "graphs_per_step", default=5, show_default=True,
              help=("number of different but equal sized graphs to run at each"
                    "step (the time measure will be the average)"))
@click.option('-f', "--failing", default="nodes", show_default=True)
@click.option("--online", is_flag=True)
@click.option('-p', "--problem",
              default="k-shortest-paths", show_default=True,
              type=click.Choice(["replacement-paths", "k-shortest-paths"],
                                case_sensitive=False))
@click.option('-k', type=click.INT, default=10, show_default=True,
              help="number of shortest paths to be generated")
@click.option("--weighted/--no-weighted", default=True, show_default=True)
@click.option("--directed", is_flag=True)
@click.option("--weights-on", default="edges-and-nodes", show_default=True,
              type=click.Choice(["edges", "nodes", "edges-and-nodes"],
                                case_sensitive=False))
@click.option("--max-edge-weight", default=1000, show_default=True)
@click.option("--max-node-weight", default=50, show_default=True)
def main(n,
         step,
         increases,
         graphs_per_step,
         failing,
         online,
         problem,
         k,
         **kwargs):

  if problem == "k-shortest-paths":
    online = True
    failing = "edges"

  modes = {
    # 'reference': [False, False, False],
    # 'parallel': [False, True, False],
    'bidirectional': [True, False, False],
    # 'bidirectional & parallel': [True, True, False],
    'dynamic': [True, False, True],
    # 'dynamic, bidirectional & parallel': [True, True, True]
  }

  times = {solver: [] for solver in modes.keys()}

  times_mean = {solver: [0] * increases for solver in modes.keys()}
  n_values = [n + i * step for i in range(increases)]

  for j in range(increases):
    for i in range(graphs_per_step):
      print((f"Graph: {j + 1}/{increases}"
             f"   Instance: {i + 1}/{graphs_per_step}"
             f"   nodes: {n}"),
            end='\r')

      times = measure(n,
                      i,
                      modes,
                      problem,
                      k,
                      times,
                      failing,
                      online,
                      **kwargs)
      gc.collect()

    for solver in modes.keys():
      times_mean[solver][j] = sum(times[solver]) / graphs_per_step
      del times[solver][:]

    n += step

  print()

  colors = iter(['b', 'm', 'g', 'k', 'c', 'y'])
  fig, ax = plt.subplots(figsize=(10.03, 6.2), dpi=200)
  for solver in modes.keys():
    color = next(colors)
    ax.plot(n_values, times_mean[solver], color=color, label=solver)
  ax.set_xlabel("n (nodes)")
  ax.set_ylabel("wall clock time (s)")
  ax.grid('major')
  ax.grid('minor', axis='y')
  fig.suptitle(f"{problem} profiling")
  fig.legend()
  plt.savefig(f"{problem}_profiling.png")
  plt.show()

if __name__ == "__main__":
  main()
