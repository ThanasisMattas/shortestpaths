#!/usr/bin/env python3
import os
from pathlib import Path
import sys
from timeit import default_timer as timer

cwd = Path(os.getcwd())
home_dir = str(cwd.parent)
sys.path.insert(0, home_dir)

import click
import numpy as np

from shortestpaths import core, graph_generator, io


def time_a_run(mode, init_config, problem, k=None, times=3):
  timings = np.zeros(times, dtype=np.float32)
  for t in range(times):
    start = timer()
    if problem == "k-shortest-paths":
      core.k_shortest_paths(k, mode, init_config)
    else:
      core.replacement_paths(mode, init_config)
    end = timer()
    timings[t] = (end - start)
  return np.mean(timings)


def measure(n,
            g,
            p,
            problems,
            k=None,
            failing=None,
            measurements_per_graph=3,
            dataset=None,
            **kwargs):
  """A measurement corresponds to a single unique graph and incorporates
  all specified problems on all specified search types (solvers).

  It is important that upon function exit, the graphs are garbage collected.

  Returns:
    timings (array) : Each row corresponds to a problem and comprises a list
                      with the timings for each solver.
  """
  if dataset is None:
    adj_list, _ = graph_generator.random_graph(n=n,
                                               random_seed=g,
                                               p_0=p / 10,
                                               **kwargs)
  else:
    adj_list = next(dataset)

  adj_list_reverse = graph_generator.adj_list_reversed(adj_list)
  init_config = {
    "adj_list": adj_list,
    "adj_list_reverse": None,
    "source": 1,
    "sink": n
  }

  timings = []

  for i, probl in enumerate(problems):
    if probl == "k-shortest-paths":
      solvers = {
        'yen_': [False, False],
        'lawler': [False, False],
        'bidirectional': [True, False],
        'dynamic': [True, True]
      }
      mode = {"failing": "edges", "online": True}
    else:
      solvers = {
        'unidirectional': [False, False],
        'bidirectional': [True, False],
        'dynamic': [True, True]
      }
      if probl.endswith("online"):
        mode = {"failing": failing, "online": True}
      else:
        mode = {"failing": failing, "online": False}
    timings.append([0] * len(solvers))

    for j, (s, m) in enumerate(solvers.items()):
      mode.update({"bidirectional": m[0], "dynamic": m[1]})
      if s == "yen_":
        mode.update({"yen_": True, "lawler": False})
      elif s == "lawler":
        mode.update({"yen_": False, "lawler": True})
      elif m[0]:
        init_config["adj_list_reverse"] = adj_list_reverse
        mode.update({"yen_": False, "lawler": False})
      timings[i][j] = time_a_run(mode,
                                 init_config,
                                 probl,
                                 k,
                                 times=measurements_per_graph)
  return timings


@click.command()
@click.option("--dataset-file", type=click.STRING,
              default=None, show_default=True)
@click.option('-n', "--n-init", default=400, show_default=True)
@click.option('-s', "--step", default=400, show_default=True,
              help="order increase step")
@click.option('-i', "--increases", default=20, show_default=True)
@click.option('-g', "--graphs-per-step", default=10, show_default=True)
@click.option('-m', "--measurements-per-graph", default=3, show_default=True)
@click.option('-f', "--failing", default="edges", show_default=True,
              type=click.Choice(["edges", "nodes"], case_sensitive=False))
@click.option('-p', "--problem",
              default="all", show_default=True,
              type=click.Choice(["replacement-paths-online",
                                 "replacement-paths-offline",
                                 "k-shortest-paths",
                                 "all"],
                                case_sensitive=False))
@click.option('-k', type=click.INT, default=10, show_default=True)
@click.option('-c', "center_portion", default=0.15, show_default=True)
@click.option("--gradient", default=0.3, show_default=True)
@click.option("--directed", is_flag=True)
@click.option("--weights-on", default="edges-and-nodes", show_default=True,
              type=click.Choice(["edges", "nodes", "edges-and-nodes"],
                                case_sensitive=False))
@click.option("--max-edge-weight", default=1000, show_default=True)
@click.option("--max-node-weight", default=50, show_default=True)
def main(dataset_file,
         n_init,
         step,
         increases,
         graphs_per_step,
         measurements_per_graph,
         failing,
         problem,
         k,
         **kwargs):

  if dataset_file:
    dataset = io.load_graphs(dataset_file)

  n_range = range(n_init, n_init + step * increases, step)
  p_range = range(1, 11)

  if problem == "all":
    problems = ["replacement-paths-online",
                "replacement-paths-offline",
                "k-shortest-paths"]
  else:
    problems = [problem]

  results = {}
  for probl in problems:
    if probl == "k-shortest-paths":
      s = 4
    else:
      s = 3
    results[probl] = np.zeros((s, len(p_range), increases, graphs_per_step),
                              dtype=np.float32)

  for i, n in enumerate(n_range):
    for j, p in enumerate(p_range):
      for g in range(1, graphs_per_step + 1):
        print(
          (f"Graph: {i * len(p_range) + j:>3}/{len(p_range) * increases}"
           f"   n: {n:>4}   p\u2080: {p / 10:.1f}"
           f"   Instance: {g:>2}/{graphs_per_step}"),
          end='\r'
        )
        # Each measurement corresponds to a single unique graph.
        measurement = measure(n,
                              g,
                              p,
                              problems,
                              k,
                              failing,
                              measurements_per_graph,
                              dataset=dataset,
                              **kwargs)
        for r, result in enumerate(results.values()):
          result[:, j, i, g - 1] = measurement[r]
  print()

  for probl in problems:
    if probl == "k-shortest-paths":
      solvers = ['yen_', 'lawler', 'bidirectional', 'dynamic']
      out_file = (f"{probl}_time_profiling_k_{k}_"
                  f"{n_init}_{n_init + (increases - 1) * step}_{step}.csv")
    else:
      solvers = ['unidirectional', 'bidirectional', 'dynamic']
      out_file = (f"{probl}_time_profiling_"
                  f"{n_init}_{n_init + (increases - 1) * step}_{step}.csv")
    # Take the mean of <graphs_per_measure> and concatenate the solvers because
    # savetxt cannot save a 3D array.
    results[probl] = np.mean(results[probl], axis=-1).reshape(
      len(solvers) * len(p_range), increases
    )
    res = np.zeros((results[probl].shape[0] + 1, results[probl].shape[1] + 1))
    res[1:, 1:] = results[probl]
    res[1:, 0] = len(solvers) * list(p_range)
    res[0, 1:] = n_range

    directed = "directed" if kwargs["directed"] else "undirected"
    np.savetxt(
      out_file,
      res,
      header=(
        f"{(probl + ' profiling').center(50, '-')}\n"
        f"{'-' * 50}\n"
        f"failing : {failing}\n"
        f"Graph   : {directed}   c = {kwargs['center_portion']}"
        f"   λ = {kwargs['gradient']}\n"
        f"Solvers : {', '.join(solvers)}\n\n"
        # f"y {' '.join([str(p_value) for p_value in p_range])}\n"
        # f"x {' '.join([str(n_value) for n_value in n_range])}"
      )
    )

if __name__ == "__main__":
  main()
