#!/usr/bin/env python3
from packaging import version

import click
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def set_edge_face_color(c):
  """Handles the <'Poly3DCollection' object has no attribute '_facecolors2d'>
  matplolib bug.

  The bug arises when passing the <label> argument to plot_surface(), in order
  to be displayed at the legend.

  Issue: https://github.com/matplotlib/matplotlib/issues/4067
  """
  if version.parse(matplotlib.__version__) >= version.parse("3.3.3"):
    return c._facecolor3d, c._edgecolor3d
  else:
    return c._facecolors3d, c._edgecolors3d


@click.command()
@click.argument("filename", type=click.STRING)
@click.option('-p', "--problem",
              default="k-shortest-paths", show_default=True,
              type=click.Choice(["replacement-paths-online",
                                 "replacement-paths-offline",
                                 "k-shortest-paths"],
                                case_sensitive=False))
@click.option("--max-probability", default=0.275, show_default=True,
              help=("Defined by the graph model parameters, except from p_0.\n"
                    " max_p * p_0 = expected-graph-density\n"
                    " (See shortestpaths.graph_generator)"))
def main(filename, problem, max_probability):
  results = np.loadtxt(filename)

  if problem == "k-shortest-paths":
    solvers = ["Yen", "Lawler", "Bidirectional Dijkstra", "Dynamic Programming"]
    colors = ['r', 'y', 'g', 'b']
  else:
    solvers = ["Unidirectional", "Bidirectional", "Dynamic Programming"]
    colors = ['r', 'g', 'b']
  # number of p values
  p = (len(results) - 1) // len(solvers)

  x = results[0, 1:].astype(int)
  y = results[1: p + 1, 0] * max_probability
  X, Y = np.meshgrid(x, y)
  Z = []
  for i in range(len(solvers)):
    Z.append(results[p * i + 1: (i + 1) * p + 1, 1:])

  fig = plt.figure()
  sub = fig.add_subplot(111, projection="3d")
  sub.set_xlabel("n (nodes")
  sub.set_ylabel("d")
  sub.set_zlabel("time (s)")
  sub.grid(False)
  fig.suptitle(f"{problem} profiling")

  for i in range(len(solvers)):
    c = sub.plot_surface(X, Y, Z[i],
                         rstride=1, cstride=1, linewidth=0,
                         color=colors[i], alpha=0.4,
                         shade=True, antialiased=False, label=solvers[i])
    c._facecolors2d, c._edgecolors2d = set_edge_face_color(c)
  plt.legend()
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  main()
