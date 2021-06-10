#!/usr/bin/env python3
from packaging import version

import click
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


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


def fit_pol(x, y, Z, order=4):
  # Quadratic
  x = x.reshape(-1, 1)
  y = y.reshape(1, -1)
  features = {}
  # 0th
  features["ones"] = np.ones(x.size * y.size)
  # 1st
  features["x"] = np.matmul(x, np.ones_like(y)).flatten()
  features["y"] = np.matmul(np.ones_like(x), y).flatten()
  if order >= 2:
    features["x*y"] = np.matmul(x, y).flatten()
    features["x^2"] = np.matmul(x**2, np.ones_like(y)).flatten()
    features["y^2"] = np.matmul(np.ones_like(x), y**2).flatten()
  if order >= 3:
    features["x^3"] = np.matmul(x**3, np.ones_like(y)).flatten()
    features["y^3"] = np.matmul(np.ones_like(x), y**3).flatten()
    features["x*y^2"] = np.matmul(x, y**2).flatten()
    features["x^2*y"] = np.matmul(x**2, y).flatten()
  if order >= 4:
    features["x^4"] = np.matmul(x**4, np.ones_like(y)).flatten()
    features["y^4"] = np.matmul(np.ones_like(x), y**4).flatten()
    features["x^3*y"] = np.matmul(x**3, y).flatten()
    features["x*y^3"] = np.matmul(x, y**3).flatten()
    features["x^2*y^2"] = np.matmul(x**2, y**2).flatten()
  if order >= 5:
    features["x^4*y"] = np.matmul(x**4, y).flatten()
    features["x*y^4"] = np.matmul(x, y**4).flatten()
    features["x^3*y^2"] = np.matmul(x**3, y**2).flatten()
    features["x^2*y^3"] = np.matmul(x**2, y**3).flatten()
  ds = pd.DataFrame(features)

  reg = LinearRegression().fit(ds.values, Z.flatten())
  z_pred = (
    reg.intercept_
    + np.matmul(ds.values, reg.coef_.reshape(-1, 1)).reshape(x.size, y.size)
  )
  return z_pred


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
@click.option("--show-plot/--no-show-plot", default=True, show_default=True)
@click.option("--save-plot/--no-save-plot", default=False, show_default=True)
def main(filename,
         problem,
         max_probability,
         show_plot,
         save_plot):
  results = np.loadtxt(filename)

  if save_plot:
    figsize = (12, 10)
    dpi = 200
    title_fontsize = 20
    legend_fontsize = 18
    tick_fontsize = 16
  else:
    figsize = (10, 8)
    dpi = 100
    title_fontsize = 18
    legend_fontsize = 18
    tick_fontsize = 16

  if problem == "k-shortest-paths":
    solvers = ["Yen + Dijkstra",
               "Lawler + Dijkstra",
               "Lawler + Bid. Dijkstra",
               "Lawler + Bid. Dijkstra + DP"]
    colors = ['r', 'y', 'g', 'b']
  else:
    solvers = ["Unidirectional Dijkstra",
               "Bidirectional Dijkstra",
               "Bid. Dijkstra+ DP"]
    colors = ['r', 'g', 'b']
  # number of p values
  p = (len(results) - 1) // len(solvers)

  x = results[0, 1:].astype(int)
  y = results[1: p + 1, 0] * max_probability
  X, Y = np.meshgrid(x, y)
  Z_fitted = []
  Z_ground_truth = []

  for i in range(len(solvers)):
    z = results[p * i + 1: (i + 1) * p + 1, 1:]
    z_pred = fit_pol(x, y, z, order=4)
    Z_ground_truth.append(z)
    Z_fitted.append(z_pred)

  fig = plt.figure(figsize=figsize, dpi=dpi)
  sub = fig.add_subplot(111, projection="3d")
  sub.set_xlabel("n (nodes)", fontsize=legend_fontsize, labelpad=15)
  sub.set_ylabel("d", fontsize=legend_fontsize, labelpad=15)
  sub.set_zlabel("time (s)", fontsize=legend_fontsize, labelpad=9)
  sub.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  sub.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  sub.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

  # sub.grid(color='gray', linewidth=0.5)
  # sub.set_yticks(y - 0.0075)
  # sub.set_ylim([y[0], y[-1]])
  # sub.view_init(0, 90)
  sub.tick_params(axis='both', which='major', labelsize=tick_fontsize)

  sub.grid(False)
  title = (f"{problem} profiling")
  fig.suptitle(title, fontsize=title_fontsize)

  for i in range(len(solvers)):
    c = sub.plot_surface(X, Y, Z_fitted[i],
                         rstride=1, cstride=1, linewidth=0,
                         color=colors[i], alpha=0.3 + i / 10,
                         shade=True, antialiased=False, label=solvers[i])
    c._facecolors2d, c._edgecolors2d = set_edge_face_color(c)
    sub.scatter(X, Y, Z_ground_truth[i], s=7, c=colors[i], alpha=0.3 + i / 10)
  plt.legend(fontsize=legend_fontsize)
  plt.tight_layout()
  if save_plot:
    plt.savefig(f"{problem}_profiling.png", dpi=fig.dpi)
  if show_plot:
    plt.show()


if __name__ == '__main__':
  main()
