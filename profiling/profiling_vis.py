#!/usr/bin/env python3
import os
from packaging import version

import click
import numpy as np

import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import pandas as pd
from sklearn.linear_model import LinearRegression


def matshow_cmap():
  viridis_r = cm.get_cmap('viridis_r', 24)
  vir_colors = viridis_r.colors
  new_colors = np.vstack([[[1, 1, 1, 0]],
                         vir_colors])
  return ListedColormap(new_colors)


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


def problem_solvers_colors(filename, problem):
  supported_problems = ["replacement-paths-offline",
                        "replacement-paths-online",
                        "k-shortest-paths"]
  if problem is None:
    for probl in supported_problems:
      if os.path.basename(filename).startswith(probl):
        problem = probl
    if problem is None:
      problem = "k-shortest-paths"

  if problem == "k-shortest-paths":
    solvers = ["Yen + Dijkstra",
               "Lawler + Dijkstra",
               "Lawler + Bid. Dijkstra",
               "Lawler + Bid. Dijkstra + DP"]
    colors = ['aqua', '#c71585', 'g', 'mediumblue']
  else:
    solvers = ["Unidirectional Dijkstra",
               "Bidirectional Dijkstra",
               "Bid. Dijkstra + DP"]
    colors = ['aqua', 'mediumblue', 'orange']
  return problem, solvers, colors


def fontsizes(save_plot):
  if save_plot:
    title_fontsize, legend_fontsize, tick_fontsize = 17, 17, 17
  else:
    title_fontsize, legend_fontsize, tick_fontsize = 18, 16, 16
  return title_fontsize, legend_fontsize, tick_fontsize


def figsize_dpi(save_plot, axlist_ydim=None):
  if axlist_ydim:
    if save_plot:
      figsize, dpi = (12.5, 4.1 * axlist_ydim), 200
    else:
      figsize, dpi = (10, 4.1 * axlist_ydim), 100
  else:
    if save_plot:
      figsize, dpi = (10, 7.8), 200
    else:
      figsize, dpi = (9, 8), 100

  return {"figsize": figsize, "dpi": dpi}


def solvers_surfaces(X, Y, Z_pred, Z_real,
                     problem, solvers, colors, save_plot):
  """Plots a surface for each solver in the same axes."""
  title_fontsize, legend_fontsize, tick_fontsize = fontsizes(save_plot)
  fig = plt.figure(**figsize_dpi(save_plot))
  sub = fig.add_subplot(111, projection="3d")
  sub.set_xlabel("n (nodes)", fontsize=tick_fontsize, labelpad=15)
  sub.set_ylabel("d", fontsize=tick_fontsize, labelpad=15)
  sub.set_zlabel("t (s)", fontsize=tick_fontsize, labelpad=9)
  sub.xaxis.set_major_locator(MaxNLocator(6))
  sub.yaxis.set_major_locator(MaxNLocator(6))
  # sub.set_zlim(0, 225)
  sub.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  sub.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  sub.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

  # hacking zaxis
  # <stackoverflow.com/questions/48442713/move-spines-in-matplotlib-3d-plot/
  #  49601745#49601745>
  sub.zaxis._axinfo['juggled'] = (1, 2, 0)

  sub.view_init(25, 130)
  sub.tick_params(axis='both', which='major', labelsize=tick_fontsize)

  sub.grid(False)
  title = (f"{problem} profiling")
  fig.suptitle(title, fontsize=title_fontsize)

  for i in range(len(solvers)):
    c = sub.plot_surface(X, Y, Z_pred[i],
                         rstride=1, cstride=1, linewidth=0,
                         color=colors[i], alpha=0.4 + i / 5,
                         shade=True, antialiased=False, label=solvers[i])
    c._facecolors2d, c._edgecolors2d = set_edge_face_color(c)
    sub.scatter(X, Y, Z_real[i], s=8, c=colors[i], alpha=0.6 + i / 5)

  sub.legend(fontsize=legend_fontsize)
  plt.tight_layout()
  if save_plot:
    plt.savefig(f"{problem}_profiling.png", dpi=fig.dpi)
  return fig


def solvers_matshows(x, y, Z_norm, Z_real,
                     problem, solvers, save_plot):
  """Plots a matshow for each solver in a different axes."""
  title_fontsize, legend_fontsize, tick_fontsize = fontsizes(save_plot)
  fig, axlist = plt.subplots(len(solvers), 1,
                             **figsize_dpi(save_plot, len(solvers)),
                             constrained_layout=True)

  for i, ax in enumerate(axlist):
    cmap = plt.get_cmap("viridis_r", 256)
    # cmap.set_under('whitesmoke')
    # cmap.set_over('k')
    # plt.set_cmap("plasma_r")
    ax.matshow(Z_norm[i], interpolation='nearest', label=solvers[i],
               cmap=cmap, vmin=0, vmax=40)
    ax.invert_yaxis()

    numrows, numcols = Z_norm[i].shape

    def format_coord(xx, yy):
      col = int(xx + 0.5)
      row = int(yy + 0.5)
      if (col >= 0) and (col < numcols) and (row >= 0) and (row < numrows):
        zz = Z_norm[i][row, col]
        return f"x={xx:.4f}, y={yy:.4f}, z={zz:.4f}"
      else:
        return f"x={xx:.4f}, y={yy:.4f}"

    ax.format_coord = format_coord

    # axis labels
    if i == len(solvers) - 1:
      ax.set_xlabel("n (nodes)", fontsize=legend_fontsize, labelpad=9)
    ax.set_ylabel("d", fontsize=legend_fontsize,
                  labelpad=9, rotation="horizontal")

    # ticks
    ax.xaxis.set_major_locator(MaxNLocator(x.size + 1))
    ax.yaxis.set_major_locator(MaxNLocator(y.size + 1))
    # ax.xaxis.tick_top()
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # tick labels
    # ax.xaxis.set_label_position('top')
    if i == len(solvers) - 1:
      ax.tick_params(axis="x",
                     which='major', labelsize=tick_fontsize,
                     bottom=True, top=False,
                     labelbottom=True, labeltop=False)
      ax.set_xticklabels(np.hstack([[0], x]))
    else:
      ax.tick_params(axis="x",
                     which='major', labelsize=tick_fontsize,
                     bottom=False, top=False,
                     labelbottom=False, labeltop=False)
      ax.set_xticklabels([])
    ylabels = [f"{la:.3f}" for la in np.round(np.hstack([[0], y]), decimals=3)]
    ax.set_yticklabels(ylabels)
    # title
    ax.set_title(solvers[i], fontsize=title_fontsize)

    # title = (f"{problem} profiling")
    # fig.suptitle(title, fontsize=title_fontsize)

    for x_idx in range(x.size):
      for y_idx in range(y.size):
        t_norm = Z_norm[i][y_idx, x_idx]
        t_real = Z_real[i][y_idx, x_idx]

        if t_real >= 100:
          t_real = np.round(t_real).astype(int)
        elif t_real >= 10:
          t_real = np.round(t_real, decimals=1)
        else:
          t_real = np.round(t_real, decimals=2)

        if t_norm > 30:
          c = 'w'
        else:
          c = 'k'
        # if t < 0:
        #   t = np.round(t, decimas=2).astype(str)
        ax.text(x_idx, y_idx, t_real, color=c, va='center', ha='center',
                fontsize=tick_fontsize)

  if save_plot:
    plt.savefig(f"{problem}_profiling_matshows.png", dpi=fig.dpi)
  return fig


def gains_matshows(x, y, Z_real,
                   problem, solvers, save_plot):
  """Plots the performance gains."""
  title_fontsize, legend_fontsize, tick_fontsize = fontsizes(save_plot)
  fig, axlist = plt.subplots(
    int(problem == "k-shortest-paths") + 1, 1,
    **figsize_dpi(save_plot, max(2, len(solvers) // 2)),
    constrained_layout=True
  )
  cmap = plt.get_cmap("viridis", 4)
  numrows, numcols = Z_real[0].shape

  if not hasattr(axlist, "__iter__"):
    axlist = [axlist]
    offset = 1
  else:
    offset = 0

  for i, ax in enumerate(axlist):
    ax_tilte = f"{solvers[2 * i + offset]} - {solvers[2 * i + 1 + offset]}"
    Z_real_gains = ((Z_real[2 * i + offset] - Z_real[2 * i + 1 + offset])
                    / (Z_real[2 * i] + 1))

    ax.matshow(Z_real_gains,
               interpolation='nearest',
               label=ax_tilte,
               cmap=cmap, vmin=0, vmax=0.20)

    def format_coord(xx, yy):
      col = int(xx + 0.5)
      row = int(yy + 0.5)
      if (col >= 0) and (col < numcols) and (row >= 0) and (row < numrows):
        zz = Z_real_gains[row, col]
        return f"x={xx:.4f}, y={yy:.4f}, z={zz:.4f}"
      else:
        return f"x={xx:.4f}, y={yy:.4f}"

    ax.format_coord = format_coord
    ax.invert_yaxis()

    # axis labels
    if i == len(solvers) // 2 - 1:
      ax.set_xlabel("n (nodes)", fontsize=legend_fontsize, labelpad=9)
    ax.set_ylabel("d", fontsize=legend_fontsize,
                  labelpad=12, rotation="horizontal")

    # ticks
    ax.xaxis.set_major_locator(MaxNLocator(numcols + 1))
    ax.yaxis.set_major_locator(MaxNLocator(numrows + 1))
    # ax.xaxis.tick_top()
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # tick labels
    # ax.xaxis.set_label_position('top')
    if i == len(solvers) // 2 - 1:
      ax.tick_params(axis="x",
                     which='major', labelsize=tick_fontsize,
                     bottom=True, top=False,
                     labelbottom=True, labeltop=False)
      ax.set_xticklabels(np.hstack([[0], x]))
    else:
      ax.tick_params(axis="x",
                     which='major', labelsize=tick_fontsize,
                     bottom=False, top=False,
                     labelbottom=False, labeltop=False)
      ax.set_xticklabels([])
    ylabels = [f"{la:.3f}" for la in np.round(np.hstack([[0], y]), decimals=3)]
    ax.set_yticklabels(ylabels)
    # title
    ax.set_title(ax_tilte, fontsize=title_fontsize)

    # title = (f"{problem} profiling")
    # fig.suptitle(title, fontsize=title_fontsize)

    for x_idx in range(x.size):
      for y_idx in range(y.size):
        t_real = (Z_real[2 * i + offset][y_idx, x_idx]
                  - Z_real[2 * i + 1 + offset][y_idx, x_idx])
        t_perc = Z_real_gains[y_idx, x_idx] * 100

        if t_real >= 100:
          t_real = np.round(t_real).astype(int)
        elif t_real >= 10:
          t_real = np.round(t_real, decimals=1)
        else:
          t_real = np.round(t_real, decimals=2)

        if t_perc >= 10:
          t_perc = np.round(t_perc, decimals=0).astype(int)
        else:
          t_perc = np.round(t_perc, decimals=1)

        if t_perc > 5:
          c = 'k'
        else:
          c = 'silver'
        # if t < 0:
        #   t = np.round(t, decimas=2).astype(str)
        ax.text(x_idx, y_idx, f"{t_real}\n{t_perc}%",
                color=c, va='center', ha='center', fontsize=tick_fontsize)

  if save_plot:
    plt.savefig(f"{problem}_profiling_gains_matshows.png", dpi=fig.dpi)
  return fig


def fit_pol(xx, yy, Z, order=4):
  # Quadratic
  x = xx.reshape(1, -1)
  y = yy.reshape(-1, 1)
  features = {}
  # 0th
  features["ones"] = np.ones_like(Z).flatten()
  # 1st
  features["x"] = np.matmul(np.ones_like(y), x).flatten()
  features["y"] = np.matmul(y, np.ones_like(x)).flatten()
  if order >= 2:
    features["x*y"] = np.matmul(y, x).flatten()
    features["x^2"] = np.matmul(np.ones_like(y), x**2).flatten()
    features["y^2"] = np.matmul(y**2, np.ones_like(x)).flatten()
  if order >= 3:
    features["x^3"] = np.matmul(np.ones_like(y), x**3).flatten()
    features["y^3"] = np.matmul(y**3, np.ones_like(x)).flatten()
    features["x*y^2"] = np.matmul(y**2, x).flatten()
    features["x^2*y"] = np.matmul(y, x**2).flatten()
  if order >= 4:
    features["x^4"] = np.matmul(np.ones_like(y), x**4).flatten()
    features["y^4"] = np.matmul(y**4, np.ones_like(x)).flatten()
    features["x^3*y"] = np.matmul(y, x**3).flatten()
    features["x*y^3"] = np.matmul(y**3, x).flatten()
    features["x^2*y^2"] = np.matmul(y**2, x**2).flatten()
  if order >= 5:
    features["x^5"] = np.matmul(np.ones_like(y), x**5).flatten()
    features["y^5"] = np.matmul(y**5, np.ones_like(x)).flatten()
    features["x^4*y"] = np.matmul(y, x**4).flatten()
    features["x*y^4"] = np.matmul(y**4, x).flatten()
    features["x^3*y^2"] = np.matmul(y**2, x**3).flatten()
    features["x^2*y^3"] = np.matmul(y**3, x**2).flatten()
  if order >= 6:
    features["x^6"] = np.matmul(np.ones_like(y), x**6).flatten()
    features["y^6"] = np.matmul(y**6, np.ones_like(x)).flatten()
    features["x^5*y"] = np.matmul(y, x**5).flatten()
    features["x*y^5"] = np.matmul(y**5, x).flatten()
    features["x^4*y^2"] = np.matmul(y**2, x**4).flatten()
    features["x^2*y^4"] = np.matmul(y**4, x**2).flatten()
    features["x^3*y^3"] = np.matmul(y**3, x**3).flatten()
    features["x^3*y^3"] = np.matmul(y**3, x**3).flatten()
  if order >= 7:
    features["x^7"] = np.matmul(np.ones_like(y), x**7).flatten()
    features["y^7"] = np.matmul(y**7, np.ones_like(x)).flatten()
    features["x^6*y"] = np.matmul(y, x**6).flatten()
    features["x*y^6"] = np.matmul(y**6, x).flatten()
    features["x^5*y^2"] = np.matmul(y**2, x**5).flatten()
    features["x^2*y^5"] = np.matmul(y**5, x**2).flatten()
    features["x^3*y^4"] = np.matmul(y**4, x**3).flatten()
    features["x^4*y^3"] = np.matmul(y**3, x**4).flatten()
  ds = pd.DataFrame(features)

  # __import__('ipdb').set_trace(context=9)

  reg = LinearRegression().fit(ds.values, Z.flatten())
  z_pred = (
    reg.intercept_
    + np.matmul(ds.values, reg.coef_.reshape(-1, 1)).reshape(Z.shape)
  )
  return z_pred


def z_real_pred_norm(x, y, results, solvers, num_p, order=4):
  Z_real = []
  Z_pred = []
  Z_norm = []

  for i in range(len(solvers)):
    z_real = results[num_p * i + 1: (i + 1) * num_p + 1, 1:]
    Z_real.append(z_real)
    z_pred = fit_pol(x, y, z_real, order=order)
    Z_pred.append(z_pred)

  # Normalize to 0-255
  #   - the min value should be in the timings of the last solver
  #   - the max value should be in the timings of the first solver
  for i in range(len(solvers)):
    Z_norm.append(
      (
        (Z_real[i] - Z_real[-1].min())
        / Z_real[0].max()
        * 255
      ).astype(int)
    )
  return Z_real, Z_pred, Z_norm


@click.command()
@click.argument("filename", type=click.STRING)
@click.option('-p', "--problem",
              default=None, show_default=True,
              type=click.Choice(["replacement-paths-online",
                                 "replacement-paths-offline",
                                 "k-shortest-paths",
                                 None],
                                case_sensitive=False))
@click.option("--max-probability", default=0.28, show_default=True,
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
  # results = np.loadtxt(filename)[:, :5]
  results = np.loadtxt(filename)

  problem, solvers, colors = problem_solvers_colors(filename, problem)
  # number of p values
  num_p = (len(results) - 1) // len(solvers)

  x = results[0, 1:].astype(int)
  # y = results[1: num_p + 1, 0]
  y = results[1: num_p + 1, 0] * max_probability
  # y = np.array([0.1, 0.28, 0.44, 0.58, 0.7]) * 0.3
  X, Y = np.meshgrid(x, y)
  Z_real, Z_pred, Z_norm = z_real_pred_norm(x, y, results,
                                            solvers, num_p, order=3)

  matshows = solvers_matshows(x, y, Z_norm, Z_real,
                              problem, solvers, save_plot)
  comparisons = diff_matshows(x, y, Z_norm, Z_real,
                              problem, solvers, save_plot)
  surfaces = solvers_surfaces(X, Y, Z_pred, Z_real,
                              problem, solvers, colors, save_plot)

  if show_plot:
    plt.show()


if __name__ == '__main__':
  main()
