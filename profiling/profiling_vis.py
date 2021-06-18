#!/usr/bin/env python3
import os
from packaging import version

import click
import numpy as np

import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

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


def problem_solvers_colors(filename, problem, param):
  supported_problems = ["replacement-paths-offline",
                        "replacement-paths-online",
                        "k-shortest-paths"]
  if problem is None:
    for probl in supported_problems:
      if os.path.basename(filename).startswith(probl):
        problem = probl
        break
    if problem is None:
      raise ValueError(f"filename <{filename}> should start with on of"
                       f" {supported_problems}.")

  if problem == "k-shortest-paths":
    solvers = ["Yen + Dijkstra",
               "Lawler + Dijkstra",
               "Lawler + Bid. Dijkstra",
               "Lawler + Bid. Dijkstra + DP"]
    colors = ['aqua', '#c71585', 'g', 'mediumblue']
  else:
    if param == 'c':
      solvers = ["Bidirectional Dijkstra",
                 "Bid. Dijkstra + DP"]
      colors = ['aqua', 'coral']
    else:
      solvers = ["Unidirectional Dijkstra",
                 "Bidirectional Dijkstra",
                 "Bid. Dijkstra + DP"]
      colors = ['aqua', 'orange', 'mediumblue']
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
      figsize, dpi = (12.8, 4.1 * axlist_ydim), 200
    else:
      figsize, dpi = (10, 4.1 * axlist_ydim), 100
  else:
    if save_plot:
      figsize, dpi = (10, 7.8), 200
    else:
      figsize, dpi = (9, 8), 100

  return {"figsize": figsize, "dpi": dpi}


def yticklabels_c_study(y):
  """Hard coded y ticks labels for sigmoid center study.

  - It creates 3 lines of y tick labels, corresponding to center, p_max and p.
  - p = p_max * p_0, with p_0 = 0.3
  """
  p_max = np.array([0.1, 0.28, 0.44, 0.58, 0.7])

  labels = [f"{y[i] / 10:.2f}\n{p_max[i]:.2f}\n{p_max[i] * 0.3:.2f}"
            for i in range(y.size)]
  return labels


def ylabel(param, mattshow=False):
  if param == 'c':
    if mattshow:
      return param
    else:
      return ("c\n"
              "$\mathregular{p_m}$"   # noqa: W605
              "$\mathregular{_a}$"    # noqa: W605
              "$\mathregular{_x}$\n"  # noqa: W605
              "d")
  elif param == 'p_0':
    return 'd'
  else:
    return param


def plot_surface_tilte(problem, param):
  """Hard coded fig title for the surface plot.

  Args:
    problem (str) : see problem_solvers_colors()
    study (str)   : the parateter of the study, other than graph order
                      options: ['c', 'p_o', 'k']
  """
  if param == 'c':
    title = (
      f"{problem} profiling\n" + "center portion: 0.15   "
      "$\mathregular{p_0}$: 0.3    "  # noqa: W605
    )
  elif param == 'p_0':
    title = (
      f"{problem} profiling\n" + "center portion: 0.15   "
      "$\mathregular{p_m}$" + "$\mathregular{_a}$" + "$\mathregular{_x}$"  # noqa: W605
      ": 0.28"
    )
  elif param == 'k':
    title = (
      f"{problem} profiling\n" + "center portion: 0.15   "
      "$\mathregular{p_0}$: 0.3   "  # noqa: W605
      "$\mathregular{p_m}$" + "$\mathregular{_a}$" + "$\mathregular{_x}$"  # noqa: W605
      ": 0.28"
    )
  else:
    accepted_param_values = ['c', 'p_0', 'k']
    raise ValueError(f"<study> positional argument should be on of"
                     f" {accepted_param_values}. Instead, got {param}.")
  return title


def fit_linear(x, y):
  x_vect = x.reshape(-1, 1)
  ones = np.ones((x_vect.size, 1))
  features = np.hstack([ones, x_vect])
  reg = LinearRegression().fit(features, y.flatten())
  y_pred = (
    reg.intercept_
    + np.matmul(features, reg.coef_.reshape(-1, 1)).flatten()
  )
  return y_pred


def fit_pol(xx, yy, Z, order=4):
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


def solvers_surfaces(X, Y, Z_pred, Z_real, y, colors, param,
                     problem, solvers, save_plot):
  """Plots a surface for each solver in the same axes."""
  title_fontsize, legend_fontsize, tick_fontsize = fontsizes(save_plot)
  fig = plt.figure(**figsize_dpi(save_plot))
  sub = fig.add_subplot(111, projection="3d")
  # axes labels
  sub.set_xlabel("n (nodes)", fontsize=tick_fontsize, labelpad=27)
  if param == 'c':
    ylabelpad = 50
  else:
    ylabelpad = 15
  sub.set_ylabel(ylabel(param), fontsize=tick_fontsize, labelpad=ylabelpad)
  sub.set_zlabel("t (s)", fontsize=tick_fontsize, labelpad=13)
  # ticks
  sub.xaxis.set_major_locator(MaxNLocator(y.size + 1))
  sub.yaxis.set_major_locator(MaxNLocator(y.size + 1))
  if param == 'k':
    sub.set_xticks(np.arange(500, 3000, 500))
  # tick labels
  sub.tick_params(axis='both', which='major', labelsize=tick_fontsize)
  sub.set_yticks(y)
  if param == 'p_0':
    sub.yaxis.set_major_formatter("{x:.3f}")
  elif param == 'c':
    sub.set_yticklabels(yticklabels_c_study(y))
  sub.zaxis.set_rotate_label(False)
  sub.yaxis.set_rotate_label(False)
  sub.xaxis.set_rotate_label(False)
  # sub.set_xticklabels()
  # remove panes
  sub.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  sub.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  sub.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  # hacking zaxis
  # <stackoverflow.com/questions/48442713/move-spines-in-matplotlib-3d-plot/
  #  49601745#49601745>
  sub.zaxis._axinfo['juggled'] = (1, 2, 0)

  if param == 'k':
    sub.view_init(15, -70)
  elif param == 'c':
    sub.view_init(16, 210)
    sub.set_zlim(0, 8.2)
  else:
    sub.view_init(16, 150)
    # sub.set_zlim(0, 110)
  sub.grid(False)
  # sub.computed_zorder = True

  title = (plot_surface_tilte(problem, param=param))
  fig.suptitle(title, fontsize=title_fontsize)
  for i in range(len(solvers)):
    c = sub.plot_surface(X, Y, Z_pred[i],
                         rstride=1, cstride=1, linewidth=0,
                         color=colors[i], alpha=0.4 + i / 5,
                         shade=True, antialiased=False, label=solvers[i])
    c._facecolors2d, c._edgecolors2d = set_edge_face_color(c)
    sub.scatter(X, Y, Z_real[i], s=8, c=colors[i], alpha=0.4 + i / 5)

  sub.legend(fontsize=legend_fontsize)
  plt.tight_layout()
  if save_plot:
    plt.savefig(f"{problem}_profiling.png", dpi=fig.dpi)
  return fig


def gen_matshow(ax, i, label, solvers_divisor,
                x, y, Z,
                cmap, vmax, solvers,
                xpad, ypad,
                save_plot, param):
  title_fontsize, legend_fontsize, tick_fontsize = fontsizes(save_plot)
  numcols, numrows = Z.shape

  ax.matshow(Z,
             interpolation='nearest',
             label=label,
             cmap=cmap, vmin=0, vmax=vmax)

  def format_coord(xx, yy):
    col = int(xx + 0.5)
    row = int(yy + 0.5)
    if (col >= 0) and (col < numcols) and (row >= 0) and (row < numrows):
      zz = Z[row, col]
      return f"x={xx:.4f}, y={yy:.4f}, z={zz:.4f}"
    else:
      return f"x={xx:.4f}, y={yy:.4f}"

  ax.format_coord = format_coord
  ax.invert_yaxis()

  # axis labels
  if i == len(solvers) // solvers_divisor - 1:
    ax.set_xlabel("n (nodes)", fontsize=legend_fontsize, labelpad=xpad)
  ax.set_ylabel(ylabel(param, mattshow=True), fontsize=legend_fontsize,
                labelpad=ypad, rotation="horizontal")

  # ticks
  ax.xaxis.set_major_locator(MaxNLocator(x.size + 1))
  ax.yaxis.set_major_locator(MaxNLocator(y.size + 1))
  ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

  # tick labels
  if i == len(solvers) // solvers_divisor - 1:
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

  if param == 'p_0':
    ylabels = [f"{y_val:.3f}"
               for y_val in np.round(np.hstack([[0], y]), decimals=3)]
    ax.set_yticklabels(ylabels)
  elif param == 'c':
    ax.set_yticklabels(yticklabels_c_study(y))
  elif param == 'k':
    ax.set_yticklabels(np.hstack([[0], y]).astype(int))
  # title
  if param == 'k':
    ax.annotate(solvers[i],
                xy=(1, 2.5), xytext=(0, 0),
                xycoords=ax.get_yaxis_transform(),
                textcoords="offset points",
                size=title_fontsize, va="center",
                rotation=-90)
  else:
    ax.set_title(solvers[i], fontsize=title_fontsize)


def solvers_matshows(x, y, Z_norm, Z_real, param,
                     problem, solvers, save_plot):
  """Plots a matshow for each solver in a different axes."""
  fig, axlist = plt.subplots(len(solvers), 1,
                             **figsize_dpi(save_plot, len(solvers)),
                             constrained_layout=True)

  cmap = plt.get_cmap("viridis_r", 256)
  title_fontsize, legend_fontsize, tick_fontsize = fontsizes(save_plot)

  for i, ax in enumerate(axlist):
    gen_matshow(ax=ax, i=i, label=solvers[i], solvers_divisor=1,
                x=x, y=y, Z=Z_norm[i],
                cmap=cmap, vmax=50, solvers=solvers,
                xpad=9, ypad=9,
                save_plot=save_plot, param=param)

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

        if t_norm > 40:
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


def gains_matshows(x, y, Z_real, param,
                   problem, solvers, save_plot):
  """Plots the performance gains."""
  fig, axlist = plt.subplots(
    int(problem == "k-shortest-paths") + 1, 1,
    **figsize_dpi(save_plot, max(2, len(solvers) // 2)),
    constrained_layout=True
  )

  if not hasattr(axlist, "__iter__"):
    axlist = [axlist]
    offset = 1
  else:
    offset = 0

  cmap = plt.get_cmap("viridis", 5)
  title_fontsize, legend_fontsize, tick_fontsize = fontsizes(save_plot)

  for i, ax in enumerate(axlist):
    ax_tilte = f"({solvers[2 * i + offset]}) - ({solvers[2 * i + 1 + offset]})"
    Z_real_gains = ((Z_real[2 * i + offset] - Z_real[2 * i + 1 + offset])
                    / (Z_real[2 * i] + 1))

    gen_matshow(ax=ax, i=i, label=ax_tilte, solvers_divisor=2,
                x=x, y=y, Z=Z_real_gains,
                cmap=cmap, vmax=0.2, solvers=solvers,
                xpad=6, ypad=12,
                save_plot=save_plot, param=param)

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


def mse_lines(x,
                          z,
                          n,
                          param,
                          problem,
                          solvers,
                          save_plot):
  # title_fontsize, legend_fontsize, tick_fontsize = fontsizes(save_plot)
  title_fontsize, legend_fontsize, tick_fontsize = 22, 22, 22
  markers = ['o', '^', 's', 'D']

  fig, ax = plt.subplots(**figsize_dpi(save_plot), constrained_layout=True)

  xlabel = 'd' if param == "p_0" else param
  ax.set_xlabel(xlabel, fontsize=tick_fontsize)
  ax.set_ylabel('t (s)', fontsize=tick_fontsize, rotation=0, labelpad=20)
  ax.xaxis.set_major_locator(MaxNLocator(x.size + 1))
  ax.yaxis.set_minor_locator(MultipleLocator(2.5))
  if param == 'k':
    colors = ['r', 'y', 'g', 'b']
    ax.set_ylim(0, 27)
    ax.set_xlim(10, 97)
    n_idxes = [0, 2, 6]
  else:
    colors = ['r', 'g', 'b']
    ax.set_ylim(0, 40)
    ax.set_xlim(0.015, 0.272)
    n_idxes = [5, 10, -1]
  # ax.yaxis.set_major_locator(MaxNLocator(x.size + 1))
  ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
  ax.set_xticks(x)
  ax.grid(True, which="both")
  # ax.yaxis.set_minor_locator(AutoMinorLocator())
  # ax.yaxis.grid(True, which="both")

  for n_idx in n_idxes:
    for solver, results in enumerate(z):
      if n_idx == n_idxes[0]:
        label = solvers[solver]
      else:
        label = None
      c = colors[solver]
      t = results[:, n_idx]
      ax.scatter(x, t,
                 c=colors[solver],
                 marker=markers[solver],
                 label=label)
      t_pred = fit_linear(x, t)
      ax.plot(x, t_pred, c=c)
      xtext, ytext = -37, 0
      xpos = 1
      t_idx = -1
      if param == 'k':
        if solver == 0:
          if n_idx == 6:
            t_idx = 1
            xpos = 0.31
            xtext, ytext = -37, -6
          elif n_idx == 2:
            t_idx = 4
            xpos = 0.83
            xtext, ytext = -8, 23
        elif solver == 1:
          if n_idx == 6:
            t_idx = 2
            xpos = 0.49
            xtext, ytext = -25, 7
      else:
        if problem.endswith("offline"):
          if solver == 0:
            if n_idx == -1:
              t_idx = 1
              xpos = 0.36
              xtext, ytext = 0, 46
          elif solver == 1:
            if n_idx == -1:
              t_idx = 2
              xpos = 0.645
              xtext, ytext = 0, 83
          elif solver == 2:
            if n_idx == -1:
              xtext, ytext = -70, -39
        else:
          if solver == 0:
            if n_idx == -1:
              t_idx = 1
              xpos = 0.37
              xtext, ytext = 0, 61
          elif solver == 1:
            if n_idx == -1:
              ytext = -5
          # elif solver == 2:
          #   if n_idx == -1:
          #     xtext, ytext = -70, -39

      ax.annotate(f"{n[n_idx]:>4}",
                  xy=(xpos, t[t_idx]), xytext=(xtext, ytext),
                  c=c,
                  xycoords=ax.get_yaxis_transform(),
                  textcoords="offset points",
                  size=tick_fontsize - 1, va="center")
      # if not (((solver == 0) and (n_idx in [2, 6]))
      #         or ((solver == 1) and (n_idx in [6]))):
      #   ax.annotate(f"{n[n_idx]:>4}",
      #               xy=(1, t[-1]), xytext=(-37, 0),
      #               c=c,
      #               xycoords=ax.get_yaxis_transform(),
      #               textcoords="offset points",
      #               size=14, va="center")

  ax.set_title(plot_surface_tilte(problem, param), fontsize=title_fontsize)
  # ax.set_title(problem, fontsize=title_fontsize)
  ax.legend(fontsize=legend_fontsize,
            loc="upper left",
            bbox_to_anchor=(0, 0.96))
  if save_plot:
    plt.savefig(f"{problem}_{xlabel}_linearity.png", dpi=fig.dpi)
  return fig


def _y(results, num_p, max_prob, param):
  """Extracts y axils values for the corresponding parameter of study."""
  if param in ['c', 'k']:
    return results[1: num_p + 1, 0]
  elif param == 'p_0':
    return results[1: num_p + 1, 0] * max_prob
  else:
    accepted_param_values = ['c', 'p_0', 'k']
    raise ValueError(f"<study> positional argument should be on of"
                     f" {accepted_param_values}. Instead, got {param}.")


@click.command()
@click.argument("filename", type=click.STRING)
@click.option('-p', "--problem",
              default=None, show_default=True,
              type=click.Choice(["replacement-paths-online",
                                 "replacement-paths-offline",
                                 "k-shortest-paths",
                                 None],
                                case_sensitive=False))
@click.option("--max-prob", default=0.28, show_default=True,
              help=("Defined by the graph model parameters, except from p_0.\n"
                    " max_p * p_0 = expected-graph-density\n"
                    " (See shortestpaths.graph_generator)"))
@click.option("--show-plot/--no-show-plot", default=True, show_default=True)
@click.option("--param", default='p_0', show_default=True,
              type=click.Choice(['p_0', 'k', 'c']))
@click.option("--save-plot/--no-save-plot", default=False, show_default=True)
def main(filename,
         problem,
         max_prob,
         show_plot,
         **kwargs):
  results = np.loadtxt(filename)
  param = kwargs["param"]
  problem, solvers, colors = problem_solvers_colors(filename, problem, param)
  kwargs.update({"problem": problem, "solvers": solvers})
  # number of p values
  num_p = (len(results) - 1) // len(solvers)

  x = results[0, 1:].astype(int)
  y = _y(results, num_p, max_prob, param)
  X, Y = np.meshgrid(x, y)
  Z_real, Z_pred, Z_norm = z_real_pred_norm(x, y, results,
                                            solvers, num_p, order=4)

  matshows = solvers_matshows(x, y, Z_norm, Z_real, **kwargs)
  matgains = gains_matshows(x, y, Z_real, **kwargs)
  surfaces = solvers_surfaces(X, Y, Z_pred, Z_real, y, colors, **kwargs)
  mselines = mse_lines(y, Z_real, x, **kwargs)

  if show_plot:
    plt.show()


if __name__ == '__main__':
  main()
