#!/usr/bin/env python3
import click
import numpy as np
import matplotlib.pyplot as plt


def set_edge_face_color(c):

  import matplotlib
  from packaging import version

  if version.parse(matplotlib.__version__) >= version.parse("3.3.3"):
    return c._facecolor3d, c._edgecolor3d
  else:
    return c._facecolors3d, c._edgecolors3d


@click.command()
@click.argument("filename", type=click.STRING)
@click.option('-p', "--problem",
              default="all", show_default=True,
              type=click.Choice(["replacement-paths-online",
                                 "replacement-paths-offline",
                                 "k-shortest-paths",
                                 "all"],
                                case_sensitive=False))
def main(filename, problem):
  results = np.loadtxt(filename)

  if problem == "k-shortest-paths":
    # number of solvers
    s = 4
    s1_label = "Yen"
    s2_label = "Lawler"
    s3_label = "Bidirectional"
    s4_label = "Dynamic Programming"
  else:
    s = 3
    s1_label = "Unidirectional"
    s2_label = "Bidirectional"
    s3_label = "Dynamic Programming"
  # number of p values
  p = (len(results) - 1) // s

  x = results[0, 1:].astype(int)
  y = results[1: p + 1, 0] / 10

  s1 = results[1: p + 1, 1:]
  s2 = results[p + 1: 2 * p + 1, 1:]
  s3 = results[2 * p + 1: 3 * p + 1, 1:]
  if problem == "k-shortest-paths":
    s4 = results[3 * p + 1: 4 * p + 1, 1:]

  X, Y = np.meshgrid(x, y)

  fig = plt.figure()
  sub = fig.add_subplot(111, projection="3d")
  sub.set_xlabel("n (nodes")
  sub.set_ylabel("${p_0}$")
  sub.set_zlabel("time (s)")
  sub.grid(False)
  fig.suptitle(f"{problem} profiling")

  c1 = sub.plot_surface(X, Y, s1,
                        rstride=1, cstride=1, linewidth=0,
                        color='b', alpha=0.4,
                        shade=True, antialiased=False, label=s1_label)
  c1._facecolors2d, c1._edgecolors2d = set_edge_face_color(c1)

  c2 = sub.plot_surface(X, Y, s2,
                        rstride=1, cstride=1, linewidth=0,
                        color='r', alpha=0.4,
                        shade=True, antialiased=False, label=s2_label)
  c2._facecolors2d, c2._edgecolors2d = set_edge_face_color(c2)

  c3 = sub.plot_surface(X, Y, s3,
                        rstride=1, cstride=1, linewidth=0,
                        color='y', alpha=0.4,
                        shade=True, antialiased=False, label=s3_label)
  c3._facecolors2d, c3._edgecolors2d = set_edge_face_color(c3)

  if problem == "k-shortest-paths":
    c4 = sub.plot_surface(X, Y, s4,
                          rstride=1, cstride=1, linewidth=0,
                          color='g', alpha=0.4,
                          shade=True, antialiased=False, label=s4_label)
    c4._facecolors2d, c4._edgecolors2d = set_edge_face_color(c4)

  plt.legend()
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  main()
