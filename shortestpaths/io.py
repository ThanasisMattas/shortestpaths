# io.py is part of ShortestPaths
#
# ShortestPaths is free software; you may redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version. You should have received a copy of the GNU General Pu-
# blic License along with this program. If not, see
# <https://www.gnu.org/licenses/>.
#
# (C) 2020 Athanasios Mattas
# ==========================================================================
import ast
import csv
import os
import pickle

import numpy as np


def append_graph_to_csv(csvfile, adj, new_graph_token="<new-graph>"):
  """Appends a graph to a csv file, starting with a <new-graph> token."""
  adj = np.array(adj)
  with open(csvfile, 'a') as af:
    af.write(f"{new_graph_token}\n")
    np.savetxt(af, adj, fmt='%s')


def read_graphs_from_csv(csvfile,
                         num_graphs=None,
                         graph_offset=0,
                         new_graph_token="<new-graph>"):
  """Creates a graph generator, reading graphs from a csv.

  Args:
    csvfile (str)         : the csv filename
    num_graphs (int)      : if provided, num_graphs to yield (default: None)
    graph_offset (int)    : the first graph to yield (default: 0)
    new_graph_token (str) : denotes that a new graph is following
                            (default: '<new-graph>')

  Yields:
    adj (list)            : adjascency list or matrix
  """
  if not os.path.isfile(csvfile):
    return

  adj = []
  graph_counter = 0
  with open(csvfile, newline='') as f:
    data_reader = csv.reader(f, delimiter=',')
    for entry in data_reader:

      if entry[0] == new_graph_token:
        if adj:
          graph_counter += 1
          if graph_counter >= graph_offset:
            yield adj
        if (num_graphs) and (graph_counter == num_graphs):
          return
        adj.clear()
        continue

      try:
        adj.append(ast.literal_eval(', '.join(entry)))
      except ValueError:
        adj.append(set())

    if (adj) and (graph_counter >= graph_offset):
      yield adj


def load_graphs(filename):
  """Creates a graph generator, reading graphs from a csv.

  NOTE: The number of pickled graphs cannot be infered before loading them all.
  """
  with open(filename, "rb") as f:
    while True:
      try:
        yield pickle.load(f)
      except EOFError:
        return
