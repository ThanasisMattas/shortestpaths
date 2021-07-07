# shortestpaths_test.py is part of ShortestPaths
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
from pathlib import Path
import subprocess
import sys

import pytest

from shortestpaths import io


cwd = Path(os.getcwd())
home_dir = str(cwd.parent)
sys.path.insert(0, home_dir)
os.environ["BIDIRECTIONAL_SYNC"] = '1'


SOLVER = ["-p", "-b", "-p -b", "-d"]
GRAPH_SIZE = [100, 300]
FAILING = ["nodes", "edges"]
ONLINE = ["--online", ""]
DIRECTED = ["--directed", ""]
SEED = [0, 1]
K = [20]


class TestShortestPaths():
  """Integration tests

  Tests if all search modes return the same results against a reference search.
  #TODO: 1. Build the graph on setup or make graphs dataset.
         2. Make reference-search dataset.
  """
  def path_costs(self, completed_process):
    paths_str = completed_process.stdout.decode('utf-8').split("path 1", 1)[1]
    paths_list = paths_str.split("path")[1: -1]
    for i in range(len(paths_list)):
      paths_list[i] = paths_list[i].split("cost", 1)[1]
    return paths_list

  @pytest.mark.parametrize(
    "r, k, n, d, s",
    [[r, k, n, d, s]
     for r in SOLVER + ["-y", "-l"]
     for k in K
     for n in GRAPH_SIZE
     for d in DIRECTED
     for s in SEED]
  )
  def test_k_shortest_paths(self, r, k, n, d, s):
    reference_cmd = f"python -m shortestpaths -v --seed {s} -l {d} -k {k} -n {n}"
    solver_cmd = f"python -m shortestpaths -v --seed {s} {r} {d} -k {k} -n {n}"
    reference = subprocess.run(reference_cmd.split(),
                               stdout=subprocess.PIPE)
    solver = subprocess.run(solver_cmd.split(),
                            stdout=subprocess.PIPE)
    reference_out = self.path_costs(reference)
    solver_out = self.path_costs(solver)
    assert reference_out == solver_out

  @pytest.mark.parametrize(
    "r, n, f, o, d, s",
    [[r, n, f, o, d, s]
     for r in SOLVER
     for n in GRAPH_SIZE
     for f in FAILING
     for o in ONLINE
     for d in DIRECTED
     for s in SEED]
  )
  def test_replacement_paths(self, r, n, f, o, d, s):
    reference_cmd = (f"python -m shortestpaths -v --seed {s} {d} -n {n}"
                     f" replacement-paths --failing {f} {o}")
    solver_cmd = (f"python -m shortestpaths -v --seed {s} {d} {r} -n {n}"
                  f" replacement-paths --failing {f} {o}")
    reference = subprocess.run(reference_cmd.split(), stdout=subprocess.PIPE)
    solver = subprocess.run(solver_cmd.split(), stdout=subprocess.PIPE)
    reference_out = self.path_costs(reference)
    solver_out = self.path_costs(solver)
    assert reference_out == solver_out


class TestIO():
  """io.py tests"""
  def setup(self):
    self.adj_list = [set(), {(4, 5), (1, 2)}, {(34, 1), (3, 6)}]
    self.csvfile = ".test_io_dataset.csv"
    self.new_graph_token = "<new-graph>"

  def teardown(self):
    self.adj_list.clear()
    if os.path.isfile(self.csvfile):
      os.remove(self.csvfile)

  def test_append_graph_to_csv(self):
    io.append_graph_to_csv(self.csvfile, self.adj_list)
    with open(self.csvfile, newline='') as f:
      data_reader = csv.reader(f, delimiter=',')
      for i, entry in enumerate(data_reader):
        if i == 0:
          assert entry[0] == self.new_graph_token
        elif i == 1:
          assert entry[0] == "set()"
        else:
          assert ast.literal_eval(", ".join(entry)) == self.adj_list[i - 1]

  @pytest.mark.parametrize("num_graphs", [1, 2, 4])
  def test_load_graphs_from_csv(self, num_graphs):
    for _ in range(num_graphs):
      io.append_graph_to_csv(self.csvfile, self.adj_list)
    graph_counter = 0
    for adj in io.load_graphs_from_csv(self.csvfile):
      graph_counter += 1
      assert adj == self.adj_list
    assert graph_counter == num_graphs
