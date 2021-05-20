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
"""Integrated tests

- replacement_paths:
    tests if all 5 optimization modes output the same results for all
    4 different scenarios.
- k_shortest_paths:
    tests if Yen's algorithm outputs the same results with and without Lawler's
    modification.

The results are not tested against an expected output, because for the time be-
ing the graph generation model is frequently being modified.
"""

"""
python -m shortestpaths -v -k 5 -y 150
python -m shortestpaths -v -k 5 -l 150

python -m shortestpaths -v 150 replacement-paths
python -m shortestpaths -v -p 150 replacement-paths
python -m shortestpaths -v -b 150 replacement-paths
python -m shortestpaths -v -b -p 150 replacement-paths
python -m shortestpaths -v -d 150 replacement-paths
python -m shortestpaths -v -d -p 150 replacement-paths

python -m shortestpaths -v 150 replacement-paths --online
python -m shortestpaths -v -p 150 replacement-paths --online
python -m shortestpaths -v -b 150 replacement-paths --online
python -m shortestpaths -v -b -p 150 replacement-paths --online
python -m shortestpaths -v -d 150 replacement-paths --online
python -m shortestpaths -v -d -p 150 replacement-paths --online

python -m shortestpaths -v 150 replacement-paths -f "edges"
python -m shortestpaths -v -p 150 replacement-paths -f "edges"
python -m shortestpaths -v -b 150 replacement-paths -f "edges"
python -m shortestpaths -v -b -p 150 replacement-paths -f "edges"
python -m shortestpaths -v -d 150 replacement-paths -f "edges"
python -m shortestpaths -v -d -p 150 replacement-paths -f "edges"

python -m shortestpaths -v 150 replacement-paths -f "edges" --online
python -m shortestpaths -v -p 150 replacement-paths -f "edges" --online
python -m shortestpaths -v -b 150 replacement-paths -f "edges" --online
python -m shortestpaths -v -b -p 150 replacement-paths -f "edges" --online
python -m shortestpaths -v -d 150 replacement-paths -f "edges" --online
python -m shortestpaths -v -d -p 150 replacement-paths -f "edges" --online
"""

import os
import subprocess
from pathlib import Path
import sys

import pytest

cwd = Path(os.getcwd())
home_dir = str(cwd.parent)
sys.path.insert(0, home_dir)
os.environ["BIDIRECTIONAL_SYNC"] = '1'


SOLVER = ["-p", "-b", "-b -p", "-d"]
GRAPH_SIZES = [150, 200]
FAILING = ["nodes", "edges"]
ONLINE = ["--online", ""]
K = [20]

class TestShortestPaths():

  def path_costs(self, completed_process):
    paths_str = completed_process.stdout.decode('utf-8').split("path 1", 1)[1]
    paths_list = paths_str.split("path")[1: -1]
    for i in range(len(paths_list)):
      paths_list[i] = paths_list[i].split("cost", 1)[1]
    return paths_list

  @pytest.mark.parametrize(
    "solver, k, n",
    [[s, k, n]
     for s in SOLVER + ["-y", "-l"] if 'd' not in s
     for k in K
     for n in GRAPH_SIZES]
  )
  def test_k_shortest_paths(self, solver, k, n):
    reference_cmd = f"python -m shortestpaths -v -l -k {k} {n}"
    solver_cmd = f"python -m shortestpaths -v {solver} -k {k} {n}"
    reference = subprocess.run(reference_cmd.split(),
                               stdout=subprocess.PIPE)
    solver = subprocess.run(solver_cmd.split(),
                            stdout=subprocess.PIPE)
    reference_out = self.path_costs(reference)
    solver_out = self.path_costs(solver)
    assert reference_out == solver_out

  @pytest.mark.parametrize(
    "solver, n, failing, online",
    [[s, n, f, o]
     for s in SOLVER
     for n in GRAPH_SIZES
     for f in FAILING
     for o in ONLINE]
  )
  def test_replacement_paths(self, solver, n, failing, online):
    reference_cmd = (f"python -m shortestpaths -v {n} replacement-paths"
                     f" --failing {failing} {online}")
    solver_cmd = (f"python -m shortestpaths -v {solver} {n} replacement-paths"
                  f" --failing {failing} {online}")
    reference = subprocess.run(reference_cmd.split(), stdout=subprocess.PIPE)
    solver = subprocess.run(solver_cmd.split(), stdout=subprocess.PIPE)
    reference_out = self.path_costs(reference)
    solver_out = self.path_costs(solver)
    assert reference_out == solver_out
