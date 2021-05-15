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


OPTIMIZATION_MODES = ["-p", "-b", "-b -p", "-d", "-d -p"]
GRAPH_SIZES = [100]
FAILING = ["nodes", "edges"]
ONLINE = [""]
K = [5, 20]

class TestShortestPaths():

  def path_costs(self, completed_process):
    paths_str = completed_process.stdout.decode('utf-8').split("path 1", 1)[1]
    paths_list = paths_str.split("path")[1: -1]
    for i in range(len(paths_list)):
      paths_list[i] = paths_list[i].split("cost", 1)[1]
    return paths_list

  @pytest.mark.parametrize("k, n", [[k, n] for k in K for n in GRAPH_SIZES])
  def test_k_shortest_paths(self, k, n, ):
    yen_no_lawler_cmd = f"python -m shortestpaths -v -y -k {k} {n}"
    yen_lawler_cmd = f"python -m shortestpaths -v -l -k {k} {n}"
    yen_no_lawler = subprocess.run(yen_no_lawler_cmd.split(),
                                   stdout=subprocess.PIPE)
    yen_lawler = subprocess.run(yen_lawler_cmd.split(),
                                stdout=subprocess.PIPE)
    yen_no_lawler_out = self.path_costs(yen_no_lawler)
    yen_lawler_out = self.path_costs(yen_lawler)
    assert yen_no_lawler_out == yen_lawler_out

  @pytest.mark.parametrize(
    "optim_mode, n, failing, online",
    [[m, n, f, o]
     for m in OPTIMIZATION_MODES
     for n in GRAPH_SIZES
     for f in FAILING
     for o in ONLINE]
  )
  def test_replacement_paths(self, optim_mode, n, failing, online):
    reference_cmd = (f"python -m shortestpaths -v {n} replacement-paths"
                     f" --failing {failing} {online}")
    opt_cmd = (f"python -m shortestpaths -v {optim_mode} {n} replacement-paths"
               f" --failing {failing} {online}")
    reference = subprocess.run(reference_cmd.split(), stdout=subprocess.PIPE)
    opt = subprocess.run(opt_cmd.split(), stdout=subprocess.PIPE)
    reference_out = self.path_costs(reference)
    opt_out = self.path_costs(opt)
    assert reference_out == opt_out
