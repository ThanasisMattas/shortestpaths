# core.py is part of ShortestPaths
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
"""Bidirectional, dynamic and parallel algorithms for the replacement paths and
the k-shortest paths.

Versions implemented:
  - edge-exclusion
  - node-exclusion
  - on-line
  - off-line

Yen's algorithm is implemented as a compare base algorithm.
"""

import copy
from functools import partial
from concurrent.futures import ProcessPoolExecutor as Pool
import math
from shortestpaths.priorityq import PriorityQueue
from typing import Hashable, Literal

from shortestpaths import dijkstra
from shortestpaths.utils import time_this


def _replacement_path(failed_index: int,
                      failed: Hashable,
                      failing: Literal["nodes", "edges"],
                      shortest_path: list,
                      adj_list: list,
                      source: Hashable,
                      sink: Hashable,
                      to_visit: PriorityQueue,
                      visited: list,
                      bidirectional: bool) -> list:
  if failing == "nodes":
    if failed == source:
      return
    if bidirectional:
      r_path_data = dijkstra.bidirectional_dijkstra(adj_list,
                                                    source,
                                                    sink,
                                                    to_visit,
                                                    failed_nodes=[failed])
      return r_path_data
    else:
      r_visited, _ = dijkstra.dijkstra(adj_list,
                                       sink,
                                       to_visit,
                                       visited,
                                       failed_nodes=[failed])
      r_path_cost = r_visited[sink][0]
      r_path = dijkstra.extract_path(r_visited,
                                     source,
                                     sink,
                                     with_hop_weights=False)
  elif failing == "edges":
    tail = failed
    # Fail the edge, by setting its weight to inf.
    for neighbor in adj_list[tail]:
      if neighbor[0] == shortest_path[failed_index + 1]:
        adj_list[tail].remove(neighbor)
        adj_list[tail].add((neighbor[0], math.inf))
        # Find the replacement path.
        r_visited, _ = dijkstra.dijkstra(adj_list,
                                         sink,
                                         to_visit,
                                         visited)
        r_path_cost = r_visited[sink][0]
        r_path = dijkstra.extract_path(r_visited,
                                       source,
                                       sink,
                                       with_hop_weights=False)
        # Restore the failed edge weight
        adj_list[tail].remove((neighbor[0], math.inf))
        adj_list[tail].add(neighbor)
        failed = (tail, shortest_path[failed_index + 1])
        break
  else:
    raise ValueError(f"Unexpected value for failing: <{failing}>. It should"
                     " be either 'edges' or 'nodes'.")
  return [r_path, r_path_cost, failed]


@time_this
def replacement_paths(adj_list,
                      source,
                      sink,
                      failing="edges",
                      bidirectional=False,
                      parallel=False,
                      memoize_states=False):
  """Generates the replacement paths.

  Returns:
    paths_data (list)         : format:
                                [
                                  [path_1, path_1_cost, avoided_nodes_1],
                                  [path_2, path_2_cost, avoided_nodes_2],
                                  ...
                                ]
  """
  n = len(adj_list) - 1
  to_visit, visited = dijkstra.dijkstra_init(n, source)
  # Find the absolute shortest path.
  if bidirectional:
    [shortest_path, shortest_path_cost, _] = dijkstra.bidirectional_dijkstra(
      adj_list,
      source,
      sink,
      copy.deepcopy(to_visit),
      memoize_states=memoize_states,
      failed_nodes=None,
      verbose=False
    )
  else:
    i_visited, _ = dijkstra.dijkstra(adj_list,
                                     sink,
                                     copy.deepcopy(to_visit),
                                     copy.deepcopy(visited))
    shortest_path_cost = i_visited[sink][0]
    shortest_path = dijkstra.extract_path(i_visited,
                                          source,
                                          sink,
                                          with_hop_weights=False)
    # The key for each replacement path is the failed edge. The value is the
    # tuple (path: list, path_cost: int). replacement_paths list is initialized
    # with the shortest path.
    # r_paths = OrderedDict()
    # r_paths[(sink, None)] = (shortest_path, shortest_path_cost)
  r_paths = [[shortest_path, shortest_path_cost, None]]

  if parallel:
    _r_path = partial(_replacement_path,
                      failing=failing,
                      shortest_path=shortest_path,
                      adj_list=adj_list,
                      source=source,
                      sink=sink,
                      to_visit=to_visit,
                      visited=visited,
                      bidirectional=bidirectional)
    with Pool() as p:
      r_paths += p.map(_r_path, range(len(shortest_path) - 1), shortest_path[:-1])
      r_paths = filter(None, r_paths)
  else:
    for i, node in enumerate(shortest_path[:-1]):
      r_path = _replacement_path(i,
                                 node,
                                 failing,
                                 shortest_path,
                                 adj_list,
                                 source,
                                 sink,
                                 copy.deepcopy(to_visit),
                                 copy.deepcopy(visited),
                                 bidirectional)
      if r_path is not None:
        r_paths.append(r_path)

  return r_paths


@time_this
def k_shortest_paths(adj_list,
                     source,
                     sink,
                     k,
                     bidirectional=False,
                     parallel=False,
                     memoize_states=False):
  """Generates k_shortest_paths

  Returns:
    k_paths (list) : [[path: list, path_cost: int, failed_edge: tuple],]
  """
  n = len(adj_list) - 1
  to_visit, visited = dijkstra.dijkstra_init(n, source)
  # Find the absolute shortest path.
  if bidirectional:
    [shortest_path, shortest_path_cost, _] = dijkstra.bidirectional_dijkstra(
      adj_list,
      source,
      sink,
      copy.deepcopy(to_visit),
      memoize_states=memoize_states,
      failed_nodes=None,
      verbose=False
    )
    # __import__('ipdb').set_trace(context=9)
  else:
    visited, _ = dijkstra.dijkstra(adj_list,
                                   sink,
                                   copy.deepcopy(to_visit),
                                   copy.deepcopy(visited))
    shortest_path_cost = visited[sink][0]
    shortest_path = dijkstra.extract_path(visited,
                                          source,
                                          sink,
                                          with_hop_weights=False)

  k_paths = [[shortest_path, shortest_path_cost, None]]

  if k == 1:
    return k_paths
