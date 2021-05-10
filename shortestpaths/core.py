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

from concurrent.futures import ProcessPoolExecutor
import copy
from functools import partial
import heapq
import math
from typing import Hashable, Literal

from shortestpaths import dijkstra
from shortestpaths.priorityq import PriorityQueue
from shortestpaths.utils import time_this


# @time_this(wall_clock=True)
def _replacement_path(failed_path_idx: int,
                      failed: Hashable,
                      failing: Literal["nodes", "edges"],
                      shortest_path: list,
                      adj_list: list,
                      source: Hashable,
                      sink: Hashable,
                      to_visit: PriorityQueue = None,
                      to_visit_reverse: PriorityQueue = None,
                      visited: list = None,
                      bidirectional: bool = False,
                      inverted_adj_list: list = None,
                      tapes: list = None) -> list:
  if failing == "nodes":
    if failed == source:
      return

    if bidirectional:
      path_data = dijkstra.bidirectional_dijkstra(
        adj_list,
        inverted_adj_list,
        source,
        sink,
        to_visit,
        to_visit_reverse,
        failed_path_idx=failed_path_idx,
        failed=failed,
        tapes=tapes
      )
      return path_data
    else:
      repl_visited = dijkstra.dijkstra(adj_list,
                                       sink,
                                       to_visit,
                                       visited,
                                       failed)
      repl_path_cost = repl_visited[sink][0]
      repl_path = dijkstra.extract_path(source,
                                        sink,
                                        repl_visited,
                                        with_hop_weights=False)
  elif failing == "edges":
    tail = failed
    head = shortest_path[failed_path_idx + 1]
    # Fail the edge, by setting its weight to inf.
    for neighbor in adj_list[tail]:
      if neighbor[0] == head:
        adj_list[tail].remove(neighbor)
        adj_list[tail].add((neighbor[0], math.inf))
        # Find the replacement path.
        if bidirectional:
          [repl_path, repl_path_cost, failed] = \
            dijkstra.bidirectional_dijkstra(adj_list,
                                            source,
                                            sink,
                                            to_visit,
                                            tapes=tapes)
        else:
          repl_visited = dijkstra.dijkstra(adj_list,
                                           sink,
                                           to_visit,
                                           visited)
          repl_path_cost = repl_visited[sink][0]
          repl_path = dijkstra.extract_path(source,
                                            sink,
                                            repl_visited,
                                            with_hop_weights=False)
        # Restore the failed edge weight
        adj_list[tail].remove((neighbor[0], math.inf))
        adj_list[tail].add(neighbor)
        failed = (tail, head)
        break
  else:
    raise ValueError(f"Unexpected value for failing: <{failing}>. It should"
                     " be either 'edges' or 'nodes'.")
  return [repl_path, repl_path_cost, failed]


# @time_this(wall_clock=True)
def replacement_paths(adj_list,
                      n,
                      source,
                      sink,
                      failing="nodes",
                      bidirectional=False,
                      parallel=False,
                      dynamic=False,
                      verbose=0):
  """Generates the replacement paths.

  Returns:
    paths_data (list)         : format:
                                [
                                  [path_1, path_1_cost, avoided_nodes_1],
                                  [path_2, path_2_cost, avoided_nodes_2],
                                  ...
                                ]
  """
  to_visit, visited, to_visit_reverse = dijkstra.dijkstra_init(n,
                                                               source,
                                                               sink,
                                                               bidirectional)
  # Find the absolute shortest path.
  if bidirectional:
    inverted_adj_list = dijkstra.invert_adj_list(adj_list)

    # If dynamic, 2 recording sessions are executed. At the first, the absolute
    # shortest path will be generated, as well as the visited nodes sequence
    # will be recorded on a tape. At the second, for all the intermediate nodes
    # of the path, the state that corresponds to the immediately proceding vi-
    # sited node will be recorded on a tape, during both searches.
    if dynamic:
      path_data, tapes = dijkstra.bidirectional_recording(adj_list,
                                                          inverted_adj_list,
                                                          source,
                                                          sink,
                                                          to_visit,
                                                          to_visit_reverse,
                                                          visited,
                                                          verbose=verbose)
    else:
      path_data = \
          dijkstra.bidirectional_dijkstra(adj_list,
                                          inverted_adj_list,
                                          source,
                                          sink,
                                          copy.deepcopy(to_visit),
                                          copy.deepcopy(to_visit_reverse),
                                          copy.deepcopy(visited),
                                          verbose=verbose)
  else:
    inverted_adj_list = None
    initial_visited = dijkstra.dijkstra(adj_list,
                                        sink,
                                        copy.deepcopy(to_visit),
                                        copy.deepcopy(visited))
    shortest_path_cost = initial_visited[sink][0]
    shortest_path = dijkstra.extract_path(source, sink, initial_visited)
    path_data = [shortest_path, shortest_path_cost, None]

  repl_paths = [path_data]

  # Next, find the replacement paths.
  shortest_path = path_data[0]
  if parallel:
    if dynamic:
      if failing == "edges":
        to_visit_values = \
            ([copy.deepcopy(to_visit)]
             + [None for _ in range(len(shortest_path) - 2)])
        to_visit_reverse_values = \
            ([copy.deepcopy(to_visit_reverse)]
             + [None for _ in range(len(shortest_path) - 2)])
        visited_values = \
            ([copy.deepcopy(visited)]
             + [None for _ in range(len(shortest_path) - 2)])
      else:
        to_visit_values = None
        to_visit_reverse_values = None
        visited_values = None
    else:
      to_visit_values = to_visit
      to_visit_reverse_values = to_visit_reverse
      visited_values = visited
      tapes = None

    _repl_path = partial(_replacement_path,
                         failing=failing,
                         shortest_path=shortest_path,
                         adj_list=adj_list,
                         source=source,
                         sink=sink,
                         to_visit=to_visit_values,
                         to_visit_reverse=to_visit_reverse_values,
                         visited=visited_values,
                         bidirectional=bidirectional,
                         inverted_adj_list=inverted_adj_list,
                         tapes=tapes)

    with ProcessPoolExecutor() as p:
      repl_paths += p.map(_repl_path,
                          range(len(shortest_path) - 1),
                          shortest_path[:-1])
      repl_paths = list(filter(None, repl_paths))

  else:
    for i, node in enumerate(shortest_path[:-1]):
      # The source cannot fail, but when failing == "edges", the source consti-
      # tudes the tail of the 1st edge that will fail.
      if (failing == "nodes") and (i == 0):
        continue

      if (not dynamic) or ((failing == "edges") and (i == 0)):
        repl_path = _replacement_path(i,
                                      node,
                                      failing,
                                      shortest_path,
                                      adj_list,
                                      source,
                                      sink,
                                      copy.deepcopy(to_visit),
                                      copy.deepcopy(to_visit_reverse),
                                      copy.deepcopy(visited),
                                      bidirectional,
                                      inverted_adj_list)
      else:
        repl_path = _replacement_path(i,
                                      node,
                                      failing,
                                      shortest_path,
                                      adj_list,
                                      source,
                                      sink,
                                      bidirectional=bidirectional,
                                      inverted_adj_list=inverted_adj_list,
                                      tapes=tapes)

      if repl_path is not None:
        repl_paths.append(repl_path)

  return repl_paths


@time_this
def k_shortest_paths(adj_list,
                     source,
                     sink,
                     k,
                     bidirectional=False,
                     parallel=False,
                     dynamic=False):
  """Generates k_shortest_paths

  Returns:
    k_paths (list) : [[path: list, path_cost: int, failed_edge: tuple],]
  """
  n = len(adj_list) - 1
  to_visit, visited, to_visit_reverse = dijkstra.dijkstra_init(n,
                                                               source,
                                                               sink,
                                                               bidirectional)
  # Find the absolute shortest path.
  if bidirectional:
    [shortest_path, shortest_path_cost, _] = dijkstra.bidirectional_dijkstra(
      adj_list,
      source,
      sink,
      copy.deepcopy(to_visit),
      recording=False,
    )
  else:
    new_visited = dijkstra.dijkstra(adj_list,
                                    sink,
                                    copy.deepcopy(to_visit),
                                    copy.deepcopy(visited))
    shortest_path_cost = new_visited[sink][0]
    shortest_path = dijkstra.extract_path(source,
                                          sink,
                                          new_visited,
                                          with_hop_weights=True,
                                          cumulative=True)

  k_paths = [[shortest_path, shortest_path_cost, None]]

  # This is the B list of the Yen's algorithm, holding the potential shortest
  # paths.
  prospects = []
  heapq.heapify(prospects)

  for _ in range(k - 1):
    # Construct the deviation paths of the last found shortest path.
    last_path = k_paths[-1][0]
    [last_path, path_cum_weights] = list(zip(*last_path))
    k_paths[-1][0] = last_path
    for i in range(len(last_path[:-1])):
      # Fail the (i, i + 1) edges of all found shortest paths.
      failed_edges = dict()
      for j in k_paths:
        if j[0][:i + 1] == last_path[:i + 1]:
          failed_edges[j[0][i + 1]] = None
      for neighbor in adj_list[last_path[i]]:
        if neighbor[0] in failed_edges.keys():
          failed_edges[neighbor[0]] = neighbor
      adj_list[last_path[i]] = adj_list[last_path[i]] - set(failed_edges.values())
      for key in failed_edges.keys():
        adj_list[last_path[i]].add((key, math.inf))
      new_to_visit = copy.deepcopy(to_visit)
      # Remove the Root path nodes from the to_visit PriorityQueue.
      for node in last_path[:i]:
        del new_to_visit[node]
      # Set i as source.
      new_to_visit[last_path[i]] = [0, last_path[i], last_path[i]]
      new_visited = dijkstra.dijkstra(adj_list,
                                      sink,
                                      new_to_visit,
                                      copy.deepcopy(visited))
      prospect_path_cost = new_visited[sink][0] + path_cum_weights[i]
      prospect_path = dijkstra.extract_path(last_path[i],
                                            sink,
                                            new_visited,
                                            with_hop_weights=True,
                                            cumulative=True)
      if prospect_path:
        [prospect_path_no_weights, __] = list(zip(*prospect_path))
        prospect_path_no_weights = last_path[:i] + prospect_path_no_weights

        prospects.append((prospect_path_cost, prospect_path))
      # Restore the failed edges.
      for ke, va in failed_edges.items():
        adj_list[last_path[i]].remove((ke, math.inf))
        adj_list[last_path[i]].add(va)
      failed_edges.clear()
    # Add the best prospect to the k_paths list
    kth_path_cost, kth_path = heapq.heappop(prospects)
    k_paths.append([kth_path, kth_path_cost, None])
  k_paths[-1][0] = prospect_path_no_weights
  return k_paths
