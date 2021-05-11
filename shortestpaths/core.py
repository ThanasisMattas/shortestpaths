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
from shortestpaths.utils import print_heap, time_this  # noqa: F401


def _first_shortest_path(adj_list,
                         source,
                         sink,
                         to_visit,
                         to_visit_reverse=False,
                         visited=False,
                         inverted_adj_list=None,
                         bidirectional=False,
                         dynamic=False,
                         mode="k_shortest_paths",
                         verbose=0):
  tapes = None
  if bidirectional:
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
                                                          mode=mode,
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
                                          mode=mode,
                                          verbose=verbose)
  else:
    initial_visited = dijkstra.dijkstra(adj_list,
                                        sink,
                                        copy.deepcopy(to_visit),
                                        copy.deepcopy(visited))
    shortest_path_cost = initial_visited[sink][0]
    shortest_path, cum_hop_weights = dijkstra.extract_path(
      source,
      sink,
      initial_visited,
      cum_hop_weights=(mode == "k_shortest_paths"),
      verbose=verbose)
    if mode == "k_shortest_paths":
      path_data = [shortest_path, cum_hop_weights, shortest_path_cost]
    else:
      path_data = [shortest_path, shortest_path_cost, None]

  return path_data, tapes


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
                      tapes: list = None,
                      verbose: int = 0) -> list:
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
        tapes=tapes,
        mode="replacement-paths",
        verbose=verbose
      )
      return path_data
    else:
      repl_visited = dijkstra.dijkstra(adj_list,
                                       sink,
                                       to_visit,
                                       visited,
                                       failed)
      repl_path_cost = repl_visited[sink][0]
      repl_path, _ = dijkstra.extract_path(source,
                                           sink,
                                           repl_visited,
                                           cum_hop_weights=False)
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
          repl_path, _ = dijkstra.extract_path(source,
                                               sink,
                                               repl_visited,
                                               cum_hop_weights=False)
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
  if bidirectional:
    inverted_adj_list = dijkstra.invert_adj_list(adj_list)
  else:
    inverted_adj_list = None

  # Find the absolute shortest path.
  path_data, tapes = _first_shortest_path(adj_list,
                                          source,
                                          sink,
                                          to_visit,
                                          to_visit_reverse,
                                          visited,
                                          inverted_adj_list,
                                          bidirectional,
                                          dynamic,
                                          mode="replacement_paths",
                                          verbose=verbose)
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

  else:  # not parallel
    for i, node in enumerate(shortest_path[:-1]):
      # The source cannot fail, but when failing == "edges", the source consti-
      # tudes the tail of the 1st edge that will fail.
      if (failing == "nodes") and (i == 0):
        continue

      # In case of replacement-paths and failing == "edges" and i == 0, a state
      # is't recorded on tape, because it is the same with the state we get at
      # initialization.
      if (not dynamic) or ((failing == "edges") and (i == 0)):
        new_to_visit = copy.deepcopy(to_visit)
        new_to_visite_reverse = copy.deepcopy(to_visit_reverse)
        new_visited = copy.deepcopy(visited)
        tapes = None
      else:
        # All necessary data will be retrieved from tapes.
        new_to_visit = new_to_visite_reverse = new_visited = None
      repl_path = _replacement_path(i,
                                    node,
                                    failing,
                                    shortest_path,
                                    adj_list,
                                    source,
                                    sink,
                                    new_to_visit,
                                    new_to_visite_reverse,
                                    new_visited,
                                    bidirectional,
                                    inverted_adj_list,
                                    tapes)

      if repl_path is not None:
        repl_paths.append(repl_path)

  return repl_paths


def _yen(sink,
         adj_list,
         to_visit,
         visited,
         K,
         shortest_path,
         shortest_path_cost,
         cum_hop_weights):
  k_paths = [[shortest_path, shortest_path_cost, None]]
  # This is the B list of the Yen's algorithm, holding the potential shortest
  # paths.
  prospects = []
  heapq.heapify(prospects)
  for k in range(1, K):
    # Construct the deviation paths of the last found shortest path.
    last_path = k_paths[-1][0]
    for i, u in enumerate(last_path[:-1]):
      # Fail the (i, i + 1) edges of all found shortest paths.
      # {head: (head, edge_cost)}
      failed_edges = dict()
      for j in k_paths:
        if j[0][:i + 1] == last_path[:i + 1]:
          failed_edges[j[0][i + 1]] = None
      for v, uv_weight in adj_list[u]:
        if v in failed_edges.keys():
          failed_edges[v] = (v, uv_weight)
      for v, edge in failed_edges.items():
        adj_list[u].remove(edge)
        adj_list[u].add((v, math.inf))

      # Remove the Root path nodes from the to_visit PriorityQueue.
      new_to_visit = copy.deepcopy(to_visit)
      for root_u in last_path[:i]:
        del new_to_visit[root_u]

      # Set i as source and initialize it's path cost to source-i path cost.
      new_to_visit[u] = [cum_hop_weights[i], u, u]
      new_visited = dijkstra.dijkstra(adj_list,
                                      sink,
                                      new_to_visit,
                                      copy.deepcopy(visited))
      prospect_cost = new_visited[sink][0]
      i_sink_path, i_sink_hop_weights = dijkstra.extract_path(
        u,
        sink,
        new_visited,
        cum_hop_weights=True,
      )
      if i_sink_path:
        prospect = last_path[:i] + i_sink_path
        prospect_hop_weights = cum_hop_weights[:i] + i_sink_hop_weights

        if ((len(prospects) < K - k)
                or (prospect_cost
                    < heapq.nsmallest(K - k, prospects)[-1][0])):
          # Check if the prospect is already found
          prospect_already_found = False
          for p_cost, p, p_hop_weights in prospects:
            if (p_cost == prospect_cost) and (p == prospect):
              prospect_already_found = True
              break
          if not prospect_already_found:
            heapq.heappush(prospects,
                           (prospect_cost, prospect, prospect_hop_weights))
      # Restore the failed edges.
      for v, edge in failed_edges.items():
        adj_list[u].remove((v, math.inf))
        adj_list[u].add(edge)
      failed_edges.clear()
    # Add the best prospect to the k_paths list
    if prospects:
      kth_path_cost, kth_path, cum_hop_weights = heapq.heappop(prospects)
      k_paths.append([kth_path, kth_path_cost, None])
    else:
      break
  return k_paths


@time_this
def k_shortest_paths(adj_list,
                     source,
                     sink,
                     K,
                     bidirectional=False,
                     parallel=False,
                     dynamic=False,
                     verbose=0):
  """Generates k_shortest_paths

  Returns:
    k_paths (list) : [[path: list, path_cost: int, failed_edge: tuple],]
  """
  n = len(adj_list) - 1
  if bidirectional:
    inverted_adj_list = dijkstra.invert_adj_list(adj_list)
  else:
    inverted_adj_list = None
  to_visit, visited, to_visit_reverse = dijkstra.dijkstra_init(n,
                                                               source,
                                                               sink,
                                                               bidirectional)
  # Find the absolute shortest path.
  [shortest_path, cum_hop_weights, shortest_path_cost], tapes = \
      _first_shortest_path(adj_list,
                           source,
                           sink,
                           to_visit,
                           to_visit_reverse,
                           visited,
                           inverted_adj_list,
                           bidirectional,
                           dynamic,
                           mode="k_shortest_paths",
                           verbose=verbose)

  k_paths = _yen(sink,
                 adj_list,
                 to_visit,
                 visited,
                 K,
                 shortest_path,
                 shortest_path_cost,
                 cum_hop_weights)
  return k_paths
