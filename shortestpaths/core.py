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
                         failing=None,
                         mode="k_shortest_paths",
                         online=False,
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
                                                          failing=failing,
                                                          mode=mode,
                                                          online=online,
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
                                          online=online,
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
      cum_hop_weights=(mode == "k_shortest_paths") or (online),
      verbose=verbose)

    if cum_hop_weights:
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
                      online: bool = False,
                      cum_hop_weights: list = None,
                      verbose: int = 0) -> list:
  if (failing == "nodes") and (failed == source):
    return

  if failing == "nodes":
    if online:
      # Delete the nodes of the root path from the PriorityQueue.
      for u in shortest_path[:failed_path_idx - 1]:
        del to_visit[u]
      # The spur node becomes the source.
      source = shortest_path[failed_path_idx - 1]
      to_visit[source] = [0, source, source]
      # Initialize the path cost with the root_cost.
      if (bidirectional) and (not tapes):
        # Delete the nodes of the root path from the reverse PriorityQueue.
        for u in shortest_path[:failed_path_idx - 1]:
          del to_visit_reverse[u]

    if bidirectional:
      path_data = dijkstra.bidirectional_dijkstra(
        adj_list,
        inverted_adj_list,
        source,
        sink,
        to_visit,
        to_visit_reverse,
        visited,
        failed_path_idx=failed_path_idx,
        failed=failed,
        tapes=tapes,
        mode="replacement-paths",
        online=online,
        shortest_path=shortest_path,
        verbose=verbose
      )
    else:
      repl_visited = dijkstra.dijkstra(adj_list,
                                       sink,
                                       to_visit,
                                       visited,
                                       failed)
      repl_path_cost = repl_visited[sink][0]
      repl_path, repl_weights = dijkstra.extract_path(
        source,
        sink,
        repl_visited,
        cum_hop_weights=online,
        verbose=verbose
      )
  elif failing == "edges":
    # When failing is nodes, the spur path will start *before* the failed node.
    # On the contrary, when failing is edges, the spur path will start *on* the
    # tail of the failed edge. So, in order to keep the same code for both in-
    # stances, failed_path_idx will be increased by one, when failing edges.
    tail = failed
    head = shortest_path[failed_path_idx + 1]

    if online:
      # Delete the nodes of the root path from the PriorityQueue.
      for u in shortest_path[:failed_path_idx]:
        del to_visit[u]
      # The spur node becomes the source.
      source = tail
      to_visit[source] = [0, source, source]
      # Initialize the path cost with the root_cost.
      if (bidirectional) and (not tapes):
        # Delete the nodes of the root path from the reverse PriorityQueue.
        for u in shortest_path[:failed_path_idx]:
          del to_visit_reverse[u]

    # Fail the edge.
    for neighbor in adj_list[tail]:
      if neighbor[0] == head:
        adj_list[tail].remove(neighbor)
        # Find the replacement path.
        if bidirectional:
          # Fail the edge on the inverted_adj_list.
          for ne in inverted_adj_list[head]:
            if ne[0] == tail:
              inverted_adj_list[head].remove(ne)
              path_data = dijkstra.bidirectional_dijkstra(
                adj_list,
                inverted_adj_list,
                source,
                sink,
                to_visit,
                to_visit_reverse,
                visited,
                failed_path_idx=(failed_path_idx, failed_path_idx + 1),
                failed=(tail, head),
                tapes=tapes,
                mode="replacement-paths",
                online=online,
                shortest_path=shortest_path,
                verbose=verbose
              )
              # Reconnect the failed edge.
              inverted_adj_list[head].add(ne)
              break
        else:
          repl_visited = dijkstra.dijkstra(adj_list,
                                           sink,
                                           to_visit,
                                           visited)
          repl_path_cost = repl_visited[sink][0]
          repl_path, repl_weights = dijkstra.extract_path(
            source,
            sink,
            repl_visited,
            cum_hop_weights=online,
            verbose=verbose
          )
        # Reconnect the failed edge.
        adj_list[tail].add(neighbor)
        break
    failed = (tail, head)
  else:
    raise ValueError(f"Unexpected value for failing: <{failing}>. It should"
                     " be either 'edges' or 'nodes'.")

  if online:
    # path_data[0] = shortest_path[: failed_path_idx - 1] + path_data[0]
    # path_data[2] += cum_hop_weights[failed_path_idx - 1]
    if failing == "edges":
      failed_path_idx += 1

    if bidirectional:
      if path_data[0]:
        path_data = [shortest_path[: failed_path_idx - 1] + path_data[0],
                     path_data[2] + cum_hop_weights[failed_path_idx - 1],
                     failed]
      else:
        path_data = [None, None, None]
    else:
      if repl_path:
        path_data = [shortest_path[: failed_path_idx - 1] + repl_path,
                     repl_path_cost + cum_hop_weights[failed_path_idx - 1],
                     failed]
      else:
        path_data = [None, None, None]
  elif not bidirectional:
    path_data = [repl_path, repl_path_cost, failed]

  return path_data


@time_this(wall_clock=True)
def replacement_paths(adj_list,
                      n,
                      source,
                      sink,
                      failing="nodes",
                      bidirectional=False,
                      parallel=False,
                      dynamic=False,
                      online=False,
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
                                          failing=failing,
                                          mode="replacement_paths",
                                          online=online,
                                          verbose=verbose)
  if online:
    [shortest_path, cum_hop_weights, shortest_path_cost] = path_data
    repl_paths = [[shortest_path, shortest_path_cost, None]]
  else:
    repl_paths = [path_data]
    cum_hop_weights = None

  # Next, find the replacement paths.
  shortest_path = path_data[0]

  if parallel:
    if dynamic:
      to_visit_values = to_visit_reverse_values = visited_values = None
    else:
      # We don't need to deepcopy here, because they will be copied on process
      # generation.
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
                         tapes=tapes,
                         online=online,
                         cum_hop_weights=cum_hop_weights,
                         verbose=verbose)

    with ProcessPoolExecutor() as p:
      repl_paths += p.map(_repl_path,
                          range(len(shortest_path) - 1),
                          shortest_path[:-1])

      def _is_path(path):
        if path and path[0]:
          return True
        return False

      repl_paths = list(filter(_is_path, repl_paths))
  else:  # not parallel
    for i, node in enumerate(shortest_path[:-1]):
      # The source cannot fail, but when failing == "edges", the source consti-
      # tudes the tail of the 1st edge that will fail.
      if (failing == "nodes") and (i == 0):
        continue

      if dynamic:
        if online:
          new_to_visit = copy.deepcopy(to_visit)
          new_visited = copy.deepcopy(visited)
          new_to_visit_reverse = None
        else:
          # All necessary data will be retrieved from tapes.
          new_to_visit = new_to_visit_reverse = new_visited = None
      else:
        new_to_visit = copy.deepcopy(to_visit)
        if bidirectional:
          new_to_visit_reverse = copy.deepcopy(to_visit_reverse)
        else:
          new_to_visit_reverse = None
        new_visited = copy.deepcopy(visited)
        tapes = None

      repl_path = _replacement_path(i,
                                    node,
                                    failing,
                                    shortest_path,
                                    adj_list,
                                    source,
                                    sink,
                                    new_to_visit,
                                    new_to_visit_reverse,
                                    new_visited,
                                    bidirectional,
                                    inverted_adj_list,
                                    tapes,
                                    online,
                                    cum_hop_weights,
                                    verbose)
      if repl_path[0]:
        repl_paths.append(repl_path)

  return repl_paths


def _yen(sink,
         adj_list,
         to_visit,
         visited,
         K,
         shortest_path,
         shortest_path_cost,
         cum_hop_weights,
         lawler=True):
  """Implementation of Yen's k-shortests paths algorithm with improvements.

  Improvements (see Brander-Sinclair 1996):
    - Not searching for deviation paths already found. (Lawler 1972)
    - Using a heap instead of list (Yen's B), to store candidate paths.
    - If at least K - k candidates with the same cost as the (k - 1)th path
      were already found, append them to the k_paths list (Yen's A) and return.
  """
  k_paths = [[shortest_path, shortest_path_cost, None]]
  # This is the B list of the Yen's algorithm, holding the potential shortest
  # paths.
  prospects = []
  heapq.heapify(prospects)
  last_path = shortest_path
  last_u_idx = 0

  for k in range(1, K):

    if not lawler:
      last_u_idx = 0

    # Construct the deviation paths of the last found shortest path.
    # (u is the spur node)
    for i, u in enumerate(last_path[last_u_idx: -1]):
      u_idx = i + last_u_idx
      # Fail the (i, i + 1) edges of all found shortest paths.
      # {head: (head, edge_cost)}
      failed_edges = dict()
      for j in k_paths:
        if j[0][:u_idx + 1] == last_path[:u_idx + 1]:
          failed_edges[j[0][u_idx + 1]] = None
      for v, uv_weight in adj_list[u]:
        if v in failed_edges.keys():
          failed_edges[v] = (v, uv_weight)
      for v, edge in failed_edges.items():
        adj_list[u].remove(edge)

      # Remove the Root path nodes from the to_visit PriorityQueue.
      new_to_visit = copy.deepcopy(to_visit)
      for root_node in last_path[:u_idx]:
        del new_to_visit[root_node]

      # Set i as source and initialize it's path cost to source-i path cost.
      new_to_visit[u] = [cum_hop_weights[u_idx], u, u]
      new_visited = dijkstra.dijkstra(adj_list,
                                      sink,
                                      new_to_visit,
                                      copy.deepcopy(visited))
      prospect_cost = new_visited[sink][0]
      spur, spur_hop_weights = dijkstra.extract_path(
        u,
        sink,
        new_visited,
        cum_hop_weights=True,
      )
      if spur:
        prospect = last_path[:u_idx] + spur
        prospect_hop_weights = cum_hop_weights[:u_idx] + spur_hop_weights

        if ((len(prospects) < K - k)
                or (prospect_cost
                    < heapq.nsmallest(K - k, prospects)[-1][0])):
          # Check if the prospect is already found
          prospect_already_found = False
          for p_cost, p, c, d in prospects:
            if (p_cost == prospect_cost) and (p == prospect):
              prospect_already_found = True
              break
          if not prospect_already_found:
            heapq.heappush(
              prospects,
              (prospect_cost, prospect, prospect_hop_weights, u_idx)
            )

      # Restore the failed edges.
      for v, edge in failed_edges.items():
        adj_list[u].add(edge)
      failed_edges.clear()

    # Add the best prospect to the k_paths list
    if prospects:
      # Check if at least K - k prospects with the same cost as the (k - 1)th
      # path were already found.
      if ((len(prospects) >= K - k)
              and heapq.nsmallest(K - k, prospects)[-1][0] == last_path[0]):
        for _ in range(K - k):
          last_path_cost, last_path, c, d = heapq.heappop(prospects)
          k_paths.append([last_path, last_path_cost, None])
        break
      last_path_cost, last_path, cum_hop_weights, last_u_idx = \
          heapq.heappop(prospects)
      k_paths.append([last_path, last_path_cost, None])
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
