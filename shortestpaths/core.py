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
from itertools import count
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
                         online=False,
                         record_only_cps=False,
                         verbose=0):
  tapes = None
  if bidirectional:
    # If dynamic, 2 recording sessions are executed. At the first, the absolute
    # shortest path will be generated, as well as the visited nodes sequence
    # will be recorded on a tape. At the second, for all the intermediate nodes
    # of the path, the state that corresponds to the immediately proceding vi-
    # sited node will be recorded on a tape, during both searches.
    if dynamic:
      path_data, tapes = dijkstra.bidirectional_recording(
        adj_list,
        inverted_adj_list,
        source,
        sink,
        to_visit,
        to_visit_reverse,
        visited,
        failing=failing,
        online=online,
        record_only_cps=record_only_cps,
        verbose=verbose
      )
    else:
      path_data = \
          dijkstra.bidirectional_dijkstra(adj_list,
                                          inverted_adj_list,
                                          source,
                                          sink,
                                          copy.deepcopy(to_visit),
                                          copy.deepcopy(to_visit_reverse),
                                          copy.deepcopy(visited),
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
      with_cum_hop_weights=online,
      verbose=verbose)

    if cum_hop_weights:
      path_data = [shortest_path, shortest_path_cost, cum_hop_weights]
    else:
      path_data = [shortest_path, shortest_path_cost, None]

  return path_data, tapes


# @time_this
def _replacement_path(failed_path_idx: int,
                      failed: Hashable,
                      failing: Literal["nodes", "edges"],
                      base_path: list,
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
                      k_paths: list = None,
                      verbose: int = 0) -> list:
  if (failing == "nodes") and (failed == source):
    return

  if failing == "nodes":
    if online:
      # Delete the nodes of the root path from the PriorityQueue.
      for u in base_path[:failed_path_idx - 1]:
        del to_visit[u]
      # The spur node becomes the source.
      source = base_path[failed_path_idx - 1]
      # Although the prev node doesn't have to be acurate, however it souldn't
      # be source, because when checking for bidirectional dijkstra termination
      # condition, it would not account as visited.
      to_visit[source] = [cum_hop_weights[failed_path_idx - 1], -1, source]
      # Initialize the path cost with the root_cost.
      if (bidirectional) and (not tapes):
        # Delete the nodes of the root path from the reverse PriorityQueue.
        for u in base_path[:failed_path_idx - 1]:
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
        online=online,
        base_path=base_path,
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
        with_cum_hop_weights=online,
        verbose=verbose
      )
  elif failing == "edges":
    # In case of dynamic k_shortest_paths, we need to record the checkpoints.
    record_cps = bool(k_paths) and bool(tapes)
    # When failing is nodes, the spur path will start *before* the failed node.
    # On the contrary, when failing is edges, the spur path will start *on* the
    # tail of the failed edge. So, in order to keep the same code for both in-
    # stances, failed_path_idx will be increased by one, when failing edges.
    tail = failed
    head = base_path[failed_path_idx + 1]

    if online:
      # Delete the nodes of the root path from the PriorityQueue.
      for u in base_path[:failed_path_idx]:
        del to_visit[u]
      # The spur node becomes the source.
      source = tail
      # Although the prev node doesn't have to be accurate, it shouldn't be the
      # source, because when checking for bidirectional dijkstra termination
      # condition, it would not account as visited.
      to_visit[source] = [cum_hop_weights[failed_path_idx], -1, source]
      # Initialize the path cost with the root_cost.
      if (bidirectional) and (not tapes):
        # Delete the nodes of the root path from the reverse PriorityQueue.
        for u in base_path[:failed_path_idx]:
          del to_visit_reverse[u]

    if k_paths:
      # Fail the (i, i + 1) edges of the found k - 1 shortest paths.
      # {head: (head, edge_cost)}
      failed_edges = dict()
      # {invertd_tail (head): (inverted_head (tail), edge_cost)}
      failed_inverted_edges = dict()
      for j in k_paths:
        if j[0][:failed_path_idx + 1] == base_path[:failed_path_idx + 1]:
          failed_edges[j[0][failed_path_idx + 1]] = None
          failed_inverted_edges[j[0][failed_path_idx + 1]] = None
      # Don't disconnect the failed edge yet, because it will be disconnected
      # in the subsequent loop.
      del failed_edges[head]
      del failed_inverted_edges[head]

      for v, uv_weight in adj_list[failed]:
        if v in failed_edges.keys():
          failed_edges[v] = (v, uv_weight)
          failed_inverted_edges[v] = (failed, uv_weight)
      for v, edge in failed_edges.items():
        adj_list[failed].remove(edge)

      if bidirectional:
        for u, edge in failed_inverted_edges.items():
          inverted_adj_list[u].remove(edge)

    # Fail the edge.
    for neighbor in adj_list[tail]:
      if neighbor[0] == head:
        adj_list[tail].discard(neighbor)
        # Find the replacement path.
        if bidirectional:
          # Fail the edge on the inverted_adj_list.
          for ne in inverted_adj_list[head]:
            if ne[0] == tail:
              inverted_adj_list[head].discard(ne)
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
                online=online,
                base_path=base_path,
                record_cps=record_cps,
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
            with_cum_hop_weights=online,
            verbose=verbose
          )
        # Reconnect the failed edge.
        adj_list[tail].add(neighbor)
        break

    # Reconnect the failed edges
    if k_paths:
      for v, edge in failed_edges.items():
        adj_list[failed].add(edge)
      if bidirectional:
        for u, edge in failed_inverted_edges.items():
          inverted_adj_list[u].add(edge)

    failed = (tail, head)
  else:
    raise ValueError(f"Unexpected value for failing: <{failing}>. It should"
                     " be either 'edges' or 'nodes'.")

  if k_paths:
    # then replacement_paths was called from k_shortest_paths
    # (online, failing edges always)
    if bidirectional:
      if tapes:
        path_data, checkpoints = path_data
      if path_data[0]:
        path_data = [
          base_path[: failed_path_idx] + path_data[0],
          path_data[1],
          cum_hop_weights[: failed_path_idx] + path_data[2],
          failed_path_idx
        ]
      else:
        path_data = [None, None, None]
      if tapes:
        path_data.append(checkpoints)
    else:
      if repl_path:
        path_data = [
          base_path[: failed_path_idx] + repl_path,
          repl_path_cost,
          cum_hop_weights[: failed_path_idx] + repl_weights,
          failed_path_idx
        ]
      else:
        path_data = [None, None, None]
  else:  # pure replacement-paths
    if online:
      if failing == "edges":
        failed_path_idx += 1

      # NOTE: The cost of the root path should be added here or used to initi-
      #       alize the source cost, where source is the spur node.
      if bidirectional:
        if path_data[0]:
          path_data = [base_path[: failed_path_idx - 1] + path_data[0],
                       path_data[1],
                       failed]
        else:
          path_data = [None, None, None]
      else:
        if repl_path:
          path_data = [base_path[: failed_path_idx - 1] + repl_path,
                       repl_path_cost,
                       failed]
        else:
          path_data = [None, None, None]
    elif not bidirectional:
      path_data = [repl_path, repl_path_cost, failed]

  return path_data


# @time_this
def replacement_paths(adj_list,
                      n,
                      source,
                      sink,
                      failing="edges",
                      bidirectional=False,
                      parallel=False,
                      dynamic=False,
                      online=False,
                      K=None,
                      k=None,
                      base_path=None,
                      parent_spur_node_idx=0,
                      k_paths=None,
                      prospects=None,
                      cum_hop_weights=None,
                      to_visit=None,
                      to_visit_reverse=None,
                      visited=None,
                      inverted_adj_list=None,
                      checkpoints=None,
                      verbose=0):
  """Wrapper that generates the replacement paths, using several different
  methods.

  Args:
    adj_list (list)      : [{(neighbor, edge_weight),},]
    n (int)              : number of nodes
    source (int)
    sink (int)
    failing (str)        : options: ["nodes", "edges"] (default: "nodes")
                           The element that fails to generate the replacement
                           paths.
    bidirectional (bool) : use the bidirectional Dijkstra's algorithm
                           (default False)
    parallel (bool)      : whether to parallelize the replacement paths search
    dynamic (bool)       : use dynamic programming (only works with the
                           bidirectional Dijkstra's algorithm) (default: False)
    online (bool)        : The on-line algorithm freezes the path until the
                           failed element. Namely, it discoveres the failure on
                           the go. On the contrary, the off-line algorithm has
                           a priory knowledge of the failure and, thus, it is
                           free to plan whatever path is best, avoiding the
                           failed element. (defalt: false)
    verbose (int)

  Returns:
    repl_paths (list)    : [[path_1, path_1_cost, failed],]
  """
  tapes = None
  if base_path:  # Then, this is called from k_shortest_path().
    repl_paths = []
    if dynamic:
      # Then, we need the reverse tape.
      # NOTE: The records start from the parent_spur_node_idx.
      _, tapes = dijkstra.bidirectional_recording(
        adj_list=None,
        inverted_adj_list=inverted_adj_list,
        source=source,
        sink=None,
        to_visit=None,
        to_visit_reverse=to_visit_reverse,
        visited=visited,
        checkpoints=checkpoints,
        failing="edges",
        online=True,
        verbose=verbose)

  else:  # pure replacement_paths
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
                                            online=online,
                                            verbose=verbose)
    if online:
      [base_path, base_path_cost, cum_hop_weights] = path_data
      repl_paths = [[base_path, base_path_cost, None]]
    else:
      repl_paths = [path_data]
      base_path = path_data[0]
      cum_hop_weights = None

  # Uncomment this to run a sanity test on the tapes.
  # dijkstra.verify_tapes(tapes, base_path, failing=failing)

  # Next, find the replacement paths.
  if dynamic:
    if online:
      # Only reverse states are recorded on tape.
      to_visit_reverse = None
    else:
      # All necessary data will be retrieved from tapes.
      to_visit = to_visit_reverse = visited = None
  else:
    # All data will be copied for each replacement path.
    tapes = None

  if parallel:
    _repl_path = partial(_replacement_path,
                         failing=failing,
                         base_path=base_path,
                         adj_list=adj_list,
                         source=source,
                         sink=sink,
                         to_visit=to_visit,
                         to_visit_reverse=to_visit_reverse,
                         visited=visited,
                         bidirectional=bidirectional,
                         inverted_adj_list=inverted_adj_list,
                         tapes=tapes,
                         online=online,
                         cum_hop_weights=cum_hop_weights,
                         k_paths=k_paths,
                         verbose=verbose)

    with ProcessPoolExecutor() as p:
      repl_paths += p.map(_repl_path,
                          range(parent_spur_node_idx, len(base_path) - 1),
                          base_path[parent_spur_node_idx: -1])
  else:
    for i, node in enumerate(base_path[parent_spur_node_idx: -1]):
      # The source cannot fail, but when failing == "edges", the source consti-
      # tudes the tail of the 1st failed edge.
      failed_path_idx = parent_spur_node_idx + i
      if (failing == "nodes") and (failed_path_idx == 0):
        continue
      repl_paths.append(_replacement_path(failed_path_idx,
                                          node,
                                          failing,
                                          base_path,
                                          adj_list,
                                          source,
                                          sink,
                                          copy.deepcopy(to_visit),
                                          copy.deepcopy(to_visit_reverse),
                                          copy.deepcopy(visited),
                                          bidirectional,
                                          inverted_adj_list,
                                          tapes,
                                          online,
                                          cum_hop_weights,
                                          k_paths,
                                          verbose))

  repl_paths = list(filter(lambda p: p and p[0], repl_paths))

  if k_paths:
    # Update the prospects heap, using the repl_paths found.
    for path in repl_paths:
      if dynamic:
        [pr_path, pr_cost, pr_hop_weights, pr_spur_node_idx, pr_cps] = path
      else:
        [pr_path, pr_cost, pr_hop_weights, pr_spur_node_idx] = path
      if ((len(prospects) < K - k)
              or (pr_cost < heapq.nsmallest(K - k, prospects)[-1][0])):
        # Check if the prospect is already found.
        prospect_already_found = False
        for p in prospects:
          if (p[0] == pr_cost) and (p[1] == pr_path):
            prospect_already_found = True
            break
        if not prospect_already_found:
          if dynamic:
            heapq.heappush(
              prospects,
              (pr_cost, pr_path, pr_hop_weights, pr_spur_node_idx, pr_cps)
            )
          else:
            heapq.heappush(
              prospects,
              (pr_cost, pr_path, pr_hop_weights, pr_spur_node_idx)
            )
    return prospects
  else:
    return repl_paths


def _yen(sink,
         adj_list,
         to_visit,
         visited,
         K,
         k,
         last_path,
         last_u_idx,
         k_paths,
         prospects,
         cum_hop_weights,
         lawler=False):
  """Implementation of Yen's k-shortests paths algorithm with improvements.

  Improvements (see Brander-Sinclair 1996):
    - Not searching for deviation paths already found. (Lawler 1972)
    - Using a heap instead of list (Yen's B), to store candidate paths.
    - If at least K - k candidates with the same cost as the (k - 1)th path
      were already found, append them to the k_paths list (Yen's A) and return.
  """
  if not lawler:
    last_u_idx = 0

  # Construct the deviation paths of the last found shortest path.
  # (u is the spur node)
  for i, u in enumerate(last_path[last_u_idx: -1]):
    next(_yen.counter)
    u_idx = i + last_u_idx
    # Fail the (i, i + 1) edges of the found k - 1 shortest paths.
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

    # Remove the root-path nodes from the to_visit PriorityQueue.
    new_to_visit = copy.deepcopy(to_visit)
    for root_node in last_path[:u_idx]:
      del new_to_visit[root_node]

    # Set the spur-node as source and initialize its cost to root-path-cost.
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
      with_cum_hop_weights=True,
    )
    if spur:
      prospect = last_path[:u_idx] + spur
      prospect_hop_weights = cum_hop_weights[:u_idx] + spur_hop_weights

      if ((len(prospects) < K - k)
              or (prospect_cost
                  < heapq.nsmallest(K - k, prospects)[-1][0])):
        # Check if the prospect is already found.
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
  return prospects


# @profile
@time_this
def k_shortest_paths(adj_list,
                     source,
                     sink,
                     K,
                     bidirectional=False,
                     parallel=False,
                     dynamic=False,
                     yen=False,
                     lawler=False,
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
  # NOTE: Always online and failing edges.
  [shortest_path, shortest_path_cost, cum_hop_weights], checkpoints = \
      _first_shortest_path(adj_list,
                           source,
                           sink,
                           to_visit,
                           to_visit_reverse,
                           visited,
                           inverted_adj_list,
                           bidirectional,
                           dynamic,
                           failing="edges",
                           online=True,
                           record_only_cps=True,
                           verbose=verbose)

  k_paths = [[shortest_path, shortest_path_cost, None]]
  # Holding the potential shortest paths (Yen's B).
  prospects = []
  heapq.heapify(prospects)
  last_path = shortest_path
  parent_spur_node_idx = 0
  _yen.counter = count(0)

  for k in range(1, K):
    if yen or lawler:
      prospects = _yen(sink,
                       adj_list,
                       to_visit,
                       visited,
                       K,
                       k,
                       last_path,
                       parent_spur_node_idx,
                       k_paths,
                       prospects,
                       cum_hop_weights,
                       lawler)
    if verbose >= 2:
      print(f"k: {k + 1:{len(str(K))}}"
            f"    spur paths: {_yen.counter.__reduce__()[1][0]}")
    else:
      prospects = replacement_paths(adj_list,
                                    n,
                                    source,
                                    sink,
                                    failing="edges",
                                    bidirectional=bidirectional,
                                    parallel=parallel,
                                    dynamic=dynamic,
                                    online=True,
                                    K=K,
                                    k=k,
                                    base_path=last_path,
                                    parent_spur_node_idx=parent_spur_node_idx,
                                    k_paths=k_paths,
                                    prospects=prospects,
                                    cum_hop_weights=cum_hop_weights,
                                    to_visit=to_visit,
                                    to_visit_reverse=to_visit_reverse,
                                    visited=visited,
                                    inverted_adj_list=inverted_adj_list,
                                    checkpoints=checkpoints,
                                    verbose=verbose)
    # Add the best prospect to the k_paths list
    if prospects:
      # Check if at least K - k prospects with the same cost as the (k - 1)th
      # path were already found.
      if ((len(prospects) >= K - k)
              and heapq.nsmallest(K - k, prospects)[-1][0] == last_path[0]):
        for _ in range(K - k):
          kth_path = heapq.heappop(prospects)
          k_paths.append([kth_path[1], kth_path[0], None])
        break
      kth_path = heapq.heappop(prospects)
      last_path = kth_path[1]
      k_paths.append([last_path, kth_path[0], None])
      cum_hop_weights = kth_path[2]
      parent_spur_node_idx = kth_path[3]
      if dynamic:
        checkpoints = kth_path[4]
    else:
      break
  return k_paths
