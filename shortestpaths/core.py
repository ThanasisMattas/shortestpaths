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

from shortestpaths import dijkstra, yen
from shortestpaths.priorityq import PriorityQueue
from shortestpaths.utils import print_heap, time_this  # noqa: F401


def _first_shortest_path(adj_list,
                         source,
                         sink,
                         to_visit,
                         to_visit_reverse,
                         visited,
                         inverted_adj_list=None,
                         bidirectional=False,
                         dynamic=False,
                         failing=None,
                         online=False,
                         verbose=0):
  if bidirectional:
    path_data = dijkstra.bidirectional_dijkstra(adj_list,
                                                inverted_adj_list,
                                                source,
                                                sink,
                                                to_visit,
                                                to_visit_reverse,
                                                visited,
                                                online=online,
                                                meeting_edge_head=True,
                                                verbose=verbose)
    if (dynamic) and (not online):
      # Save the states of the algorithm just before each path-node is visited.
      tapes = dijkstra.record_states(adj_list,
                                     inverted_adj_list,
                                     source,
                                     sink,
                                     to_visit,
                                     to_visit_reverse,
                                     visited,
                                     path_data[0],
                                     path_data[-1],
                                     failing=failing,
                                     verbose=verbose)
      # Uncomment this to run a sanity test on the tapes.
      # dijkstra.verify_tapes(tapes, path_data[0], failing=failing)
      return path_data, tapes
  else:
    visited_out = dijkstra.dijkstra(adj_list,
                                    sink,
                                    copy.deepcopy(to_visit),
                                    copy.deepcopy(visited))
    shortest_path_cost = visited_out[sink][0]

    shortest_path, cum_hop_weights = dijkstra.extract_path(
      source,
      sink,
      visited_out,
      with_cum_hop_weights=online,
      verbose=verbose)

    if online:
      # When online, we need the cumulative hop weights, to retrieve the root-
      # path cost up until the failed node/edge.
      path_data = [shortest_path, shortest_path_cost, cum_hop_weights]
    else:
      path_data = [shortest_path, shortest_path_cost, None]

  return path_data


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
                      visited_reverse: list = None,
                      discovered_reverse: set = None,
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
      if not visited_reverse:
        del to_visit[base_path[:failed_path_idx - 1]]
      # The spur node becomes the source.
      source = base_path[failed_path_idx - 1]
      # Although the prev node doesn't have to be acurate, however it souldn't
      # be source, because when checking for bidirectional dijkstra termination
      # condition, it would not account as visited.
      to_visit[source] = [cum_hop_weights[failed_path_idx - 1], -1, source]
      # Initialize the path cost with the root_cost.
      if (bidirectional) and (not tapes) and (not visited_reverse):
        # Delete the nodes of the root path from the reverse PriorityQueue.
        del to_visit_reverse[base_path[:failed_path_idx - 1]]

    if bidirectional:
      path_data = dijkstra.bidirectional_dijkstra(
        adj_list,
        inverted_adj_list,
        source,
        sink,
        to_visit,
        to_visit_reverse,
        visited,
        visited_reverse=visited_reverse,
        discovered_reverse=discovered_reverse,
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
    if k_paths:
      # Fail the (i, i + 1) edges of the found k - 1 shortest paths.
      failed_edges, failed_inverted_edges = yen.fail_found_spur_edges(
        adj_list,
        failed,
        failed_path_idx,
        base_path,
        k_paths,
        inverted_adj_list=inverted_adj_list,
        head=head
      )

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
                visited_reverse=visited_reverse,
                discovered_reverse=discovered_reverse,
                failed_path_idx=(failed_path_idx, failed_path_idx + 1),
                failed=(tail, head),
                tapes=tapes,
                online=online,
                base_path=base_path,
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
      yen.reconnect_spur_edges(failed,
                               adj_list,
                               failed_edges,
                               inverted_adj_list,
                               failed_inverted_edges)

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
  if base_path:  # Then, this is a k-shortest paths search.
    repl_paths = []
  else:  # pure replacement_paths
    to_visit, visited, to_visit_reverse, inverted_adj_list = \
        dijkstra.dijkstra_init(n,
                               source,
                               sink,
                               adj_list,
                               bidirectional)

    # Find the absolute shortest path.
    path_data = _first_shortest_path(adj_list,
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
      if bidirectional:
        [base_path, base_path_cost, cum_hop_weights, meeting_edge_head] = \
            path_data
      else:
        [base_path, base_path_cost, cum_hop_weights] = path_data
      repl_paths = [[base_path, base_path_cost, None]]
    else:
      if dynamic:
        path_data, tapes = path_data
      repl_paths = [path_data]
      base_path = path_data[0]
      cum_hop_weights = None

  # Next, find the replacement paths.
  if dynamic and (not online):
    # All necessary data will be retrieved from tapes.
    to_visit = to_visit_reverse = visited = None
  else:
    tapes = None

  if (parallel) and (not (dynamic and online)):
    # dynamic and online cannot run in parallel
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
  elif dynamic and online:
    meeting_edge_head_idx = base_path.index(meeting_edge_head) - 1
    visited_reverse = copy.deepcopy(visited)
    discovered_reverse = {sink}
    # Delete the entries of the path-nodes from the to_visit PriorityQueue.
    if failing == "nodes":
      del to_visit[base_path[:-2]]
      failed_path_idx = len(base_path) - 2
    else:
      del to_visit[base_path[:-1]]
      failed_path_idx = len(base_path) - 1

    while to_visit_reverse and failed_path_idx:

      if failed_path_idx > meeting_edge_head_idx:
        u_next = to_visit_reverse.peek()[-1]
      else:
        u_next = base_path[failed_path_idx]
      if u_next == base_path[failed_path_idx]:
        if failing == "nodes":
          failed = base_path[failed_path_idx]
          root_path = base_path[:failed_path_idx - 1]
          failed_idx = failed_path_idx
        else:
          # In case of failing edges, failed is the tail of the failed_node.
          failed = base_path[failed_path_idx - 1]
          root_path = base_path[:failed_path_idx]
          failed_idx = failed_path_idx - 1
        # Disconnect root-path-nodes.
        root_path_to_visit_entries = []
        discovered_root_nodes = set()
        for u_root in root_path:
          if u_root in to_visit_reverse:
            root_path_to_visit_entries.append(to_visit_reverse[u_root])
            del to_visit_reverse[u_root]
          if u_root in discovered_reverse:
            discovered_reverse.remove(u_root)
            discovered_root_nodes.add(u_root)
        repl_paths.append(_replacement_path(failed_idx,
                                            failed,
                                            failing,
                                            base_path,
                                            adj_list,
                                            source,
                                            sink,
                                            to_visit,
                                            to_visit_reverse,
                                            visited,
                                            visited_reverse,
                                            discovered_reverse,
                                            True,
                                            inverted_adj_list,
                                            None,
                                            True,
                                            cum_hop_weights,
                                            k_paths,
                                            verbose))
        # Reconnect failed root-path-nodes.
        for entry in root_path_to_visit_entries:
          to_visit_reverse[entry[-1]] = entry
        discovered_reverse.update(discovered_root_nodes)
        discovered_root_nodes.clear()
        root_path_to_visit_entries.clear()

        if failing == "nodes":
          spur_node = base_path[failed_path_idx - 1]
        else:
          spur_node = failed
        to_visit[spur_node] = [math.inf, spur_node, spur_node]
        failed_path_idx -= 1

      if failed_path_idx <= meeting_edge_head_idx:
        # Carry on using current node as source for the reverse search, because
        # the search sphere of the underlying single-direcional search is very
        # big and at this point the reverse search will always start from here.
        continue

      u_path_cost, u_prev, u = to_visit_reverse.pop_low()
      discovered_reverse.discard(u_next)

      if u_path_cost == math.inf:
        visited_reverse[u][0] = -1
        continue
      visited_reverse[u][0] = u_path_cost
      visited_reverse[u][1] = u_prev

      if u == source:
        break

      for v, uv_weight in inverted_adj_list[u]:
        if v in to_visit_reverse:
          dijkstra.relax_path_cost(v,
                                   u,
                                   uv_weight,
                                   u_path_cost,
                                   to_visit_reverse)
          discovered_reverse.add(v)
    if not k_paths:
      repl_paths = [repl_paths[0]] + list(reversed(repl_paths[1:]))
  else:
    for i, node in enumerate(base_path[parent_spur_node_idx: -1]):
      # The source cannot fail, but when failing == "edges", the source con-
      # stitutes the tail of the 1st failed edge.
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
                                          None,
                                          None,
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
      # [pr_path, pr_cost, pr_hop_weights, pr_spur_node_idx] = path
      yen.push_prospect(*path, K, k, prospects)
    return prospects
  else:
    return repl_paths


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
  to_visit, visited, to_visit_reverse, inverted_adj_list = \
      dijkstra.dijkstra_init(n,
                             source,
                             sink,
                             adj_list,
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
  parent_spur_node_idx = 1

  for k in range(1, K):
    if yen or lawler:
      prospects = yen.update_prospects(sink,
                                       adj_list,
                                       copy.deepcopy(to_visit),
                                       copy.deepcopy(visited),
                                       K,
                                       k,
                                       last_path,
                                       parent_spur_node_idx,
                                       k_paths,
                                       prospects,
                                       cum_hop_weights,
                                       lawler,
                                       verbose)
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
                                    verbose=verbose)
    # Add the best prospect to the k_paths list
    if prospects:
      cum_hop_weights, parent_spur_node_idx = yen.push_kth_path(
        prospects,
        K,
        k,
        last_path,
        k_paths
      )
      if cum_hop_weights is None:
        # Then, all k_paths were found.
        break
    else:
      break
  return k_paths
