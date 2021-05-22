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
  tapes = None
  if bidirectional:
    path_data = dijkstra.bidirectional_dijkstra(adj_list,
                                                inverted_adj_list,
                                                source,
                                                sink,
                                                to_visit,
                                                to_visit_reverse,
                                                visited,
                                                online=online,
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
  else:
    path_data = dijkstra.unidirectional_dijkstra(
      adj_list,
      sink,
      copy.deepcopy(to_visit),
      copy.deepcopy(visited),
      failed=None,
      with_cum_hop_weights=online,
      verbose=verbose
    )

  # Insert None for failed node/edge, to forn the _replacement_path() return
  # format.
  path_data = path_data[:3] + (None, path_data[3])
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
                      visited_reverse: list = None,
                      discovered_reverse: set = None,
                      bidirectional: bool = False,
                      inverted_adj_list: list = None,
                      tapes: list = None,
                      online: bool = False,
                      cum_hop_weights: list = None,
                      k_paths: list = None,
                      verbose: int = 0) -> list:
  if failing == "nodes":
    if failed == source:
      return
  else:
    tail = failed
    head = base_path[failed_path_idx + 1]
    failed = (tail, head)

  if online:
    # NOTE: When failing nodes, the spur path will start *before* <failed>,
    # which is the failed node. On the contrary, when failing edges the spur
    # path will start *on* <failed>, which is the tail of the failed edge.
    if failing == "nodes":
      spur_node_idx = failed_path_idx - 1
    else:
      spur_node_idx = failed_path_idx

    if not visited_reverse:
      # visited_reverse is passed only when dynamic and online, where the
      # PriorityQueue is not deepcopied and, thus, the root-path-nodes are
      # reconnected after the search.
      # Delete the root-path-nodes from the PriorityQueue.
      del to_visit[base_path[:spur_node_idx]]
    # The spur node becomes the source.
    source = base_path[spur_node_idx]
    # Although the prev node doesn't have to be accurate, it shouldn't be the
    # source, because when checking for bidirectional Dijkstra's algorithm
    # termination condition, it would not account as visited.
    # Initialize the path cost with the root_cost.
    to_visit[source] = [cum_hop_weights[spur_node_idx], -1, source]
    if (bidirectional) and (not tapes) and (not visited_reverse):
      # Delete the root-path-nodes from the reverse PriorityQueue.
      del to_visit_reverse[base_path[:spur_node_idx]]

  if failing == "nodes":
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
      path_data = dijkstra.unidirectional_dijkstra(
        adj_list,
        sink,
        to_visit,
        visited,
        failed,
        with_cum_hop_weights=online,
        verbose=verbose
      )
  else:  # failing == "edges"
    if k_paths:
      # Fail the (i, i + 1) edges of the found k - 1 shortest paths.
      failed_edges, failed_inverted_edges = yen.fail_found_spur_edges(
        adj_list,
        tail,
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
                failed=failed,
                tapes=tapes,
                online=online,
                base_path=base_path,
                verbose=verbose
              )
              # Reconnect the failed edge.
              inverted_adj_list[head].add(ne)
              break
        else:
          path_data = dijkstra.unidirectional_dijkstra(
            adj_list,
            sink,
            to_visit,
            visited,
            failed=None,
            with_cum_hop_weights=online,
            verbose=verbose
          )
        # Reconnect the failed edge.
        adj_list[tail].add(neighbor)
        break
    if k_paths:
      # Reconnect the failed edges
      yen.reconnect_spur_edges(tail,
                               adj_list,
                               failed_edges,
                               inverted_adj_list,
                               failed_inverted_edges)
  repl_path, repl_path_cost, repl_weights, meeting_edge_head = path_data
  # NOTE: The cost of the root path should be added here or used to initialize
  #       the source cost, where the spur node becomes the source (currently,
  #       we are doing the second).
  if repl_path:
    if online:
      repl_path = base_path[:spur_node_idx] + repl_path
      repl_weights = cum_hop_weights[:spur_node_idx] + repl_weights
    # if tapes or visited_reverse:  # offline or online dynamic
    if k_paths:
      # We pass the spur-node-idx with <failed>.
      failed = failed_path_idx
  return repl_path, repl_path_cost, repl_weights, failed, meeting_edge_head


def _dynamic_online_replacement_paths(adj_list,
                                      source,
                                      sink,
                                      failing,
                                      base_path,
                                      parent_spur_node_idx=1,
                                      meeting_edge_head=None,
                                      k_paths=None,
                                      repl_paths=None,
                                      cum_hop_weights=None,
                                      to_visit=None,
                                      to_visit_reverse=None,
                                      visited=None,
                                      inverted_adj_list=None,
                                      verbose=0):
  """Knowing the base_path in advance, generates the replacement-paths, using
  dynamic programming.

  It runs a reverse seach and, every time it is about to visit a path-node, it
  uses the underlying state as the reverse state of a bidirectional spur-path
  search. This way, each search of a spur-path (replacement-path) doesn't start
  from the begining (sink). This carries on until the meeting point of the bi-
  directional search that found the base_path, because from that point onwards
  the search sphere is getting too big and the underlying reverse serach too
  expensive. So, each subsequent spur-path search will get this last state and
  the extra search work will be handled bidirectionally.

  NOTE: - failed node is not deleted, but avoided
        - failed edge is deleted from both adjacency lists and reconnected back
        - spur-edges that are included to previous found k-1 paths with the
          same root-path as the potential path being probed are deleted from
          both adjacency lists and reconnected back
        - root-path-nodes are deleted from the PriorityQueue and reconnected
          back
  """
  # meeting_edge_head of the base_path is the point where we stop updating
  # the reverse search state, using that final state for all the subsequent
  # searches.
  last_state_path_idx = base_path.index(meeting_edge_head) - 1
  visited_reverse = copy.deepcopy(visited)
  discovered_reverse = {sink}

  # Delete the path-nodes of the 1st root-path from the to_visit PriorityQueue.
  # The 1st root-path is formed by failing the second to last node or the last
  # edge. Note that the spur-node, which is the source of the spur-path, can be
  # deleted here, too, because it will be set downstream. After the completion
  # of each spur-path search, the spur-node will be restored, to form the next
  # root-path.
  if failing == "nodes":
    del to_visit[base_path[:-2]]
    failed_path_idx = len(base_path) - 2
  else:
    del to_visit[base_path[:-1]]
    # failed_path_idx is the head of each failed edge.
    failed_path_idx = len(base_path) - 1

  # Terminate the spur-paths search at the spur_node of the parent-path, as
  # suggested by Lawler.
  while to_visit_reverse and (failed_path_idx > parent_spur_node_idx):
    # The reverse search state will keep updating until the meeting point of
    # the bidirectional search of the parent_path, because from that point on
    # the unidirectional search for updating the state becomes very expensive,
    # since the search space becomes large. Therefore, from that point on the
    # reverse state of each subsequent bidirectional spur-path search will get
    # this last state.
    if failed_path_idx > last_state_path_idx:
      # print(failed_path_idx)
      u_next = to_visit_reverse.peek()[-1]
    else:
      u_next = base_path[failed_path_idx]

    if u_next == base_path[failed_path_idx]:
      # Before popping the next node, we have the reverse state to start a bi-
      # directional spur-path search.
      if failing == "nodes":
        failed = base_path[failed_path_idx]
        root_path = base_path[:failed_path_idx - 1]
        spur_node = base_path[failed_path_idx - 1]
        failed_idx = failed_path_idx
      else:
        # In case of failing edges, we pass as failed the tail of the failed
        # edge.
        failed = base_path[failed_path_idx - 1]
        root_path = base_path[:failed_path_idx]
        spur_node = failed
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

      # Bidirectional spur-path search
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

      # The only item that changed from the to_visit PriorityQueue before it
      # was deepcopied on multiporcessing was the spur node, so here we restore
      # its value. Also, this way we form the root-path for the next search.
      to_visit[spur_node] = [math.inf, spur_node, spur_node]
      failed_path_idx -= 1

    if failed_path_idx <= last_state_path_idx:
      # Carry on using current node as source for the reverse search, because
      # the search sphere of the underlying unidirecional search got very big
      # and building more informed states became very heavy. The extra distance
      # will be handled bidirectionally at each spur-path search.
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
  return repl_paths


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
                      meeting_edge_head=None,
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
  repl_paths = []

  if not base_path:
    to_visit, visited, to_visit_reverse, inverted_adj_list = \
        dijkstra.dijkstra_init(n,
                               source,
                               sink,
                               adj_list,
                               bidirectional)

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
    repl_paths.append(path_data)
    base_path = path_data[0]
    cum_hop_weights = path_data[2]
    meeting_edge_head = path_data[4]

  # Next, find the replacement paths.
  if dynamic and (not online):
    # All necessary data will be retrieved from tapes.
    to_visit = to_visit_reverse = visited = None
  else:
    tapes = None

  if (parallel) and (not (dynamic and online)):
    # NOTE: dynamic and online is the only mode that cannot run in parallel.
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
    repl_paths = _dynamic_online_replacement_paths(
      adj_list,
      source,
      sink,
      failing,
      base_path,
      parent_spur_node_idx,
      meeting_edge_head,
      k_paths,
      repl_paths,
      cum_hop_weights,
      to_visit,
      to_visit_reverse,
      visited,
      inverted_adj_list,
      verbose)
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
      yen.push_prospect(path[0],
                        path[1],
                        path[2],
                        failed_path_idx,
                        K,
                        k,
                        prospects)
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
                     yen_=False,
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
  path_data, _ = _first_shortest_path(adj_list,
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
                                      verbose=verbose)

  k_paths = [path_data]
  last_path = path_data[0]
  cum_hop_weights = path_data[2]
  meeting_edge_head = path_data[4]
  # Holding the potential shortest paths (Yen's B).
  prospects = []
  heapq.heapify(prospects)
  parent_spur_node_idx = 0

  for k in range(1, K):
    if yen_ or lawler:
      prospects = yen.update_prospects(sink,
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
                                    meeting_edge_head=meeting_edge_head,
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
