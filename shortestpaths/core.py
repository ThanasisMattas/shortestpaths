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
"""Bidirectional, parallel and dynamic programming algorithms for the
replacement paths and the k-shortest paths problems.

Replacement-paths versions implemented:
  - edge-exclusion
  - node-exclusion
  - on-line
  - off-line

Yen's algorithm is implemented as a reference to compare with.
"""

from concurrent.futures import ProcessPoolExecutor
import copy
from functools import partial
import heapq
import math
from typing import Hashable

from shortestpaths import dijkstra, yen
from shortestpaths.utils import print_heap, time_this  # noqa: F401


def first_shortest_path(mode, init_config):
  """Generates the 1st shortest path, initializing the replacement-paths or the
  k-shortest paths search.

  When using dynamic programming on offline mode, tapes holding the algorithm's
  states are recorded.

  Args:
    mode (dict)        : the configuration of the problem
    init_config (dict) : kwargs for dijkstra_init()

  Returns:
    path_data (list)   : [
                           path,
                           path_cost,
                           cum_hop_weights,
                           None,  # failed node/edge
                           meeting_edge_head
                         ]
    tapes (tuple)      : The recorded states used for dynamic programming.
                         Only when (mode["dynamic"]) and (not mode["online"]).
                         (
                           [
                             to_visit,
                             visited,
                             discovered_forward
                           ],
                           [
                             to_visit_reverse,
                              visited_reverse,
                              discovered_reverse
                           ]
                         )
                         (default: None)
  """
  tapes = None
  forward_config, reverse_config = dijkstra.dijkstra_init(**init_config)
  if mode["bidirectional"]:
    dynamic = mode["dynamic"]
    mode["dynamic"] = False
    path_data = dijkstra.bidirectional_dijkstra(forward_config,
                                                reverse_config,
                                                mode)
    mode["dynamic"] = dynamic
    if (mode["dynamic"]) and (not mode["online"]):
      # Save the states of the algorithm just before each path-node is visited.
      tapes = dijkstra.record_states(mode,
                                     init_config,
                                     path_data[0],
                                     path_data[3])
      # Uncomment this to run a sanity test on the tapes.
      # dijkstra.verify_tapes(tapes, path_data[0], failing=mode["failing"])
  else:
    path_data = dijkstra.unidirectional_dijkstra(forward_config, mode)

  # Insert None for <failed>, to forn the _replacement_path() return format.
  path_data = path_data[:3] + (None, path_data[3])
  return path_data, tapes


def _delete_root_path_nodes(mode,
                            root_path,
                            to_visit,
                            to_visit_reverse=None):
  """Deletes the root-path-nodes from the PriorityQueue, n case of the *online*
  replacement paths.

  The PriorityQueue's are mutated.

  Args:
    mode (dict)                      : the configuration of the problem
    root_path (list)                 : without the spur-node
    to_visit (PriorityQueue)         : the forward PriorityQueue
    to_visit_reverse (PriorityQueue) : the reverse PriorityQueue (dufalt: None)
  """
  if not mode["online"]:
    raise Exception("Trying to delete root-path-nodes in offline mode.")
  del to_visit[root_path]
  if mode["bidirectional"]:
    del to_visit_reverse[root_path]


# @profile
# @time_this
def _replacement_path(failed_path_idx: int,
                      failed: Hashable,
                      base_path: list,
                      mode: dict,
                      init_config: dict,
                      forward_config: dict = None,
                      reverse_config: dict = None,
                      cum_hop_weights: list = None,
                      tapes: list = None,
                      k_paths: list = None,
                      discovered_reverse: set = None) -> list:
  """Generates one replacement path, failing one node or edge.

  When using dynamic programming in off-line mode, the *tapes* will hold the
  memoized states of both searches. The function retrieves some state before
  the failed node for both directions. In case of failing edges instead of no-
  des, the forward search retrieves the state of a node visited before the tail
  and the reverse search retrieves the state of a node visited before the head.

  An insight when retrieving a state, is that the algorithm completely ignores
  the path, as Dijkstra's algorithm would do while solving.

  When solving for the k-shortest paths, the 3rd return value is the
  failed_path_idx instead of failed, being the parent_spur_node_idx for the
  forthcoming replacement paths of this path.

    - failed node   :  *
      forward state :  .
      reverse state :  o
      both visited  : .o

           .     .   .o  o
              .   .       o     o
        .   .  .     *   o  o
           .     .o     o     o
          .    .   .     .o

    - failed edge : *---*

           .     .   .o  o
              .   .       o     o
        .   .  .     *---*  o
           .     .o     o     o
          .    .   .     .o

  Args:
    failed_path_idx (int)    : the index of the failed node (or the tail of the
                               failed edge) on the base-path
    failed (int | tuple)     : the failed node or edge
    base_path (list)         : the parent-path
    mode (dict)              : the configuration of the problem
    init_config (dict)       : kwargs for dijkstra_init()
    forward_config (dict)    : forward search kwargs (default: None)
    reverse_config (dict)    : reverse search kwargs (default: None)
    cum_hop_weights (lsit)   : the cumulative hop weights of the base-path
    tapes (tuple)            : the algorithm states - when offline and using
                               dynamic programming (default: None)
    k_paths (list)           : the k-shortest paths holder (default: None)
    discovered_reverse (set) : discovered but not visited nodes of the reverse
                               search (default: None)

  Returns:
    repl_path, repl_path_cost, repl_weights, failed, meeting_edge_head
  """
  if mode["failing"] == "nodes":
    if failed == init_config["source"]:
      return
  else:
    tail = failed
    head = base_path[failed_path_idx + 1]
    failed = (tail, head)

  if mode["bidirectional"]:
    # These will be populated when dynamic & offline.
    prospect = top_r = None

  if mode["dynamic"]:
    if mode["online"]:
      discovered_forward = set()
    else:
      forward_config, discovered_forward = dijkstra.retrieve_state(
          "forward",
          tapes[0],
          init_config,
          base_path,
          failed_path_idx,
          mode["failing"]
      )
      reverse_config, discovered_reverse = dijkstra.retrieve_state(
          "reverse",
          tapes[1],
          init_config,
          base_path,
          failed_path_idx + int(mode["failing"] == "edges"),
          mode["failing"]
      )
      if mode["failing"] == "edges":
        # When source and sink are the tail of a failed edge, we don't have a
        # state before them to recover from; therefore, the corresponding re-
        # corded states are the ones visiting the source and the sink and we
        # just have to un-discover head of the failed edge.
        # NOTE: When online, the source is not base_path[0].
        if failed[0] == base_path[0]:
          # then un-discover the head
          discovered_forward.discard(failed[1])
          forward_config["to_visit"][failed[1]] = \
              [math.inf, failed[1], failed[1]]
        elif failed[1] == init_config["sink"]:
          # then un-discover the tail
          discovered_reverse.discard(failed[0])
          reverse_config["to_visit"][failed[0]] = \
              [math.inf, failed[0], failed[0]]

      # Retrieve the prospect path of the state.
      # prospect: [path_cost, meeting_edge_tail, meeting_edge_head, edge_weight]
      prospect = dijkstra.prospect_init(forward_config["to_visit"],
                                        reverse_config["to_visit"],
                                        forward_config["visited"],
                                        reverse_config["visited"],
                                        discovered_forward,
                                        discovered_reverse,
                                        root_path=None)
      # Check if termination condition is already met.
      top_f = forward_config["to_visit"].peek()[0]
      if top_f == math.inf:
        top_f = 0
      top_r = reverse_config["to_visit"].peek()[0]
      if top_r == math.inf:
        top_r = 0

      if (top_f + top_r >= prospect[0]) and (sum(prospect) != 0):
        repl_path_cost = prospect[0]
        repl_path, repl_weights = dijkstra.extract_bidirectional_path(
            init_config["source"],
            init_config["sink"],
            prospect,
            mode,
            visited=forward_config["visited"],
            visited_reverse=reverse_config["visited"]
        )
        return repl_path, repl_path_cost, repl_weights, failed, prospect[2]
  else:
    forward_config, reverse_config = dijkstra.dijkstra_init(**init_config)

  if mode["online"]:
    # NOTE: When failing nodes, the spur will start *before* <failed>, which is
    # the failed node. On the contrary, when failing edges, the spur will start
    # *on* <failed>, which is the tail of the failed edge.
    spur_node_idx = failed_path_idx - int(mode["failing"] == "nodes")
    root_path = base_path[:spur_node_idx]
    if spur_node_idx > 0:
      _delete_root_path_nodes(mode,
                              root_path,
                              forward_config.get("to_visit"),
                              reverse_config.get("to_visit"))
    # The spur node becomes the source.
    source = base_path[spur_node_idx]
    # NOTE: prev_node of source is -1
    forward_config["to_visit"][source] = \
        [cum_hop_weights[spur_node_idx], -1, source]
    if mode["bidirectional"]:
      reverse_config["sink"] = source

  if mode["failing"] == "nodes":
    if mode["bidirectional"]:
      path_data = dijkstra.bidirectional_dijkstra(forward_config,
                                                  reverse_config,
                                                  mode,
                                                  failed,
                                                  prospect,
                                                  top_r)
    else:
      path_data = dijkstra.unidirectional_dijkstra(forward_config,
                                                   mode,
                                                   failed)
  else:
    if k_paths:
      # Fail the (i, i + 1) edges of the found k - 1 shortest paths.
      failed_edges, failed_inverted_edges = yen.fail_found_spur_edges(
          forward_config["adj_list"],
          tail,
          failed_path_idx,
          base_path,
          k_paths,
          adj_list_reverse=reverse_config.get("adj_list"),
          head=head
      )

    # Fail the edge.
    for neighbor in forward_config["adj_list"][tail]:
      if neighbor[0] == head:
        forward_config["adj_list"][tail].remove(neighbor)
        # Find the replacement path.
        if mode["bidirectional"]:
          # Fail the edge on the adj_list_reverse.
          for ne in reverse_config["adj_list"][head]:
            if ne[0] == tail:
              reverse_config["adj_list"][head].remove(ne)
              path_data = dijkstra.bidirectional_dijkstra(forward_config,
                                                          reverse_config,
                                                          mode,
                                                          None,
                                                          prospect,
                                                          top_r)
              # Reconnect the failed edge.
              reverse_config["adj_list"][head].add(ne)
              break
        else:
          path_data = dijkstra.unidirectional_dijkstra(forward_config, mode)
        # Reconnect the failed edge.
        forward_config["adj_list"][tail].add(neighbor)
        break
    if k_paths:
      # Reconnect the failed edges
      yen.reconnect_spur_edges(tail,
                               forward_config["adj_list"],
                               failed_edges,
                               reverse_config.get("adj_list"),
                               failed_inverted_edges)
  repl_path, repl_path_cost, repl_weights, meeting_edge_head = path_data
  # NOTE: The cost of the root path should be added here or used to initialize
  #       the source cost, where the spur node becomes the source (currently,
  #       we are doing the second).
  if repl_path:
    if mode["online"]:
      repl_path = base_path[:spur_node_idx] + repl_path
      repl_weights = cum_hop_weights[:spur_node_idx] + repl_weights
    if k_paths:
      # We pass the spur-node-idx with <failed>.
      failed = failed_path_idx
  return repl_path, repl_path_cost, repl_weights, failed, meeting_edge_head


def _dijktra_step(to_visit, visited, sink, adj_list):
  """Expands one node.

  Note that there is no check for reaching the sink, because it shouldn't. The
  execution should end at the parent_spur_node (according to Lawler's
  modification) or at the meeting_edge.
  """
  u_path_cost, u_prev, u = to_visit.pop_low()

  if u_path_cost == math.inf:
    visited[u][0] = -1
    return
  visited[u][0] = u_path_cost
  visited[u][1] = u_prev

  for v, uv_weight in adj_list[u]:
    if v in to_visit:
      to_visit.relax_priority([u_path_cost + uv_weight, u, v])


# @profile
def _dynamic_online_replacement_paths(mode,
                                      init_config,
                                      base_path,
                                      cum_hop_weights,
                                      parent_spur_node_idx,
                                      meeting_edge_head,
                                      k_paths=None):
  """Generates the online replacement-paths, using dynamic programming.

  It runs a reverse seach and, every time it is about to visit a path-node, it
  uses the underlying state as the reverse state of a bidirectional spur
  search. This way, each search of a spur (replacement-path) doesn't start
  from the begining (sink). This carries on until the meeting point of the bi-
  directional search that found the base_path, because from that point onwards
  the search sphere is getting too big and the underlying reverse serach too
  expensive. So, each subsequent spur search will get this last state and
  the extra search work will be handled bidirectionally.

  NOTE: - failed node is not deleted, but avoided
        - failed edge is deleted from both adjacency lists and reconnected back
        - spur-edges that are included to previous found k-1 paths with the
          same root-path as the potential path being probed are deleted from
          both adjacency lists and reconnected back
        - root-path-nodes are deleted from both PriorityQueue's

  Args:
    mode (dict)                : the configuration of the problem
    init_config (dict)         : kwargs for dijkstra_init()
    base_path (list)           : the parent-path
    cum_hop_weights (list)     : the cumulative hop weights of the base-path
    parent_spur_node_idx (int) : used as a starting point of spur-paths search,
                                 as suggested by Lawler
    meeting_edge_head (int)    : used as the last point of updating the reverse
                                 search state
    k_paths (list)             : the k-shortest paths holder (default: None)

  Returns:
    repl_paths (list)          : see _replacement_path()
  """
  repl_paths = []
  adj_list = init_config["adj_list"]
  adj_list_reverse = init_config["adj_list_reverse"]
  init_config["adj_list"] = None
  _, reverse_config = dijkstra.dijkstra_init(**init_config)
  init_config["adj_list"] = adj_list
  init_config["adj_list_reverse"] = None
  # The reverse search state stops being updated at meeting_edge_head. That
  # final state is used for all the subsequent searches.
  last_state_path_idx = base_path.index(meeting_edge_head) - 1

  # Initialize the first failed node or edge, which are the second to last node
  # or the last edge.
  if mode["failing"] == "nodes":
    failed_path_idx = len(base_path) - 2
  else:
    # Here, failed_path_idx is the head of each failed edge, because when we
    # stumble upon a path-node, this will be the head of the failed edge. So,
    # basically we loop through heads, when scanning backwards. Note that
    # _replacement_path() takes the tail of the failed edge as failed_path_idx.
    failed_path_idx = len(base_path) - 1

  # Terminate the spurs search at the spur_node of the parent-path, as sugge-
  # sted by Lawler.
  while failed_path_idx > parent_spur_node_idx:
    # The reverse search state will keep updating until the meeting point of
    # the bidirectional search of the parent_path, because from that point on
    # the unidirectional search for updating the state becomes very expensive,
    # since the search space becomes large. Therefore, from that point on the
    # reverse state of each subsequent bidirectional spur search will get this
    # last state.
    if failed_path_idx > last_state_path_idx:
      u_next = reverse_config["to_visit"].peek()[-1]
    else:
      u_next = base_path[failed_path_idx]

    if u_next == base_path[failed_path_idx]:
      # Before popping the next node, we have the reverse state to start a bi-
      # directional spur search.
      if mode["failing"] == "nodes":
        failed = base_path[failed_path_idx]
        # root_path = base_path[:failed_path_idx - 1]
        # spur_node = base_path[failed_path_idx - 1]
        failed_idx = failed_path_idx
      else:
        # When failing edges, <failed> is the tail of the failed edge, but
        # currently we are looping through heads.
        failed = base_path[failed_path_idx - 1]
        # root_path = base_path[:failed_path_idx]
        # spur_node = failed
        failed_idx = failed_path_idx - 1

      # Bidirectional spur search
      forward_config, _ = dijkstra.dijkstra_init(**init_config)
      reverse_config_copy = {
        "adj_list": adj_list_reverse,
        "sink": reverse_config["sink"],
        "to_visit": copy.deepcopy(reverse_config["to_visit"]),
        "visited": [entry.copy() for entry in reverse_config["visited"]]
      }
      repl_paths.append(_replacement_path(failed_idx,
                                          failed,
                                          base_path,
                                          mode,
                                          init_config,
                                          forward_config,
                                          reverse_config_copy,
                                          cum_hop_weights,
                                          None,
                                          k_paths))

      failed_path_idx -= 1

    if failed_path_idx > last_state_path_idx:
      # Update reverse search state.
      _dijktra_step(**reverse_config)

  if k_paths:
    for path_node in base_path[:-1]:
      forward_config["to_visit"][path_node] = [math.inf, path_node, path_node]
    init_config["adj_list_reverse"] = adj_list_reverse
  else:
    repl_paths = list(reversed(repl_paths))
  return repl_paths


# @profile
# @time_this
def replacement_paths(mode,
                      init_config,
                      path_data=None,
                      k_paths=None):
  """Wrapper that generates the replacement paths, using several different
  methods.

  Args:
    mode (dict)        : the configuration of the problem
    init_config (dict) : kwargs for dijkstra_init()
    path_data (list)   : see _replacement_path() (default: None)
    k_paths (list)     : when used with k-paths search (default: None)

  Returns:
    repl_paths (list)  : [
                           (
                             path,
                             path_cost,
                             cum_hop_weights,
                             failed,  # failed_path_idx when k-shortest paths
                             meeting_edge_head
                           ),
                         ]
  """
  # 1. Find/retrieve the absolute shortest path.
  if path_data:  # then, replacement_paths() was called from k_shortest_paths()
    tapes = None
    repl_paths = []
    parent_spur_node_idx = path_data[3]
  else:
    path_data, tapes = first_shortest_path(mode, init_config)
    repl_paths = [path_data]
    parent_spur_node_idx = 0
  base_path = path_data[0]
  cum_hop_weights = path_data[2]

  # 2. Find the replacement paths.
  if mode.get("dynamic") and mode.get("online"):
    repl_paths += _dynamic_online_replacement_paths(
        mode,
        init_config,
        base_path,             # path_data[0]
        cum_hop_weights,       # path_data[2]
        parent_spur_node_idx,  # path_data[3]
        path_data[4],          # meeting_edge_head
        k_paths
    )
  elif mode.pop("parallel", False):
    _repl_path = partial(_replacement_path,
                         base_path=base_path,
                         mode=mode,
                         init_config=init_config,
                         cum_hop_weights=cum_hop_weights,
                         tapes=tapes,
                         k_paths=k_paths,
                         discovered_reverse=None)
    with ProcessPoolExecutor() as p:
      repl_paths += p.map(_repl_path,
                          range(parent_spur_node_idx, len(base_path) - 1),
                          base_path[parent_spur_node_idx: -1])
  else:
    for i, failed in enumerate(base_path[parent_spur_node_idx: -1]):
      failed_path_idx = parent_spur_node_idx + i
      repl_paths.append(_replacement_path(failed_path_idx,
                                          failed,
                                          base_path,
                                          mode,
                                          init_config,
                                          cum_hop_weights=cum_hop_weights,
                                          tapes=tapes,
                                          k_paths=k_paths))

  repl_paths = list(filter(lambda p: p and p[0], repl_paths))
  return repl_paths


# @profile
@time_this
def k_shortest_paths(K, mode, init_config):
  """Generates k_shortest_paths

  NOTE: Always online and failing edges.

  Args:
    K (int)            : number of shortest paths to generate
    mode (dict)        : the configuration of the problem
    init_config (dict) : kwargs for dijkstra_init()

  Returns:
    k_paths (list)     : [[path: list, path_cost],]
  """
  # Find the absolute shortest path.
  path_data, _ = first_shortest_path(mode, init_config)
  path_data = path_data[:3] + (0, path_data[4])
  k_paths = [(path_data[0], path_data[1])]
  # Holding the potential shortest paths (Yen's B).
  prospects = []
  heapq.heapify(prospects)

  for k in range(1, K):
    if mode.get("yen_") or mode.get("lawler"):
      prospects = yen.update_prospects(k,
                                       K,
                                       k_paths,
                                       mode,
                                       init_config,
                                       path_data[3],
                                       prospects,
                                       path_data[2])
    else:
      repl_paths = replacement_paths(mode,
                                     init_config,
                                     path_data=path_data,
                                     k_paths=k_paths)
      for path in repl_paths:
        yen.push_prospect(*path, K, k, prospects)

    # Add the best prospect to the k_paths list
    if prospects:
      # Check if at least K - k prospects with the same cost as the (k - 1)th
      # path were already found.
      border_cost = heapq.nsmallest(K - k, prospects)[-1][0]
      last_path_cost = path_data[1]
      if (len(prospects) >= K - k) and (border_cost == last_path_cost):
        for _ in range(K - k):
          path_data = heapq.heappop(prospects)
          k_paths.append((path_data[1], path_data[0]))
        break
      path_data = heapq.heappop(prospects)
      k_paths.append((path_data[1], path_data[0]))
      path_data[0], path_data[1] = path_data[1], path_data[0]
    else:
      break
  return k_paths
