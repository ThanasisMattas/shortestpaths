# dijkstra.py is part of ShortestPaths
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
"""Bidirectional, parallel and dynamic programming implementations of
Dijkstra's algorithm
"""

import copy
import logging
import math
import multiprocessing as mp
from multiprocessing import (Array,
                             current_process,
                             Event,
                             get_logger,
                             log_to_stderr,
                             Process,
                             Queue)
from typing import Hashable
import warnings

from shortestpaths.priorityq import PriorityQueue
from shortestpaths.utils import time_this  # noqa: F401


def was_visited(visited, u):
  if (visited[u][0] == 0) and (visited[u][1] == u):
    return False
  return True


def prospect_init(to_visit,
                  to_visit_reverse,
                  visited,
                  visited_reverse,
                  discovered_forward,
                  discovered_reverse,
                  root_path=None):
  """Initializes the prospect path, upon states retrieval.

  Args:
    to_visit (PriorityQueue)         : the forward PriorityQueue
    to_visit_reverse (PriorityQueue) : the reverse PriorityQueue
    visited (list)                   : the forward Dijkstra results
    visited_reverse (list)           : the reverse Dijkstra results
    discovered_forward (set)         : discovered but not visited - forward
    discovered_reverse (set)         : discovered but not visited - reverse
    root_path (list)                 : when online, in order to ignore the node
                                       being probed, if it belongs to the root

  Returns:
    prospect (list)                  : [
                                         path_cost,
                                         meeting_edge_tail,
                                         meeting_edge_head,
                                         meeting_edge_weight
                                       ]
  """
  prospect_cost = math.inf
  prospect = [0, 0, 0, 0]
  for u in range(1, len(visited)):
    if (root_path) and (u in root_path):
      continue
    # There are two ways of forming a prospect path.
    #  1. The two searches had visited the same node
    #  2. One search discovered a node the other visited.
    if was_visited(visited, u):
      if was_visited(visited_reverse, u):
        new_prospect_cost = visited[u][0] + visited_reverse[u][0]
        if new_prospect_cost < prospect_cost:
          prospect = [new_prospect_cost, u, u, 0]
          prospect_cost = new_prospect_cost
      elif u in discovered_reverse:
        # Then, it was discovered when visiting the tail v:
        v = to_visit_reverse[u][-2]
        uv_weight = to_visit_reverse[u][0] - visited_reverse[v][0]
        new_prospect_cost = (visited[u][0]
                             + uv_weight
                             + visited_reverse[v][0])
        if new_prospect_cost < prospect_cost:
          prospect = [new_prospect_cost, u, v, uv_weight]
          prospect_cost = new_prospect_cost
    if was_visited(visited_reverse, u):
      if u in discovered_forward:
        # Then, it was discovered when visiting the tail v:
        v = to_visit[u][-2]
        uv_weight = to_visit[u][0] - visited[v][0]
        new_prospect_cost = (visited[v][0]
                             + uv_weight
                             + visited_reverse[u][0])
        if new_prospect_cost < prospect_cost:
          prospect = [new_prospect_cost, v, u, uv_weight]
          prospect_cost = new_prospect_cost
  return prospect


def to_visit_init(n, source):
  """Initializes the PriorityQueue.

  Each entry is a triple : [path_to_node_cost, prev_node, node]
  Initialized with       : [inf, node, node]
  """
  to_visit = PriorityQueue([[math.inf, i, i] for i in range(1, n + 1)])
  to_visit[source] = [0, -1, source]
  return to_visit


def visited_init(n):
  """Initializes the list holding the Dijkstra's algorithm results.

  Each entry is a 2-list : [path_to_node_cost, prev_node]
  Initialized with       : [0, current_node]
  """
  return [[0, i] for i in range(n + 1)]


def dijkstra_init(source, sink, adj_list=None, adj_list_reverse=None):
  """Initializes the data structures that are used by Dijkstra's algorithm.

    each config:
      adj_list (list)          : the adjacency list
      to_visit (PriorityQueue) : Holds the data of the nodes not yet visited,
                                 sorted by path-cost.
                                 - path-costs initialized to inf
                                 - u_prev initialized to u
                                 - format: [[u_path_cost, u_prev, u],]
      visited (list)           : Holds the data of the visited nodes.
                                 - path-costs initialized to 0
                                 - u_prev initialized to u
                                 - format:
                                     [
                                       [0, 0]
                                       [1_path_cost, 1_prev],
                                       [2_path_cost, 2_prev],
                                       ...
                                       [n_path_cost, n_prev]
                                     ]
      sink (hashable)

  Returns:
    config_forward (dict)
    config_reverse (dict)
  """

  if adj_list:
    n = len(adj_list) - 1
    forward_config = {
      "adj_list": adj_list,
      "to_visit": to_visit_init(n, source),
      "visited": visited_init(n),
      "sink": sink
    }
  else:
    forward_config = {}

  if adj_list_reverse:
    n = len(adj_list_reverse) - 1
    reverse_config = {
      "adj_list": adj_list_reverse,
      "to_visit": to_visit_init(n, sink),
      "visited": visited_init(n),
      "sink": source
    }
  else:
    reverse_config = {}

  return forward_config, reverse_config


def dijkstra(adj_list,
             sink,
             to_visit,
             visited,
             failed=None,
             tapes_queue=None,
             subpath=None):
  """Dijkstra's algorithm

  Args:
    adj_list (list)          : Each entry refers to the node that corresponds
                               to its index and comprises a list of 2-tuples,
                               each for one neighbor of the index-node:
                               (neighbor_id, weight)
    sink (hashable)          : The sink node id
    to_visit (PriorityQueue) : The nodes not yet visited by the algorithm.
                               Each entry is a list:
                               [path_cost, prev_node_id, node_id]
    visited (list)           : Each entry is a 2-list:
                               [path_cost, prev_node_id]
    failed (hashable)        : Failed node to avoid (defaults to None)
    tapes_queue
        (mp.queues.Queue)    : If true, the step-wise state of the algorithm
                               will be recorded on a tape.
    subpath (list)           : The subpath for which states are recorded.

  Returns:
    visited (list)           : Each entry is a 2-list for each node:
                               [path_cost, prev_node_id]
    tape (list)              : Each record is a state for each failed node:
                               [to_visit, visited, discovered_nodes]
  """
  if subpath:
    tape = []
    subpath_iter = iter(subpath)
    next_path_node = next(subpath_iter)
    # Discovered, yet not visited, nodes. This is used to create the potential
    # shortest paths at bidirectional Dijkstra termination condition.
    discovered = {to_visit.peek()[-1]}

  while to_visit:
    if subpath:
      u_next = to_visit.peek()[-1]
      if u_next == next_path_node:
        # print(f"{current_process().name} u_next: {u_next}")
        tape.append([copy.deepcopy(to_visit),
                     [entry.copy() for entry in visited],
                     discovered.copy()])
        try:
          next_path_node = next(subpath_iter)
        except StopIteration:
          if isinstance(tapes_queue, mp.queues.Queue):
            tapes_queue.put(tape)
            return
          else:
            return tape
      discovered.remove(u_next)

    u_path_cost, u_prev, u = to_visit.pop_low()

    if u_path_cost == math.inf:
      # -1 denotes an unconnected node and, in that case, node and previous
      # node are the same by initialization.
      visited[u][0] = -1
      continue
    visited[u][0] = u_path_cost
    visited[u][1] = u_prev

    if u == sink:
      if subpath:
        raise Exception("The states recording reached the sink, which is not"
                        " part of the recording subpath."
                        f" Process: {current_process().name}")
      return visited

    for v, uv_weight in adj_list[u]:
      if v == failed:
        continue
      if v in to_visit:
        to_visit.relax_priority([u_path_cost + uv_weight, u, v])
        if subpath:
          # print(f"{current_process().name} disovered: {v}")
          discovered.add(v)

  return visited


def _state_idx(i, tape, path, direction, failing):
  """Evaluates the state index in the tape that corresponds to the failed node.

  Args:
    i (int)         : the index of the failed node in the path
    tape (list)     : either forward or reverse
    path (list)     : the base shortest path
    direction (str) : "forward" or "reverse"
    failing (str)   : "nodes" or "edges"

  Returns:
    idx (int)
  """
  # Offset the index by source/sink node, because when failing nodes, those
  # cannot fail.
  source_offset = bool(failing == "edges")
  if direction == "forward":
    if source_offset + i < len(tape):
      idx = source_offset + i - 1
    else:
      idx = -1
  else:
    if len(path) - i < len(tape) + 1 - source_offset:
      idx = -(len(tape) + 1 - source_offset - (len(path) - i) + 1)
    else:
      idx = - 1
  # print(f"path_idx: {i}  {direction}: {idx:2}  len(tape): {len(tape)}")
  return idx


def verify_tapes(tapes, path, failing="nodes"):
  """Sanity test that checks if the states recorded are indeed the previous
  states of each failed node, by verifying that the failed node isn't visited.

  NOTE: Tapes recording only when dynamic and not online.

  A recorded state:
    [to_visit, visited, discovered_but_not_visited]
  """
  error_msg = ("Record for failed node <{node}> with path index <{idx}> in the"
               " {direction} tape is not a previous state, because the node"
               " was visited.")

  check_tapes = {"reverse": tapes[1]}
  if tapes[0]:
    check_tapes["forward"] = tapes[0]

  offset = int(failing == "nodes")

  for i, u in enumerate(path[offset: -offset]):
    i_real = i + offset
    for direction, tape in check_tapes.items():
      state = tape[_state_idx(i_real, tape, path, direction, failing)]
      if ((u not in state[0])
              or (state[1][u][0] != 0)
              or (state[1][u][1] != u)):
        raise KeyError(error_msg.format(node=u,
                                        idx=i_real,
                                        direction=direction))


# @profile
# @time_this
def record_states(mode, init_config, base_path, meeting_edge_head):
  """Memoizes the states of the algorithm on a tape.

  Args:
    mode (dict)             : the configuration of the problem
    init_config (dict)      : kwargs for dijkstra_init()
    base_path (list)        : the parent-path
    meeting_edge_head (int) : used to separate forward and reverse sub-paths

  Returns:
  tape_forward (list)       : [
                                [
                                  to_visit,
                                  visited,
                                  discovered_forward
                                ],
                              ]
  tape_reverse (list)       : [
                                [
                                  to_visit_reverse,
                                  visited_reverse,
                                  discovered_reverse
                                ],
                              ]
  """
  if mode.get("verbose", 0) >= 2:
    log_to_stderr()
    logger = get_logger()
    logger.setLevel(logging.INFO)

  forward_config, reverse_config = dijkstra_init(**init_config)

  # When failing nodes, states for source and sink will not be recorded.
  meeting_edge_head_idx = base_path.index(meeting_edge_head)
  if mode["failing"] == "nodes":
    forward_subpath = base_path[1: meeting_edge_head_idx]
    reverse_subpath = list(reversed(base_path[meeting_edge_head_idx: -1]))
  else:
    forward_subpath = base_path[:meeting_edge_head_idx]
    reverse_subpath = list(reversed(base_path[meeting_edge_head_idx:]))
  if forward_subpath == []:
    forward_subpath = [reverse_config["sink"]]
  if reverse_subpath == []:
    reverse_subpath = [forward_config["sink"]]

  tape_forward = dijkstra(**forward_config, subpath=forward_subpath)
  tape_reverse = dijkstra(**reverse_config, subpath=reverse_subpath)
  return tape_forward, tape_reverse


def record_states_parallel(mode, init_config, base_path, meeting_edge_head):
  """Memoizes the states of the algorithm on a tape (parallel version).

  Args:
    mode (dict)             : the configuration of the problem
    init_config (dict)      : kwargs for dijkstra_init()
    base_path (list)        : the parent-path
    meeting_edge_head (int) : used to separate forward and reverse sub-paths

  Returns:
  tape_forward (list)       : [
                                [
                                  to_visit,
                                  visited,
                                  discovered_forward
                                ],
                              ]
  tape_reverse (list)       : [
                                [
                                  to_visit_reverse,
                                  visited_reverse,
                                  discovered_reverse
                                ],
                              ]
  """
  if mode.get("verbose", 0) >= 2:
    log_to_stderr()
    logger = get_logger()
    logger.setLevel(logging.INFO)

  forward_config, reverse_config = dijkstra_init(**init_config)

  # When failing nodes, states for source and sink will not be recorded.
  # tapes_queue = Queue()
  meeting_edge_head_idx = base_path.index(meeting_edge_head)
  if mode["failing"] == "nodes":
    forward_subpath = base_path[1: meeting_edge_head_idx]
    reverse_subpath = list(reversed(base_path[meeting_edge_head_idx: -1]))
  else:
    forward_subpath = base_path[:meeting_edge_head_idx]
    reverse_subpath = list(reversed(base_path[meeting_edge_head_idx:]))
  if forward_subpath == []:
    forward_subpath = [reverse_config["sink"]]
  if reverse_subpath == []:
    reverse_subpath = [forward_config["sink"]]

  tapes_queue = Queue()

  forward_search = Process(name="forward_unidirectional_search",
                           target=dijkstra,
                           kwargs={**forward_config,
                                   "tapes_queue": tapes_queue,
                                   "subpath": forward_subpath})
  reverse_search = Process(name="reverse_unidirectional_search",
                           target=dijkstra,
                           kwargs={**reverse_config,
                                   "tapes_queue": tapes_queue,
                                   "subpath": reverse_subpath})
  # Deaemonize the processes to be terminated upon exit.
  forward_search.daemon = True
  reverse_search.daemon = True
  forward_search.start()
  reverse_search.start()
  # We need to use the consumer before joining the processes, because the data
  # is quite big. The underlying thread that pops from the dequeue and makes
  # data available, uses a pipe or a Unix socket, which have a limited capaci-
  # ty. When the pipe or socket are full, the thread blocks on the syscall,
  # resulting to a deadlock, because join waits for the thread to terminate.
  tape_1 = tapes_queue.get()
  tape_2 = tapes_queue.get()
  # Find out which is which.
  if len(tape_1) == len(forward_subpath):
    return tape_1, tape_2
  else:
    return tape_2, tape_1


def retrieve_state(direction,
                   tape,
                   init_config,
                   base_path,
                   failed_path_idx,
                   failing):
  """Retrieves the state that corresponds to the failed node or edge.

  Args:
    direction (str)       : "forward" or "reverse"
    tapes (list)          : the recored states
    init_config (dict)    : kwargs for dijkstra_init()
    base_path (list)      : the parent-path
    failed_path_idx (int) : the base-path-idx of the failed node
                            When failing edges, for the forward search pass the
                            tail idx and for the reverse pass the head idx.
    failing (str)         : "nodes" or "edges"

  Returns:
    config (dict)         : kwargs for dijkstra.dijkstra()
    discovered (set)      : nodes discovered but not visited
  """

  if direction == "forward":
    adj_list = init_config["adj_list"]
    sink = init_config["sink"]
  else:
    adj_list = init_config["adj_list_reverse"]
    sink = init_config["source"]
  state_idx = _state_idx(failed_path_idx,
                         tape,
                         base_path,
                         direction,
                         failing=failing)
  [to_visit, visited, discovered] = tape[state_idx]
  if state_idx == -1:
    # NOTE: We will use again the last states of each tape, so we need to deep-
    #       copy them.
    to_visit = copy.deepcopy(to_visit)
    visited = [entry.copy() for entry in visited]
    discovered = discovered.copy()
  config = {
    "adj_list": adj_list,
    "to_visit": to_visit,
    "visited": visited,
    "sink": sink
  }
  return config, discovered


def unidirectional_dijkstra(dijkstra_config, mode, failed=None):
  """Wrapper of dijkstra() and extract_path(), in order to have the same usage
  with bidirectional_dijkstra().

  Args:
    dijkstra_config (dict) : kwargs for dijkstra.dijkstra()
    mode (dict)            : the configuration of the problem
    failed (hashable)      : the failed node or edge (default: None)

  Returns:
    path, path_cost, cum_hop_weights, None
  """
  source = dijkstra_config["to_visit"].peek()[-1]
  sink = dijkstra_config["sink"]
  visited_out = dijkstra(**dijkstra_config, failed=failed)
  path_cost = visited_out[sink][0]
  # When online, we need the cumulative hop weights, to retrieve the root-path
  # cost up until the failed node/edge.
  path, cum_hop_weights = extract_path(source, sink, visited_out, mode)
  # None stands for meeting_edge_head, as returned by bidirectional_dijkstra().
  return path, path_cost, cum_hop_weights, None


def _visited_offsets(n):
  """Offset to retrieve the column major packed visited data for each search.
  """
  process_name = current_process().name
  if process_name.startswith("forward"):
    is_forward = True
    visited_offset = 0
    opposite_visited_offset = n
  elif process_name.startswith("reverse"):
    is_forward = False
    visited_offset = n
    opposite_visited_offset = 0
  else:
    raise Exception(f"Unknown process: {process_name}")
  return is_forward, visited_offset, opposite_visited_offset


def _biderectional_dijkstra_branch(adj_list: list,
                                   sink: int,
                                   to_visit: PriorityQueue,
                                   visited_costs: Array,
                                   visited_prev_nodes: Array,
                                   prospect: Array,
                                   priorityq_top: Array,
                                   kill: Event,
                                   failed: Hashable = None,
                                   sync: tuple = None):
  """The target function for either the forward or the reverse search process.

  Args:
    adj_list (list)            : the adjacency list
    sink (int)                 : the source or the spur-node, if it is the
                                 reverse search
    to_visit (PriorityQueue)   : the PriorityQueue
    visited_costs (Array)      : column major packed costs of both searches
    visited_prev_nodes (Array) : column major packed prev nodes of the searches
    prospect (Array)           : see bidirectional Dijkstra termination
    priorityq_top (Array)      : the top values of both PriorityQueue's
    kill (Event)               : flag to notify the other process to finish
    failed (hashable)          : the failed node
    sync (tuple)               : 2 Event's to synchronize the two searches
  """
  # visited_costs and visited_prev_nodes are a single vector shared by both
  # searches; thus, each search has to work with the proper slice.
  n = len(adj_list) - 1
  is_forward, visited_offset, opposite_visited_offset = _visited_offsets(n)

  # Force the synchronization of the processes.
  sync[int(not is_forward)].set()
  sync[int(is_forward)].wait()

  while to_visit:
    u_path_cost, u_prev, u = to_visit.pop_low()

    if u == failed:
      continue

    # -1 denotes an unconnected node and, in that case, node and previous node
    # are the same by initialization.
    with visited_costs.get_lock():
      if u_path_cost == math.inf:
        visited_costs[u + visited_offset] = -1
        continue
      visited_costs[u + visited_offset] = u_path_cost
    with visited_prev_nodes.get_lock():
      visited_prev_nodes[u + visited_offset] = u_prev

    if (kill.is_set()) or (u == sink):
      kill.set()
      return

    for v, uv_weight in adj_list[u]:

      if v == failed:
        continue

      if v in to_visit:
        # Check if v is visited by the other process and, if yes, construct the
        # prospect path.
        with visited_prev_nodes.get_lock():
          if ((visited_prev_nodes[v + opposite_visited_offset] != v)
                  and (visited_costs[v + opposite_visited_offset] != 0)):
            # print(f"{current_process().name}: u: {u}  v: {v}"
            #       f"  v visted from:"
            #       f" {visited_prev_nodes[v + opposite_visited_offset]}")
            uv_prospect_cost = (u_path_cost
                                + uv_weight
                                + visited_costs[v + opposite_visited_offset])
            with prospect.get_lock():
              if (uv_prospect_cost < prospect[0]) or (sum(prospect) == 0):
                # then this is the shortest prospect yet or the 1st one.
                prospect[0] = uv_prospect_cost
                prospect[3] = uv_weight
                if is_forward:
                  prospect[1] = u
                  prospect[2] = v
                else:
                  prospect[1] = v
                  prospect[2] = u
                # print(f"{current_process().name}: {prospect[0]}"
                #       f"  {prospect[1]}  {prospect[2]}\n")

        to_visit.relax_priority([u_path_cost + uv_weight, u, v])

    # Termination condition
    pq_top = to_visit.peek()[0]
    if pq_top == math.inf:
      pq_top = 0
    # print(f"{current_process().name}:  prospect[0]: {prospect[0]}"
    #       f" topf+topr: {priorityq_top[0]} + {priorityq_top[1]}")
    with prospect.get_lock():
      with priorityq_top.get_lock():
        priorityq_top[int(not is_forward)] = pq_top
        if (sum(priorityq_top) >= prospect[0] != 0) or (kill.is_set()):
          kill.set()
          return


def bidirectional_dijkstra_parallel(forward_config,
                                    reverse_config,
                                    mode,
                                    failed=None):
  """Implementation of the bidirectional Dijkstra's algorithm.
  (parallel version)

  Calls forward and reverse searches on two processes and builds the path.

  For the termination condition see:
  Goldberg et al. 2006 "Efficient Point-to-Point Shortest Path Algorithms".

  Args:
    forward_config (dict) : forward search kwargs
    reverse_config (dict) : reverse serach kwargs
    mode (dict)           : the configuration of the simulation
    failed (hashable)     : the failed node, if any (default: None)

  Returns:
    path, path_cost, cum_hop_weights, meeting_edge_head
  """
  n = len(forward_config["adj_list"]) - 1
  to_visit = forward_config["to_visit"]
  visited = forward_config["visited"]
  to_visit_reverse = reverse_config["to_visit"]
  visited_reverse = reverse_config["visited"]
  source = reverse_config["sink"]
  sink = forward_config["sink"]

  if mode.get("verbose", 0) >= 2:
    log_to_stderr()
    logger = get_logger()
    logger.setLevel(logging.INFO)

  # visited_costs and visited_prev_nodes are being concatenated, because they
  # have to be shared between the 2 searches and multiplocessing.Array() only
  # accepts single dimension arrays of one type.
  visited_forward_zip = list(zip(*visited))
  visited_reverse_zip = list(zip(*visited_reverse))
  visited_costs = (list(visited_forward_zip[0])
                   + list(visited_reverse_zip[0][1:]))
  visited_prev_nodes = (list(visited_forward_zip[1])
                        + list(visited_reverse_zip[1][1:]))
  visited_costs = Array('i', visited_costs)
  visited_prev_nodes = Array('i', visited_prev_nodes)
  prospect = Array('i', [0, 0, 0, 0])
  priorityq_top = Array('i', [0, 0])

  shared_memory = {
    "prospect": prospect,
    "priorityq_top": priorityq_top,
    "visited_costs": visited_costs,
    "visited_prev_nodes": visited_prev_nodes,
    "kill": Event(),
    "sync": (Event(), Event()),
    "failed": failed
  }

  forward_search = Process(name="forward_search",
                           target=_biderectional_dijkstra_branch,
                           args=(forward_config["adj_list"],
                                 sink,
                                 to_visit),
                           kwargs=shared_memory)
  reverse_search = Process(name="reverse_search",
                           target=_biderectional_dijkstra_branch,
                           args=(reverse_config["adj_list"],
                                 source,
                                 to_visit_reverse),
                           kwargs=shared_memory)

  forward_search.start()
  reverse_search.start()
  forward_search.join()
  reverse_search.join()

  if sum(prospect):
    # Then, the two searches met, as expected.
    path_cost = prospect[0]
  else:
    # Then, one process finished either before the other started or before they
    # meet.
    if (visited_costs[sink] != 0) and (visited_costs[-1] == 0):
      # then forward_search finished
      path_cost = visited_costs[n]
      prospect = [visited_costs[n], sink, sink]
    elif (visited_costs[sink] == 0) and (visited_costs[-1] != 0):
      # then reverse_search finished
      path_cost = visited_costs[-1]
      prospect = [visited_costs[-1], source, source]
    else:
      raise Exception(f"The two searches didn't meet and neither of them"
                      f" visited the sink.\n"
                      f" visited_costs:\n{list(visited_costs)}\n"
                      f" visited_prev_nodes:\n{list(visited_prev_nodes)}")

  path, cum_hop_weights = extract_bidirectional_path(
      source,
      sink,
      prospect,
      mode,
      n,
      visited_costs,
      visited_prev_nodes,
  )
  return path, path_cost, cum_hop_weights, prospect[2]


# @profile
def _dijkstra_step(adj_list,
                   sink,
                   to_visit,
                   visited,
                   opposite_visited=None,
                   prospect=None,
                   is_forward=None,
                   failed=None):
  """One step inside the <while PriorityQueue> loop.

  Used for the alternately steps of the two searches at bidirectional Dijkstra.
  """
  u_path_cost, u_prev, u = to_visit.pop_low()
  priorityq_top = to_visit.peek()[0]
  if priorityq_top == math.inf:
    priorityq_top = 0

  if u == failed:
    return visited, priorityq_top, prospect

  if u_path_cost == math.inf:
    visited[u][0] = -1
    return visited, priorityq_top, prospect
  else:
    visited[u][0] = u_path_cost
    visited[u][1] = u_prev

  if u == sink:
    return visited, priorityq_top, prospect

  for v, uv_weight in adj_list[u]:
    if v == failed:
      continue
    if v in to_visit:
      if was_visited(opposite_visited, v):
        prospect_cost = (u_path_cost + uv_weight + opposite_visited[v][0])
        if (prospect_cost < prospect[0]) or (sum(prospect) == 0):
          prospect[0] = prospect_cost
          if is_forward:
            prospect[1] = u
            prospect[2] = v
          else:
            prospect[1] = v
            prospect[2] = u
          prospect[3] = uv_weight
      to_visit.relax_priority([u_path_cost + uv_weight, u, v])
  return visited, priorityq_top, prospect


def bidirectional_dijkstra(forward_config,
                           reverse_config,
                           mode,
                           failed=None,
                           prospect=None,
                           top_reverse=None):
  """Implementation of the bidirectional Dijkstra's algorithm.

  Forward and reverse searches run alternately, building and updating a poten-
  tial path, when one discovers a node the other had already met, until the sum
  of the min values of the two PriorityQueue's is greater or equal to the cost
  of the best potential path.

  For the termination condition see:
  Goldberg et al. 2006 "Efficient Point-to-Point Shortest Path Algorithms".

  Args:
    forward_config (dict) : forward search kwargs
    reverse_config (dict) : reverse serach kwargs
    mode (dict)           : the configuration of the simulation
    failed (hashable)     : the failed node, if any (default: None)

  Returns:
    path, path_cost, cum_hop_weights, meeting_edge_head
  """
  if prospect is None:
    prospect = [0, 0, 0, 0]
    top_reverse = 0

  while forward_config["to_visit"] and reverse_config["to_visit"]:
    # Forward step
    visited, top_forward, prospect = _dijkstra_step(
        **forward_config,
        opposite_visited=reverse_config["visited"],
        prospect=prospect,
        is_forward=True,
        failed=failed
    )
    if top_forward + top_reverse > prospect[0] != 0:
      break

    # Reverse step
    visited_reverse, top_reverse, prospect = _dijkstra_step(
        **reverse_config,
        opposite_visited=forward_config["visited"],
        prospect=prospect,
        is_forward=False,
        failed=failed
    )
    if top_forward + top_reverse > prospect[0] != 0:
      break

  path, cum_hop_weights = extract_bidirectional_path(
    reverse_config["sink"],
    forward_config["sink"],
    prospect,
    mode,
    visited=visited,
    visited_reverse=visited_reverse,
  )
  return path, prospect[0], cum_hop_weights, prospect[2]


def extract_bidirectional_path(source,
                               sink,
                               prospect,
                               mode,
                               n=None,
                               visited_costs=None,
                               visited_prev_nodes=None,
                               visited=None,
                               visited_reverse=None):
  """Calls extract_path() for both searches and connects the subpaths.

  Args:
    source, sink (hashable)
    prospect (Array | list)    : [
                                   path_cost,
                                   meeting_edge_tail,
                                   meeting_edge_head,
                                   meeting_edge_weight
                                 ]
    mode (dict)                : the configuration of the problem
    n (int)                    : the order of the graph
    visited_costs (Array)      : column major packed visited costs
    visited_prev_nodes (Array) : column major packed visited prev nodes
    visited (list)             : Dijkstra's alg forward results
    visited_reverse (list)     : Dijkstra's alg reverse results

  Returns:
    path (list)                : the path-nodes
    weights (list)             : the cumulative hop weights of the path
  """
  if (visited is None) and (visited_reverse is None):
    visited_concatenated = list(zip(visited_costs, visited_prev_nodes))
    visited = visited_concatenated[:n + 1]
    visited_reverse = visited_concatenated[n:]
  path, weights = extract_path(source,
                               prospect[1],
                               visited,
                               mode)
  # The slice starts from n to account for 0th index.
  reverse_path, reverse_weights = extract_path(sink,
                                               prospect[2],
                                               visited_reverse,
                                               mode)
  if not (path and reverse_path):
    return [], []

  if prospect[1] == prospect[2]:
    path += reversed(reverse_path[:-1])
    if mode.get("online"):
      # reverse_weights:
      # [0, 10, 30] --> [30 , 20] + weights[-1] each
      if reverse_weights:
        reverse_weights = [reverse_weights[-1] - w + weights[-1]
                           for w in reverse_weights[:-1]]
      weights += reversed(reverse_weights)
  else:
    path += reversed(reverse_path)
    if mode.get("online"):
      # reverse_weights:
      # [0, 10, 30] --> [30, 20, 0] + weights[-1] + edge_weights each
      reverse_weights = [reverse_weights[-1] - w + prospect[3] + weights[-1]
                         for w in reverse_weights]
      weights += reversed(reverse_weights)
  return path, weights


def extract_path(source, sink, visited, mode):
  """Extracts the shortest-path from a Dijkstra's algorithm output.

  Dijkstra's algorithm saves the shortest path cost for each node of the graph,
  as well as its previous node on the path, so as to retrieve the path by
  jumping through previous nodes, until the source node.

  Args:
    source, sink (hashable) : the ids of source and sink nodes
    visited (list)          : Dijkstra's alg forward results
                              each entry is a 2-list: [path_cost, prev_node]
    mode (dict)             : the configuration of the problem

  Returns:
    path (list)             : list of the consecutive nodes in the path
    weights (list | None)   : the cumulative hop-weights of the path
  """
  if mode.get("online"):
    weights = [visited[sink][0]]
    path = [sink]
    u = sink
    while u != source:
      u_prev = visited[u][1]
      u_prev_cost = visited[u_prev][0]
      if u == u_prev:
        # Some node/edge failures may disconnect the graph. This can be dete-
        # cted because at initialization u_prev is set to u. In that case, a
        # warning is printed and the execution moves to the next path, if any.
        if mode.get("verbose", 0) >= 2:
          warnings.warn(f"The source ({source}) is not connected to the sink"
                        f" ({sink}).")
        return [], []
      # The corresponding costs are path-costs. In order to get the hop-cost,
      # we have to offset with the path-cost of the previous node in the path.
      path.append(u_prev)
      weights.append(u_prev_cost)
      u = u_prev
    path.reverse()
    weights.reverse()
    return path, weights
  else:
    path = [sink]
    u = sink
    while u != source:
      u_prev = visited[u][1]
      path.append(u_prev)
      if u == u_prev:
        if mode.get("verbose", 0) >= 2:
          warnings.warn(f"The source ({source}) is not connected to the sink"
                        f" ({sink}).")
        return [], None
      u = u_prev
    path.reverse()
    return path, None
