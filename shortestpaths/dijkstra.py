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
"""Bidirectional and dynamic implementations of Dijkstra's algorithm"""

import copy
import logging
import math
from multiprocessing import (Array,
                             current_process,
                             Event,
                             get_logger,
                             Lock,
                             log_to_stderr,
                             Process,
                             Queue)
from typing import Hashable
import warnings

from shortestpaths.priorityq import PriorityQueue
from shortestpaths.utils import time_this  # noqa: F401


def _was_visited(visited, u):
  if (visited[u][0] != 0) and (visited[u][1] != u):
    return True
  return False


def _initialize_prospect_path(n,
                              to_visit,
                              to_visit_reverse,
                              visited,
                              visited_reverse,
                              discovered_forward,
                              discovered_reverse):
  prospect_cost = math.inf
  prospect = [0, 0, 0]
  for u in range(1, n + 1):
    if _was_visited(visited, u):
      if _was_visited(visited_reverse, u):
        new_prospect_cost = visited[u][0] + visited_reverse[u][0]
        if new_prospect_cost < prospect_cost:
          prospect = [new_prospect_cost, u, u]
          prospect_cost = new_prospect_cost

      if u in discovered_reverse:
        v = to_visit_reverse[u][-2]
        uv_weight = to_visit_reverse[u][0] - visited_reverse[v][0]
        new_prospect_cost = (visited[u][0]
                             + uv_weight
                             + visited_reverse[v][0])
        if new_prospect_cost < prospect_cost:
          prospect = [new_prospect_cost, u, v]
          prospect_cost = new_prospect_cost
    if _was_visited(visited_reverse, u):
      if u in discovered_forward:
        v = to_visit[u][-2]
        uv_weight = to_visit[u][0] - visited[v][0]
        new_prospect_cost = (visited[v][0]
                             + uv_weight
                             + visited_reverse[u][0])
        if new_prospect_cost < prospect_cost:
          prospect = [new_prospect_cost, v, u]
          prospect_cost = new_prospect_cost
  return prospect


def dijkstra_init(n, source, sink, bidirectional):
  """Initializes the data structures that are used by Dijkstra's algorithm.

  Returns:
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
  """
  to_visit = [[math.inf, node, node] for node in range(1, n + 1)]
  to_visit = PriorityQueue(to_visit)
  to_visit[source] = [0, source, source]
  visited = [[0, node] for node in range(n + 1)]
  if bidirectional:
    to_visit_reverse = copy.deepcopy(to_visit)
    to_visit_reverse[source] = [math.inf, source, source]
    to_visit_reverse[sink] = [0, sink, sink]
  else:
    to_visit_reverse = None
  return to_visit, visited, to_visit_reverse


# @profile
def _relax_path_cost(v, u, uv_weight, u_path_cost, to_visit):
  if u_path_cost + uv_weight < to_visit[v][0]:
    to_visit[v] = [u_path_cost + uv_weight, u, v]


def invert_adj_list(adj_list):
  """Creates the adj_list of the inverted graph in O(n^2).

  The inverted graph is the same with the direct, but with inverted edges
  direction.
  """
  i_adj_list = [set() for _ in range(len(adj_list))]
  for node, neighbors in enumerate(adj_list):
    for neighbor in neighbors:
      i_adj_list[neighbor[0]].add((node, neighbor[1]))
  return i_adj_list


def dijkstra(adj_list,
             sink,
             to_visit,
             visited,
             failed=None,
             recording=False,
             tapes_queue=None,
             checkpoints=None):
  """Dijkstra's algorithm

  Args:
    adj_list (list)          : Each entry refers to the node that corresponds
                               to its index and comprises a list of 2-tuples,
                               each for one neighbor of the index-node:
                               (neighbor_id, weight)
    sink (Hashable)          : The sink node id
    to_visit (PriorityQueue) : The nodes not yet visited by the algorithm.
                               Each entry is a list:
                               [path_cost, prev_node_id, node_id]
    visited (2D list)        : Each entry is a 2-list:
                               [path_cost, prev_node_id]
    dynamic (bool)           : If true, the step-wise state of the algorithm
                               will be recorded on a tape.
    failed (Hashable)        : Nodes to avoid (defaults to None)

  Returns:
    visited (2D list)        : Each entry is a 2-list for each node:
                               [path_cost, prev_node_id]
    tape (OrderedDict)       : Each entry is the step-wise state of the
                               algorithm as a pair:
                               {node_id: (to_visit, visited)}
                               using as key the id of the expanded node
  """
  if recording:
    if checkpoints:
      tape = []
      cps = iter(checkpoints)
      cp = next(cps)
      discovered = {to_visit.peek()[-1]}
    else:
      n = max(sink, to_visit.peek()[-1])
      visited_nodes_sequence = [0 for _ in range(n)]
      it = 0

  while to_visit:
    u_path_cost, u_prev, u = to_visit.pop_low()

    # -1 denotes an unconnected node and, in that case, node and previous node
    # are the same by initialization.
    visited[u][0] = u_path_cost if u_path_cost != math.inf else -1
    visited[u][1] = u_prev

    if recording:
      if checkpoints:
        # u is now visited
        discovered.remove(u)
      else:
        # visited_nodes_sequence.append(u)
        visited_nodes_sequence[it] = u
        it += 1

    if u == sink:
      if recording:
        if checkpoints:
          raise Exception("The states (2nd) recording reached the sink."
                          f" Process: {current_process().name}")
        else:
          # Here exits the 1st recording session (visited nodes sequence).
          tapes_queue.put((visited, visited_nodes_sequence[:it]))
          return
      else:
        return visited

    for v, uv_weight in adj_list[u]:

      if v == failed:
        continue

      if v in to_visit:
        _relax_path_cost(v, u, uv_weight, u_path_cost, to_visit)
        if recording and checkpoints:
          # print(f"{current_process().name} disovered: {v}")
          discovered.add(v)

    if recording and checkpoints:
      if u == cp:
        # print(f"{current_process().name} cp: {cp}")
        tape.append([copy.deepcopy(to_visit),
                     copy.deepcopy(visited),
                     copy.deepcopy(discovered)])
        try:
          cp = next(cps)
        except StopIteration:
          # Here exits the 2nd recording session (Dijkstra states).
          if current_process().name == "reverse_search":
            # Reverse back to the failing nodes sequence.
            tape.reverse()
          tapes_queue.put(tape)
          # print(f"{current_process().name} putting tape")
          return

  return visited


# @time_this
# @profile
def bidirectional_recording(adj_list,
                            inverted_adj_list,
                            source,
                            sink,
                            to_visit,
                            to_visit_reverse,
                            visited,
                            checkpoints=None,
                            mode="k_shortest_paths",
                            verbose=0):
  """Memoizes the needed states of the algorithm on a tape.

  Two recordings are taking place on both directions. The 1st one records the
  visited nodes sequence, so as to get the immediate previously visited node
  for each internal node of the absolute shortest path. The 2nd recording uses
  those nodes as checkpoints, in order to save the corresponding states of the
  algorithm.
  """
  if verbose >= 2:
    log_to_stderr()
    logger = get_logger()
    logger.setLevel(logging.INFO)

  tapes_queue = Queue()

  if checkpoints:
    cps_forward = checkpoints[0]
    cps_reverse = checkpoints[1]
    # __import__("shortestpaths.utils").deb_trace(f"cps forward: {cps_forward}")
    # __import__("shortestpaths.utils").deb_trace(f"cps reverse: {cps_reverse}")
  else:
    cps_forward = None
    cps_reverse = None

  # Visited nodes sequence will be recorded for both directions, in order to
  # extract the appropriate checkpoint nodes.
  forward_search = Process(name="forward_search",
                           target=dijkstra,
                           args=(adj_list,
                                 sink,
                                 to_visit,
                                 visited,
                                 None,
                                 True,
                                 tapes_queue,
                                 cps_forward))
  reverse_search = Process(name="reverse_search",
                           target=dijkstra,
                           args=(inverted_adj_list,
                                 source,
                                 to_visit_reverse,
                                 visited,
                                 None,
                                 True,
                                 tapes_queue,
                                 cps_reverse))

  forward_search.daemon = True
  reverse_search.daemon = True
  forward_search.start()
  reverse_search.start()
  # We need to use the consumer before joining the processes, because the data
  # is quite big. The underlying thread that pops fromt the dequeue and makes
  # data available, usues a pipe or a Unix socket, which have a limited capaci-
  # ty. When the pipe or socket are full, the thread blocks on the syscall and
  # we get a deadlock, because join waits for the thread to terminate.
  if checkpoints:  # then this is the 2nd recording (Dijsktra states)
    tape_1 = tapes_queue.get()
    tape_2 = tapes_queue.get()
    # Find out which is which.
    if sink in tape_1[0][0]:
      return tape_1, tape_2
    else:
      return tape_2, tape_1

  # The rest refer to the 1st recording (visited nodes sequence)
  visited_1, visited_nodes_sequence_1 = tapes_queue.get()
  visited_2, visited_nodes_sequence_2 = tapes_queue.get()
  forward_search.join()
  reverse_search.join()

  # Find out which is which.
  if visited_1[source][0] == 0:  # then 1 is the forward search
    visited_forward = visited_1
    forward_seq, reverse_seq = \
        visited_nodes_sequence_1, visited_nodes_sequence_2
  else:  # then 1 is the reverse search
    visited_forward = visited_2
    forward_seq, reverse_seq = \
        visited_nodes_sequence_2, visited_nodes_sequence_1
  shortest_path_cost = visited_forward[sink][0]
  shortest_path, cum_hop_weights = extract_path(
    source,
    sink,
    visited_forward,
    cum_hop_weights=(mode == "k_shortest_paths"),
    verbose=verbose)
  if mode == "k_shortest_paths":
    path_data = [shortest_path, cum_hop_weights, shortest_path_cost]
  else:
    path_data = [shortest_path, shortest_path_cost, None]

  # Now, record the tapes.
  checkpoints_forward = []
  checkpoints_reverse = []
  for node in shortest_path[1: -1]:
    checkpoints_forward.append(forward_seq[forward_seq.index(node) - 1])
    checkpoints_reverse.append(reverse_seq[reverse_seq.index(node) - 1])
  # Although the sequence of the failed nodes are the same for both searches,
  # when constructing the corresponding replacement path, when recording, the
  # reverse search will visit them in reversed order, thus the reverse check-
  # point list has to be reversed and the reverse tape has to be reversed
  # back, to match the failing sequence.
  checkpoints_reverse.reverse()
  checkpoints = (checkpoints_forward, checkpoints_reverse)

  tapes = bidirectional_recording(adj_list,
                                  inverted_adj_list,
                                  source,
                                  sink,
                                  to_visit,
                                  to_visit_reverse,
                                  visited,
                                  checkpoints,
                                  verbose=verbose)

  return path_data, tapes


def _visited_offsets(n):
  process_name = current_process().name
  if process_name == "forward_search":
    is_forward = True
    visited_offset = 0
    opposite_visited_offset = n
  elif process_name == "reverse_search":
    is_forward = False
    visited_offset = n
    opposite_visited_offset = 0
  else:
    raise Exception(f"Unknown process: {process_name}")
  return is_forward, visited_offset, opposite_visited_offset


def _biderectional_dijkstra_branch(adj_list: list,
                                   sink: int,
                                   n: int,
                                   to_visit: PriorityQueue,
                                   visited_costs: Array,
                                   visited_prev_nodes: Array,
                                   prospect: Array,
                                   priorityq_top: Array,
                                   kill: Event,
                                   locks: dict,
                                   failed: Hashable = None):
  # visited_costs and visited_prev_nodes are a single vector shared by both
  # searches, thus each search has to work with the proper slice.
  is_forward, visited_offset, opposite_visited_offset = _visited_offsets(n)

  while to_visit:
    u_path_cost, u_prev, u = to_visit.pop_low()

    # -1 denotes an unconnected node and, in that case, node and previous node
    # are the same by initialization.
    locks["visited"].acquire()
    visited_costs[u + visited_offset] = \
        u_path_cost if u_path_cost != math.inf else -1
    visited_prev_nodes[u + visited_offset] = u_prev
    locks["visited"].release()

    pq_top = to_visit.peek()[0]
    pq_top = 0 if pq_top == math.inf else pq_top
    locks["priorityq_top"].acquire()
    priorityq_top[int(not is_forward)] = pq_top
    # print(f"{current_process().name}  {priorityq_top[int(not is_forward)]}")
    locks["priorityq_top"].release()

    locks["kill"].acquire()
    if (kill.is_set()) or (u == sink):
      kill.set()
      locks["kill"].release()
      return
    locks["kill"].release()

    for v, uv_weight in adj_list[u]:

      if v == failed:
        continue

      locks["visited"].acquire()
      if v in to_visit:
        # Check if v is visited by the other process and, if yes, construct the
        # prospect path.
        if visited_prev_nodes[v + opposite_visited_offset] != v:
          # print(f"{current_process().name}: u: {u}  v: {v}   visited_prev_nodes[v + opposite_visited_offset]: {visited_prev_nodes[v + opposite_visited_offset]}")
          uv_prospect_cost = (u_path_cost
                              + uv_weight
                              + visited_costs[v + opposite_visited_offset])
          locks["prospect"].acquire()
          if (uv_prospect_cost < prospect[0]) or (sum(prospect) == 0):
            # then this is the shortest prospect yet or the 1st one.
            prospect[0] = uv_prospect_cost
            if is_forward:
              prospect[1] = u
              prospect[2] = v
            else:
              prospect[1] = v
              prospect[2] = u
            # print(f"{current_process().name}: {prospect[0]}  {prospect[1]}  {prospect[2]}\n")
          locks["prospect"].release()
        _relax_path_cost(v, u, uv_weight, u_path_cost, to_visit)
      locks["visited"].release()

    # Termination condition
    locks["prospect"].acquire()
    locks["priorityq_top"].acquire()
    # print(f"prospect[0]: {prospect[0]} topf+topr: {pq_top + priorityq_top[int(is_forward)]}")
    if sum(priorityq_top) >= prospect[0] != 0:
      locks["kill"].acquire()
      kill.set()
      locks["kill"].release()
      locks["priorityq_top"].release()
      locks["prospect"].release()
      return
    locks["priorityq_top"].release()
    locks["prospect"].release()


# @time_this
def bidirectional_dijkstra(adj_list,
                           inverted_adj_list,
                           source,
                           sink,
                           to_visit=None,
                           to_visit_reverse=None,
                           failed_path_idx=None,
                           failed=None,
                           tapes=None,
                           mode="k_shortest_paths",
                           verbose=0):
  n = len(adj_list) - 1

  if verbose > 2:
    log_to_stderr()
    logger = get_logger()
    logger.setLevel(logging.INFO)

  if tapes:
    # Retrieve the forward and reverse states.
    # NOTE: tapes start from the 2nd path-node, so the path_idx is offset by 1.
    tape_forward, tape_reverse = tapes
    [to_visit, visited, discovered_forward] = \
        tape_forward[failed_path_idx - 1]
    [to_visit_reverse, visited_reverse, discovered_reverse] = \
        tape_reverse[failed_path_idx - 1]

    to_visit_reverse[failed] = [math.inf, failed, failed]
    to_visit[failed] = [math.inf, failed, failed]

    # Retrieve the prospect path of the state.
    # prospect: [path_cost, forward_search_node, backward_search_node]
    prospect = _initialize_prospect_path(n,
                                         to_visit,
                                         to_visit_reverse,
                                         visited,
                                         visited_reverse,
                                         discovered_forward,
                                         discovered_reverse)

    # Check if termination condition is already met.
    top_f = to_visit.peek()[0]
    top_r = to_visit_reverse.peek()[0]
    prospect = Array('i', prospect, lock=False)

    if (top_f + top_r >= prospect[0]) and (sum(prospect) != 0):
      path_cost = prospect[0]

      path = extract_bidirectional_path(source,
                                        sink,
                                        n,
                                        prospect,
                                        visited=visited,
                                        visited_reverse=visited_reverse)
      return [path, path_cost, failed]

    # visited_costs and visited_prev_nodes are being concatenated, because they
    # have to be shared between the 2 searches and multiplocessing.Array() only
    # accepts single dimension arrays of one type.
    visited_forward_zip = list(zip(*visited))
    visited_backward_zip = list(zip(*visited_reverse))
    visited_costs = (list(visited_forward_zip[0])
                     + list(visited_backward_zip[0][1:]))
    visited_prev_nodes = (list(visited_forward_zip[1])
                          + list(visited_backward_zip[1][1:]))
    priorityq_top = Array('i', [top_f, top_r], lock=False)
  else:  # not tapes
    visited_costs = [0 for _ in range(2 * n + 1)]
    visited_prev_nodes = ([i for i in range(n + 1)]
                          + [i for i in range(1, n + 1)])
    prospect = Array('i', [0, 0, 0], lock=False)
    priorityq_top = Array('i', [0, 0], lock=False)

  visited_costs = Array('i', visited_costs, lock=False)
  visited_prev_nodes = Array('i', visited_prev_nodes, lock=False)
  kill = Event()
  shared_vars = ["visited", "priorityq_top", "prospect", "kill"]
  locks = {var: Lock() for var in shared_vars}

  forward_search = Process(name="forward_search",
                           target=_biderectional_dijkstra_branch,
                           args=(adj_list,
                                 sink,
                                 n,
                                 to_visit,
                                 visited_costs,
                                 visited_prev_nodes,
                                 prospect,
                                 priorityq_top,
                                 kill,
                                 locks,
                                 failed))
  reverse_search = Process(name="reverse_search",
                           target=_biderectional_dijkstra_branch,
                           args=(inverted_adj_list,
                                 source,
                                 n,
                                 to_visit_reverse,
                                 visited_costs,
                                 visited_prev_nodes,
                                 prospect,
                                 priorityq_top,
                                 kill,
                                 locks,
                                 failed))

  forward_search.start()
  reverse_search.start()
  forward_search.join()
  reverse_search.join()

  path_cost = prospect[0]
  path, cum_hop_weights = extract_bidirectional_path(
    source,
    sink,
    n,
    prospect,
    visited_costs,
    visited_prev_nodes,
    cum_hop_weights=(mode == "k_shortest_paths"),
    verbose=verbose)
  if mode == "k_shortest_paths":
    return [path, cum_hop_weights, path_cost, failed]
  else:
    return [path, path_cost, failed]


def extract_bidirectional_path(source,
                               sink,
                               n,
                               prospect,
                               visited_costs=None,
                               visited_prev_nodes=None,
                               visited=None,
                               visited_reverse=None,
                               cum_hop_weights=False,
                               verbose=0):
  if (visited is None) and (visited_reverse is None):
    visited_concatenated = list(zip(visited_costs, visited_prev_nodes))
    visited = visited_concatenated[:n + 1]
    visited_reverse = visited_concatenated[n:]
  path, weights = extract_path(source,
                               prospect[1],
                               visited,
                               cum_hop_weights,
                               verbose)
  # The slice starts from n to account for 0th index.
  reverse_path, reverse_weights = extract_path(sink,
                                               prospect[2],
                                               visited_reverse,
                                               cum_hop_weights,
                                               verbose)
  if prospect[1] == prospect[2]:
    path += reversed(reverse_path[:-1])
    if cum_hop_weights:
      weights += reversed(reverse_weights[:-1])
  else:
    path += reversed(reverse_path)
    if cum_hop_weights:
      weights += reversed(reverse_weights)
  return path, weights


def extract_path(source,
                 sink,
                 visited,
                 cum_hop_weights=False,
                 verbose=0):
  """Extracts the shortest-path from a Dijkstra's algorithm output.

  Dijkstra's algorithm saves the shortest path cost for each node of the graph,
  as well as its previous node on the path, so as to retrieve the path by
  jumping through previous nodes, until the source node.

  Args:
    visited (2D list)       : each entry is a 2-list: [path_cost, u_prev]
    source, sink (hashable) : the ids of source and sink nodes
    cum_hop_weights (bool)  : if True, returns 2 lists, the path and the cumu-
                              lative hop-weights

  Returns:
    path (list)             : list of the consecutive nodes in the path
    weights (list)          : the comulative hop-weights (if cum_hop_weights)
  """
  if cum_hop_weights:
    weights = [visited[sink][0]]
    path = [sink]
    u = sink
    while u != source:
      u_prev = visited[u][1]
      u_prev_cost = visited[u_prev][0]
      if u == u_prev:
        # Some node/edge failures may disconnect the graph. This can be dete-
        # cted because at initialization u_prev is set to u. In
        # that case, a warning is printed and we move to the next path, if any.
        if verbose >= 2:
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
        if verbose >= 2:
          warnings.warn(f"The source ({source}) is not connected to the sink"
                        f" ({sink}).")
        return [], None
      u = u_prev
    path.reverse()
    return path, None
