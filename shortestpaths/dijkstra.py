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


def _was_visited(visited, u):
  if (visited[u][0] == 0) and (visited[u][1] == u):
    return False
  return True


def _initialize_prospect_path(n,
                              to_visit,
                              to_visit_reverse,
                              visited,
                              visited_reverse,
                              discovered_forward,
                              discovered_reverse,
                              root_path):
  prospect_cost = math.inf
  prospect = [0, 0, 0]
  for u in range(1, n + 1):
    if u in root_path:
      continue
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


def dijkstra_init(n, source, sink, adj_list, bidirectional):
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
    inverted_adj_list = _invert_adj_list(adj_list)
  else:
    to_visit_reverse = None
    inverted_adj_list = None
  return to_visit, visited, to_visit_reverse, inverted_adj_list


def relax_path_cost(v, u, uv_weight, u_path_cost, to_visit):
  if u_path_cost + uv_weight < to_visit[v][0]:
    to_visit[v] = [u_path_cost + uv_weight, u, v]


def _invert_adj_list(adj_list):
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
             tapes_queue=None,
             subpath=None):
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
    tape (list)              : Each record is a state for each failed node
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
                    copy.deepcopy(visited),
                    copy.copy(discovered)])
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
        relax_path_cost(v, u, uv_weight, u_path_cost, to_visit)
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

  A recorded state:
    [to_visit, visited, discovered_but_not_visited]
  When dynamic + online:
    [to_visit, visited, discovered_but_not_visited, visited_sequence]
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
              or (state[1][u][1] != u)
              or ((state[3]) and (u in state[3]))):
        raise KeyError(error_msg.format(node=u,
                                        idx=i_real,
                                        direction=direction))


# @profile
# @time_this
def record_states(adj_list,
                  inverted_adj_list,
                  source,
                  sink,
                  to_visit,
                  to_visit_reverse,
                  visited,
                  base_path,
                  meeting_edge_head,
                  failing=None,
                  verbose=0):
  """Memoizes the states of the algorithm on a tape."""
  if verbose >= 2:
    log_to_stderr()
    logger = get_logger()
    logger.setLevel(logging.INFO)

  # When failing nodes, states for source and sink will not be recorded.
  tapes_queue = Queue()
  if failing == "nodes":
    forward_subpath = base_path[1: base_path.index(meeting_edge_head)]
    reverse_subpath = list(reversed(
      base_path[base_path.index(meeting_edge_head): -1]
    ))
  else:
    forward_subpath = base_path[:base_path.index(meeting_edge_head)]
    reverse_subpath = list(reversed(
      base_path[base_path.index(meeting_edge_head):]
    ))
  if forward_subpath == []:
    forward_subpath = [source]
  if reverse_subpath == []:
    reverse_subpath = [sink]

  forward_search = Process(name="forward_unidirectional_search",
                           target=dijkstra,
                           args=(adj_list,
                                 sink,
                                 to_visit,
                                 visited,
                                 failing,
                                 tapes_queue,
                                 forward_subpath))
  reverse_search = Process(name="reverse_unidirectional_search",
                           target=dijkstra,
                           args=(inverted_adj_list,
                                 source,
                                 to_visit_reverse,
                                 visited,
                                 failing,
                                 tapes_queue,
                                 reverse_subpath))

  # Deaemonize the processes to be terminated upon exit.
  forward_search.daemon = True
  reverse_search.daemon = True
  forward_search.start()
  reverse_search.start()
  # We need to use the consumer before joining the processes, because the
  # data is quite big. The underlying thread that pops from the dequeue and
  # makes data available, usues a pipe or a Unix socket, which have a limited
  # capacity. When the pipe or socket are full, the thread blocks on the sys-
  # call, resulting to a deadlock, because join waits for the thread to ter-
  # minate.
  tape_1 = tapes_queue.get()
  tape_2 = tapes_queue.get()
  # Find out which is which.
  if len(tape_1) == len(forward_subpath):
    return tape_1, tape_2
  else:
    return tape_2, tape_1


def unidirectional_dijkstra(adj_list,
                            sink,
                            to_visit,
                            visited,
                            failed,
                            with_cum_hop_weights=False,
                            verbose=0):
  """Wrapper of dijkstra() and extract_path(), in order to have the same usage
  with bidirectional_dijkstra().
  """
  source = to_visit.peek()[-1]
  visited_out = dijkstra(adj_list,
                         sink,
                         to_visit,
                         visited,
                         failed)
  path_cost = visited_out[sink][0]
  # When online, we need the cumulative hop weights, to retrieve the root-
  # path cost up until the failed node/edge.
  path, cum_hop_weights = extract_path(
    source,
    sink,
    visited_out,
    with_cum_hop_weights=with_cum_hop_weights,
    verbose=verbose
  )
  # None stands for meeting_edge_head returned by bidirectional_dijkstra().
  return path, path_cost, cum_hop_weights, None


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
                                   failed: Hashable = None,
                                   sync: tuple = None):
  # visited_costs and visited_prev_nodes are a single vector shared by both
  # searches; thus, each search has to work with the proper slice.
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
      else:
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
                if is_forward:
                  prospect[1] = u
                  prospect[2] = v
                else:
                  prospect[1] = v
                  prospect[2] = u
                # print(f"{current_process().name}: {prospect[0]}"
                #       f"  {prospect[1]}  {prospect[2]}\n")

        relax_path_cost(v, u, uv_weight, u_path_cost, to_visit)

    # Termination condition
    pq_top = to_visit.peek()[0]
    if pq_top == math.inf:
      pq_top = 0
    # print(f"{current_process().name}:  prospect[0]: {prospect[0]}"
    #       f" topf+topr: {pq_top + priorityq_top[int(is_forward)]}")
    with prospect.get_lock():
      with priorityq_top.get_lock():
        priorityq_top[int(not is_forward)] = pq_top
        if sum(priorityq_top) >= prospect[0] != 0:
          kill.set()
          return


# @time_this
def bidirectional_dijkstra(adj_list,
                           inverted_adj_list,
                           source,
                           sink,
                           to_visit=None,
                           to_visit_reverse=None,
                           visited=None,
                           visited_reverse=None,
                           discovered_reverse=None,
                           failed_path_idx=None,
                           failed=None,
                           tapes=None,
                           online=False,
                           base_path=None,
                           meeting_edge_head=False,
                           verbose=0):
  """Implementation of the bidirectional Dijkstra's algorithm.

  Calls forward and reverse searches on two processes and builds the path. For
  the termination condition see <Goldberg et al. 2006 "Efficient Point-to-Point
  Shortest Path Algorithms">.

  When using dynamic programming, tapes will hold the states of both searches.
  The function retrieves one state before the failed node for both directions.
  In case of failing edges instead of nodes, in order to get the replacement
  paths, the state of tha tail of the edge is retrieved for the forward search
  and the state that corresponds to the head for the reverse.

  An insight when retrieving a state, is that the algorithm completely ignores
  the path, as Dijkstra's algorithm would do while solving.

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
  """
  n = len(adj_list) - 1

  if verbose >= 2:
    log_to_stderr()
    logger = get_logger()
    logger.setLevel(logging.INFO)

  if (tapes) or (visited_reverse):
    discovered_forward = set()
    if isinstance(failed, tuple):
      failing = "edges"
      # failed = (tail, head)
      failed_forward, failed_reverse = failed
      idx_forward, idx_reverse = failed_path_idx
      root_path = base_path[:idx_forward]
    else:
      failing = "nodes"
      failed_forward = failed_reverse = failed
      idx_forward = idx_reverse = failed_path_idx
      root_path = base_path[:idx_forward - 1]

  if tapes:
    # print(f"source: {source}  len(tapes[1][0]): {len(tapes[1][0])}")
    # Retrieve the forward and reverse states.
    # NOTE: We will use again the last states of each tape, so we need to
    #       deepcopy them. Currently, deepcopy is done at subporcesses crea-
    #       tion.
    [to_visit, visited, discovered_forward] = \
        tapes[0][_state_idx(idx_forward,
                            tapes[0],
                            base_path,
                            "forward",
                            failing=failing)]
    [to_visit_reverse, visited_reverse, discovered_reverse] = \
        tapes[1][_state_idx(idx_reverse,
                            tapes[1],
                            base_path,
                            "reverse",
                            failing=failing)]
    # if failing == "edges":
    #   # When source and sink involve in a failing edge, we don't have a state
    #   # before them to recover from; therefore, they are used as checkpoints of
    #   # their own and we just have to un-discover the
    #   # other edge node.
    #   # NOTE: When online, the source is not base_path[0].
    #   if failed_forward == base_path[0]:
    #     # then un-discover the head
    #     discovered_forward.discard(failed_reverse)
    #     to_visit[failed_reverse] = [math.inf, failed_reverse, failed_reverse]
    #   elif failed_reverse == sink:
    #     # then un-discover the tail
    #     discovered_reverse.discard(failed_forward)
    #     to_visit_reverse[failed_forward] = \
    #         [math.inf, failed_forward, failed_forward]

  if tapes or visited_reverse:
    # Retrieve the prospect path of the state.
    # prospect: [path_cost, forward_search_node, backward_search_node]
    prospect = _initialize_prospect_path(n,
                                         to_visit,
                                         to_visit_reverse,
                                         visited,
                                         visited_reverse,
                                         discovered_forward,
                                         discovered_reverse,
                                         root_path)

    # Check if termination condition is already met.
    top_f = to_visit.peek()[0]
    if top_f == math.inf:
      top_f = 0
    top_r = to_visit_reverse.peek()[0]
    if top_r == math.inf:
      top_r = 0
    # top_f = visited[failed_forward][0]
    # top_r = visited_reverse[failed_reverse][0]

    if (top_f + top_r >= prospect[0]) and (sum(prospect) != 0):
      path_cost = prospect[0]

      path, cum_hop_weights = extract_bidirectional_path(
        source,
        sink,
        n,
        prospect,
        visited=visited,
        visited_reverse=visited_reverse,
        with_cum_hop_weights=online,
        verbose=verbose
      )
      if online:
        return [path, path_cost, cum_hop_weights]
      else:
        return [path, path_cost, failed, prospect[2]]

    # visited_costs and visited_prev_nodes are being concatenated, because they
    # have to be shared between the 2 searches and multiplocessing.Array() only
    # accepts single dimension arrays of one type.
    visited_forward_zip = list(zip(*visited))
    visited_reverse_zip = list(zip(*visited_reverse))
    visited_costs = (list(visited_forward_zip[0])
                     + list(visited_reverse_zip[0][1:]))
    visited_prev_nodes = (list(visited_forward_zip[1])
                          + list(visited_reverse_zip[1][1:]))
    prospect = Array('i', prospect)
    priorityq_top = Array('i', [top_f, top_r])
    visited_costs = Array('i', visited_costs)
    visited_prev_nodes = Array('i', visited_prev_nodes)
  else:  # not tapes
    prospect = Array('i', [0, 0, 0])
    priorityq_top = Array('i', [0, 0])
    visited_costs = Array('i', [0 for _ in range(2 * n + 1)])
    visited_prev_nodes = Array('i', ([i for i in range(n + 1)]
                                     + [i for i in range(1, n + 1)]))
  kill = Event()
  sync = (Event(), Event())

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
                                 failed,
                                 sync))
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
                                 failed,
                                 sync))

  forward_search.start()
  reverse_search.start()
  forward_search.join()
  reverse_search.join()

  if sum(prospect):
    # Then, the two searches met, as expected.
    path_cost = prospect[0]

    if (online) and (prospect[1] != prospect[2]):
      # Then, we need the (prospect[1], prospect[2]) edge weight.
      for u, uv_weight in adj_list[prospect[1]]:
        if u == prospect[2]:
          edge_weight = uv_weight
          break
    else:
      edge_weight = None
  else:
    # Then, one process finished either before the other started or before they
    # meet.
    edge_weight = None
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
                      f" visited_costs:\n{visited_costs}\n"
                      f" visited_prev_nodes:\n{visited_prev_nodes}")

  path, cum_hop_weights = extract_bidirectional_path(
    source,
    sink,
    n,
    prospect,
    visited_costs,
    visited_prev_nodes,
    with_cum_hop_weights=online,
    verbose=verbose,
    edge_weight=edge_weight
  )
  if online:
    if meeting_edge_head:
      return [path, path_cost, cum_hop_weights, prospect[2]]
    else:
      return [path, path_cost, cum_hop_weights]
  else:
    return [path, path_cost, failed, prospect[2]]


def extract_bidirectional_path(source,
                               sink,
                               n,
                               prospect,
                               visited_costs=None,
                               visited_prev_nodes=None,
                               visited=None,
                               visited_reverse=None,
                               with_cum_hop_weights=False,
                               verbose=0,
                               edge_weight=None):
  if (visited is None) and (visited_reverse is None):
    visited_concatenated = list(zip(visited_costs, visited_prev_nodes))
    visited = visited_concatenated[:n + 1]
    visited_reverse = visited_concatenated[n:]
  path, weights = extract_path(source,
                               prospect[1],
                               visited,
                               with_cum_hop_weights,
                               verbose)
  # The slice starts from n to account for 0th index.
  reverse_path, reverse_weights = extract_path(sink,
                                               prospect[2],
                                               visited_reverse,
                                               with_cum_hop_weights,
                                               verbose)
  if prospect[1] == prospect[2]:
    path += reversed(reverse_path[:-1])
    if with_cum_hop_weights:
      # reverse_weights:
      # [0, 10, 30] --> [30 , 20] + weights[-1] each
      if reverse_weights:
        reverse_weights = [reverse_weights[-1] - w + weights[-1]
                           for w in reverse_weights[:-1]]
      weights += reversed(reverse_weights)
  else:
    path += reversed(reverse_path)
    if with_cum_hop_weights:
      # reverse_weights:
      # [0, 10, 30] --> [30, 20, 0] + weights[-1] + edge_weights each
      reverse_weights = [reverse_weights[-1] - w + edge_weight + weights[-1]
                         for w in reverse_weights]
      weights += reversed(reverse_weights)
  return path, weights


def extract_path(source,
                 sink,
                 visited,
                 with_cum_hop_weights=False,
                 verbose=0):
  """Extracts the shortest-path from a Dijkstra's algorithm output.

  Dijkstra's algorithm saves the shortest path cost for each node of the graph,
  as well as its previous node on the path, so as to retrieve the path by
  jumping through previous nodes, until the source node.

  Args:
    visited (2D list)           : each entry is a 2-list: [path_cost, u_prev]
    source, sink (hashable)     : the ids of source and sink nodes
    with_cum_hop_weights (bool) : if True, returns 2 lists, the path and the
                                  cumulative hop-weights

  Returns:
    path (list)                 : list of the consecutive nodes in the path
    weights (list | None)       : the comulative hop-weights
  """
  if with_cum_hop_weights:
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
