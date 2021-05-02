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

from collections import OrderedDict
import copy
import logging
import math
from multiprocessing import (Array,
                             current_process,
                             Event,
                             get_logger,
                             Lock,
                             log_to_stderr,
                             Process)
from typing import Hashable
import warnings

# from networkx.utils.misc import iterable

from shortestpaths.priorityq import PriorityQueue


def dijkstra_init(n: int, source: Hashable):
  """Initializes the data structures that are used by Dijkstra's algorithm.

  Returns:
    to_visit (PriorityQueue) : holds the data of the nodes not yet visited
                               - costs initialized to inf
                               - u_prev initialized to u
                               - format:
                                   [
                                     [cost_to_node_x, u_prev_x, x],
                                     [cost_to_node_y, u_prev_y, y],
                                     ...
                                   ]
                              where cost_to_node_x <= cost_to_node_y

    visited (list)           : holds the data of the visited nodes
                               - costs initialized to 0
                               - u_prev initialized to None
                               - format:
                                   [
                                     [0, None]
                                     [cost_to_node_1, u_prev_1],
                                     [cost_to_node_2, u_prev_2],
                                     ...
                                     [cost_to_node_n, u_prev_n]
                                   ]
  """
  to_visit = [[math.inf, node, node] for node in range(1, n + 1)]
  to_visit = PriorityQueue(to_visit)
  to_visit[source] = [0, source, source]
  visited = [[0, None] for _ in range(n + 1)]
  return to_visit, visited


# @profile
def _relax_path_cost(v, u, uv_weight, u_path_cost, to_visit):
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
             memoize_states=False,
             failed_nodes=None):
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
    visited (2D list)        : Each entry is a 2-list:
                               [path_cost, prev_node_id]
    memoize_states (bool)    : If true, the step-wise state of the algorithm
                               will be saved in an OrderedDict.
    failed_nodes (list)      : Nodes to avoid (defaults to None)

  Returns:
    visited (2D list)        : Each entry is a 2-list for each node:
                               [path_cost, prev_node_id]
    tape (OrderedDict)       : Each entry is the step-wise state of the
                               algorithm as a pair:
                               {node_id: (to_visit, visited)}
                               using as key the id of the expanded node
  """
  if not hasattr(failed_nodes, '__iter__'):
    failed_nodes = [failed_nodes]

  if memoize_states:
    tape = OrderedDict()
  else:
    tape = None

  while to_visit:
    u_path_cost, u_prev, u = to_visit.pop_low()

    # -1 denotes an unconnected node and, in that case, node and previous node
    # are the same by initialization.
    visited[u][0] = u_path_cost if u_path_cost != math.inf else -1
    visited[u][1] = u_prev

    if memoize_states:
      tape[u] = (copy.deepcopy(to_visit), copy.deepcopy(visited))

    if u == sink:
      return visited, tape

    # v is the neighbor id
    for v, uv_weight in adj_list[u]:

      if v in failed_nodes:
        continue

      if v in to_visit:
        _relax_path_cost(v, u, uv_weight, u_path_cost, to_visit)

  return visited, tape


def _visited_offsets(n):
  process_name = current_process().name
  if process_name == "forward_search":
    forward = True
    visited_offset = 0
    opposite_visited_offset = n
  elif process_name == "reverse_search":
    forward = False
    visited_offset = n
    opposite_visited_offset = 0
  else:
    raise Exception(f"Unknown process: {process_name}")
  return forward, visited_offset, opposite_visited_offset


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
                                   failed_nodes: list = None):
  """Dijkstra's algorithm

  Args:
    adj_list (list)          : Each entry refers to the node that corresponds
                               to its index and comprises a list of 2-tuples,
                               each for one neighbor of the index-node:
                               (neighbor_id, weight)
    sink (int)               : The sink node id
    to_visit (PriorityQueue) : The nodes not yet visited by the algorithm.
        _relax_path_cost(v, u, uv_weight, u_path_cost, to_visit)
                               Each entry is a list:
                               [path_cost, prev_node_id, node_id]
    visited (2D list)        : Each entry is a 2-list:
                               [path_cost, prev_node_id]
    memoize_states (bool)    : If true, the step-wise state of the algorithm
                               will be saved in an OrderedDict.
    failed_nodes (list)      : Nodes to avoid (defaults to None)

  Returns:
    visited (2D list)        : Each entry is a 2-list for each node:
                               [path_cost, prev_node_id]
    tape (OrderedDict)       : Each entry is the step-wise state of the
                               algorithm as a pair:
                               {node_id: (to_visit, visited)}
                               using as key the id of the expanded node
  """
  forward, visited_offset, opposite_visited_offset = _visited_offsets(n)

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
    priorityq_top[int(not forward)] = pq_top
    # print(f"{current_process().name}  {priorityq_top[int(not forward)]}")
    locks["priorityq_top"].release()

    # Check if the opposite search already found u. So, we are checking whe-
    # ther prev_node of u is still u or it is updated.
    locks["kill"].acquire()
    if (kill.is_set()) or (u == sink):
      kill.set()
      locks["kill"].release()
      return
    locks["kill"].release()

    for v, uv_weight in adj_list[u]:

      if v in failed_nodes:
        continue

      locks["visited"].acquire()
      locks["prospect"].acquire()
      if v in to_visit:
        # Check if v is visited by the other process and, if yes, construct the
        # prospect path.
        if visited_prev_nodes[v + opposite_visited_offset] != v:
          # print(f"{current_process().name}: u: {u}  v: {v}   visited_prev_nodes[v + opposite_visited_offset]: {visited_prev_nodes[v + opposite_visited_offset]}")
          uv_prospect_cost = (u_path_cost
                              + uv_weight
                              + visited_costs[v + opposite_visited_offset])
          if (sum(prospect) == 0) or (uv_prospect_cost < prospect[0]):
            # then this is the 1st prospect or the shortest yet
            prospect[0] = uv_prospect_cost
            if forward:
              prospect[1] = u
              prospect[2] = v
            else:
              prospect[1] = v
              prospect[2] = u
            # print(f"{current_process().name}: {prospect[0]}  {prospect[1]}  {prospect[2]}\n")
        _relax_path_cost(v, u, uv_weight, u_path_cost, to_visit)
      locks["prospect"].release()
      locks["visited"].release()

    # Termination condition
    locks["prospect"].acquire()
    locks["priorityq_top"].acquire()
    # print(f"prospect[0]: {prospect[0]} topf+topr: {pq_top + priorityq_top[int(forward)]}")
    if pq_top + priorityq_top[int(forward)] >= prospect[0] != 0:
      locks["kill"].acquire()
      kill.set()
      locks["kill"].release()
      locks["priorityq_top"].release()
      locks["prospect"].release()
      # print("returningggg")
      return
    locks["priorityq_top"].release()
    locks["prospect"].release()


def bidirectional_dijkstra(adj_list,
                           source,
                           sink,
                           to_visit,
                           memoize_states=False,
                           failed_nodes=None,
                           verbose=False):
  if not hasattr(failed_nodes, '__iter__'):
    failed_nodes = [failed_nodes]
  inverted_adj_list = _invert_adj_list(adj_list)
  inverted_to_visit = copy.deepcopy(to_visit)
  inverted_to_visit[source] = [math.inf, source, source]
  inverted_to_visit[sink] = [0, sink, sink]
  n = len(adj_list) - 1

  if verbose:
    log_to_stderr()
    logger = get_logger()
    logger.setLevel(logging.INFO)

  # A shared list with the visited status of each node. Each entry is a 2-list,
  # where the first item corresponds to the forward search and the second to
  # the reverse search. When a process visits a node, it will render the node
  # visited by setting the corresponding value to the node's path_cost. This
  # information is needed in order for the prospect paths to be evaluated.
  visited_costs = Array('i', [0 for _ in range(2 * n + 1)], lock=False)
  visited_prev_nodes = [i for i in range(n + 1)] + [i for i in range(1, n + 1)]
  visited_prev_nodes = Array('i', visited_prev_nodes, lock=False)
  # The prospect shortest path:
  # (path_cost, forward_search_node, backward_search_node)
  prospect = Array('i', [0, 0, 0], lock=False)
  priorityq_top = Array('i', [0, 0], lock=False)
  # When a process visites a node already visited by the other process, it will
  # will set this flag to notify the other process that they 've met.
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
                                 failed_nodes))
  reverse_search = Process(name="reverse_search",
                           target=_biderectional_dijkstra_branch,
                           args=(inverted_adj_list,
                                 source,
                                 n,
                                 inverted_to_visit,
                                 visited_costs,
                                 visited_prev_nodes,
                                 prospect,
                                 priorityq_top,
                                 kill,
                                 locks,
                                 failed_nodes))
  forward_search.start()
  reverse_search.start()
  forward_search.join()
  reverse_search.join()

  path_cost = prospect[0]
  path = extract_bidirectional_path(source,
                                    sink,
                                    n,
                                    visited_costs,
                                    visited_prev_nodes,
                                    prospect)
  return [path, path_cost, failed_nodes]


def extract_bidirectional_path(source,
                               sink,
                               n,
                               visited_costs,
                               visited_prev_nodes,
                               prospect_path):
  visited = list(zip(visited_costs, visited_prev_nodes))
  path = extract_path(visited[:n + 1], source, prospect_path[1])
  # The slice starts from n to account for 0th index.
  backwards_path = extract_path(visited[n:], sink, prospect_path[2])
  path += reversed(backwards_path)
  return path


def extract_path(visited, source, sink, with_hop_weights=False):
  """Extracts the shortest-path from a Dijkstra's algorithm output.

  Dijkstra's algorithm saves the shortest path cost for each node of the graph,
  as well as its previous node on the path, so as to retrieve the path by
  jumping through previous nodes, until the source node.

  Args:
    visited (2D list)        : each entry is a 2-list:
                               [path_cost, u_prev]
    source, sink (hashable)  : the ids of source and sink nodes
    with_hop_weights (bool)  : - True : returns just the u's
                               - False: returns 2-lists:
                                        (u, hop_cost)

  Returns:
    path (list)              : if with_hop_weights:
                                 each entry is a 2-list,
                                 [u, edge-cost]
                               else:
                                 list of the consecutive nodes in the path
  """
  if with_hop_weights:
    path = [[sink, visited[sink][0]]]
    node = sink
    while node != source:
      prev_node = visited[node][1]
      prev_node_cost = visited[prev_node][0]
      # The corresponding costs are path-costs. In order to get the hop-cost, we
      # have to offset with the path-cost of the previous node in the path.
      path[-1][1] -= prev_node_cost
      path.append([prev_node, prev_node_cost])
      if node == prev_node:
        # Some node/edge failures may disconnect the graph. This can be dete-
        # cted because at initialization u_prev is set to u. In
        # that case, a warning is printed and we move to the next path, if any.
        warnings.warn(f"The source ({source}) is not connected to the sink"
                      f" ({sink}).")
        return []
      node = prev_node
  else:
    path = [sink]
    node = sink
    while node != source:
      prev_node = visited[node][1]
      path.append(prev_node)
      if node == prev_node:
        warnings.warn(f"The source ({source}) is not connected to the sink"
                      f" ({sink}).")
        return []
      node = prev_node

  path.reverse()
  return path
