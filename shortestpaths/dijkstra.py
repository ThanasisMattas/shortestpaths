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
                             Event,
                             get_logger,
                             Lock,
                             log_to_stderr,
                             Process,
                             Queue,
                             Value)
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
def _relax_path_cost(v, uv_weight, to_visit, u, u_path_cost):
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
        _relax_path_cost(v, uv_weight, to_visit, u, u_path_cost)

  return visited, tape


def biderectional_dijkstra_branch(adj_list: list,
                                  sink: Hashable,
                                  to_visit: PriorityQueue,
                                  visited: list,
                                  both_visited: Array,
                                  failed_nodes: list = None,
                                  lock: Lock = None,
                                  we_ve_met: Event = None,
                                  meeting_node: Value = None,
                                  queue: Queue = None):
  """Dijkstra's algorithm

  Args:
    adj_list (list)          : Each entry refers to the node that corresponds
                               to its index and comprises a list of 2-tuples,
                               each for one neighbor of the index-node:
                               (neighbor_id, weight)
    sink (hashable)          : The sink node id
    to_visit (PriorityQueue) : The nodes not yet visited by the algorithm.
        _relax_path_cost(v, uv_weight, to_visit, u, u_path_cost)
                               Each entry is a list:
                               [path_cost, prev_node_id, node_id]
    visited (2D list)        : Each entry is a 2-list:
                               [path_cost, prev_node_id]
    memoize_states (bool)  : If true, the step-wise state of the algorithm
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

  while to_visit:
    u_path_cost, u_prev, u = to_visit.pop_low()

    # -1 denotes an unconnected node and, in that case, node and previous node
    # are the same by initialization.
    visited[u][0] = u_path_cost if u_path_cost != math.inf else -1
    visited[u][1] = u_prev

    # Check if the opposite search already found u. So, we are checking whe-
    # ther prev_node of u is still u or it is updated.
    lock.acquire()
    if (we_ve_met.is_set() or both_visited[u] or (u == sink)):
      if not we_ve_met.is_set():
        meeting_node.value = u
      we_ve_met.set()
      queue.put(visited)
      lock.release()
      return
    lock.release()
    both_visited[u] = 1

    # v is the neighbor id
    for v, uv_weight in adj_list[u]:
      # if both_visited[neighbor[0]]:

      if v in failed_nodes:
        continue

      if v in to_visit:
        _relax_path_cost(v, uv_weight, to_visit, u, u_path_cost)

  queue.put(visited)


def bidirectional_dijkstra(adj_list,
                           sink,
                           to_visit,
                           visited,
                           memoize_states=False,
                           failed_nodes=None,
                           verbose=False):
  source = to_visit.peek()[-1]
  inverted_adj_list = _invert_adj_list(adj_list)
  inverted_visited = copy.deepcopy(visited)
  inverted_to_visit = copy.deepcopy(to_visit)
  inverted_to_visit[source] = [math.inf, source, source]
  inverted_to_visit[sink] = [0, sink, sink]

  if verbose:
    log_to_stderr()
    logger = get_logger()
    logger.setLevel(logging.INFO)

  # This will hold visited and inverted_visited, after the 2 searches meet.
  queue = Queue()
  # A shared list with the visited status of each node. When a process visits a
  # node, it will first check if this node was already visited by the other
  # process; if yes, it will exit there, if no, it will render the node visited
  # by setting its value in the list at 1 and proceed with the search.
  both_visited = Array('i', [0 for _ in range(len(adj_list))], lock=False)
  # When a process visites a node already visited by the other process, it will
  # will set this flag to notify the other process that they 've met.
  we_ve_met = Event()
  meeting_node = Value('i', 0)
  lock = Lock()

  forward_process = Process(name="forward_search",
                            target=biderectional_dijkstra_branch,
                            args=(adj_list,
                                  sink,
                                  to_visit,
                                  visited,
                                  both_visited,
                                  failed_nodes,
                                  lock,
                                  we_ve_met,
                                  meeting_node,
                                  queue))
  backwards_process = Process(name="backwards_search",
                              target=biderectional_dijkstra_branch,
                              args=(inverted_adj_list,
                                    source,
                                    inverted_to_visit,
                                    inverted_visited,
                                    both_visited,
                                    failed_nodes,
                                    lock,
                                    we_ve_met,
                                    meeting_node,
                                    queue))
  forward_process.start()
  backwards_process.start()
  forward_process.join()
  backwards_process.join()

  v1 = queue.get()
  v2 = queue.get()
  if v1[source][1] == source:
    visited, inverted_visited = v1, v2
  else:
    visited, inverted_visited = v2, v1
  path_cost = (visited[meeting_node.value][0]
               + inverted_visited[meeting_node.value][0])
  path = extract_bidirectional_path(visited,
                                    inverted_visited,
                                    source,
                                    sink,
                                    meeting_node.value)
  return [path, path_cost, failed_nodes]


def extract_bidirectional_path(visited,
                               inverted_visited,
                               source,
                               sink,
                               meeting_node,
                               failed=None):
  path = extract_path(visited, source, meeting_node)
  backwards_path = extract_path(inverted_visited, sink, meeting_node)
  path += reversed(backwards_path[:-1])
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
