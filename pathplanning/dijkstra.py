# dijkstra.py is part of PathPlanning
#
# PathPlanning is free software; you may redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. You should have received a copy of the GNU
# General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.
#
# (C) 2020 Athanasios Mattas
# =======================================================================
"""Implemantation of the Dijkstra's algorithm."""

from collections import OrderedDict
import copy
import math
import random
from operator import itemgetter
import sys

from pathplanning.priorityq import PriorityQueue
from pathplanning import utils
from pathplanning.utils import time_this

sys.setrecursionlimit(1500)


# @profile
def _dijkstra(adj_list,
              to_visit,
              goal,
              dijkstra_output,
              saving_states=True,
              dijkstra_states=OrderedDict(),
              avoided_nodes=[]):
  """Runs an adaptive Dijkstra's algorithm recursively.

  This is an optimization of the Dijkstra's algorithm, using memoization, in
  case of multiple paths are to be generated.

  Args:
    adj_list (list)               : each entry is a list of 2-tuples for the
                                    neighbors of the corresponding node:
                                    (neighbor_id, weight)
    to_visit (PriorityQueue)      : holds the nodes not yet visited by the
                                    algorithm, each entry is a list:
                                    [path_cost, prev_node_id, node_id]
    goal (any hashable type)      : the goal node_id
    dijkstra_output (2D list)     : each entry is a 2-list:
                                    (path_cost, prev_node_id)
    saving_states (bool)          : If true, the step-wise state of the algo-
                                    rithm will be saved in a list, so as to be
                                    able to retrieve any of the states.
    dijkstra_states (OrderedDict) : If num_paths > 1, it will hold the
                                    step-wise state of the algorithm in pair:
                                    {node_id: (to_visit, dijkstra_output)},
                                    using as key the expanded node_id.
                                    NOTE: OrderedDict is used, in order to be
                                          able to retrieve the needed state at
                                          the correct order of occurance.
    avoided_nodes (list)          : (defaults to [])

  Returns:
    dijkstra_output (2D list)     : each entry is a 2-list for each node:
                                    [path_cost, prev_node_id]
    dijkstra_states (OrderedDict) : each entry is the step-wise state of the
                                    algorithm as a 2-tuple:
                                    node_id: (to_visit, dijkstra_output)
  """
  if to_visit.empty():
    return dijkstra_output, dijkstra_states

  # visiting_node = [path_cost, prev_node_id, node_id]
  visiting_node = to_visit.pop_low()
  path_to_node_cost = visiting_node[0]
  prev_node_id = visiting_node[1]
  node_id = visiting_node[2]

  # Save its final shortest path
  # (-1 denotes an unconnected node and, in that case, node and previous node
  # are the same by initialization.)
  dijkstra_output[node_id][0] = \
      path_to_node_cost if path_to_node_cost != math.inf else -1
  # Save its previous node
  dijkstra_output[node_id][1] = prev_node_id

  # Momoizing the algorithm step-wise states, so as to later retrieve the
  # appropriate state, in order to calculate an alternative path.
  if saving_states:
    dijkstra_states[node_id] = (copy.deepcopy(to_visit),
                                copy.deepcopy(dijkstra_output))

  if node_id == goal:
    return dijkstra_output, dijkstra_states

  # neighbor = (neighbor_id, weight)
  for neighbor in adj_list[node_id]:
    neighbor_id = neighbor[0]
    neighbor_weight = neighbor[1]

    if neighbor_id in avoided_nodes:
      continue

    if neighbor_id in to_visit:
      old_path_to_neighbor_cost = to_visit[neighbor_id][0]
      new_path_to_neighbor_cost = path_to_node_cost + neighbor_weight
      if new_path_to_neighbor_cost < old_path_to_neighbor_cost:
        to_visit[neighbor_id] = [new_path_to_neighbor_cost,
                                 node_id,
                                 neighbor_id]

  return _dijkstra(adj_list,
                   to_visit,
                   goal,
                   dijkstra_output,
                   saving_states,
                   dijkstra_states,
                   avoided_nodes)


@time_this
def _alternative_paths(num_paths,
                       path,
                       path_cost,
                       saving_states,
                       dijkstra_states,
                       adj_list,
                       start,
                       goal):
  """Generates up to <num_paths> best alternative paths, disregarding the most
  weightful nodes.

  Returns:
    paths_data (list)      : [[path, path_cost, nodes_disconnected],]
                             types: [[list, int, list],]
  """
  path_nodes = list(list(zip(*path))[0])
  paths_data = [[path_nodes, path_cost, []]]

  # Uncomment to sort the path nodes by cost, descending.
  # (Exclude the start and goal nodes, because they will not be avoided.)
  # sorted_path_by_edge_cost = sorted(path[1: -1],
  #                                   key=itemgetter(1),
  #                                   reverse=True)

  if saving_states:
    visited_nodes = list(dijkstra_states.keys())
    # For each alternative path, exclude one node.
    for node in path[1: -1]:
      # Retrieve the algorithm state that corresponds to the previous step.
      checkpoint_node = visited_nodes[visited_nodes.index(node[0]) - 1]
      to_visit, dijkstra_output = dijkstra_states[checkpoint_node]

      # Disconnect the node
      del to_visit[node[0]]

      # Continue with the algorithm execution
      new_dijkstra_output, _ = _dijkstra(adj_list,
                                         to_visit,
                                         goal,
                                         dijkstra_output,
                                         saving_states=False,
                                         avoided_nodes=[node[0]])
      new_path = utils.extract_path(new_dijkstra_output, start, goal)
      new_path_cost = new_dijkstra_output[goal][0]
      if new_path:
        paths_data.append([new_path, new_path_cost, [node[0]]])
  else:
    # Build a new PriorityQueue
    num_nodes = len(adj_list) - 1
    to_visit = [[math.inf, n, n] for n in range(1, num_nodes + 1)]
    to_visit = PriorityQueue(to_visit)
    to_visit[start] = [0, start, start]

    dijkstra_output = [[0, None] for _ in range(num_nodes + 1)]

    for node in path[1: -1]:
      # Disconnect the node
      new_to_visit = copy.deepcopy(to_visit)
      del new_to_visit[node[0]]

      new_adj_list = copy.deepcopy(adj_list)
      for neighbor in new_adj_list[node[0]]:
        # NOTE: Uncomment this, instead of the loop, in case of simple graphs.
        #       In case of weights on nodes, the graph can be considered as a
        #       directed multigraph, where for each edge there is one of oppo-
        #       site direction and with different weight.
        #       Example:
        #         weight(a, b) = weight(b)
        #         weight(b, a) = weight(a)
        # new_adj_list[neighbor[0]].remove([node[0], neighbor[1]])
        for ne in new_adj_list[neighbor[0]]:
          if ne[0] == node[0]:
            new_adj_list[neighbor[0]].remove(ne)
            break
      new_adj_list[node[0]].clear()

      new_dijkstra_output, _ = _dijkstra(new_adj_list,
                                         new_to_visit,
                                         goal,
                                         copy.deepcopy(dijkstra_output),
                                         saving_states)
      new_path = utils.extract_path(new_dijkstra_output, start, goal)
      new_path_cost = new_dijkstra_output[goal][0]
      if new_path:
        paths_data.append([new_path, new_path_cost, [node[0]]])

  paths_data = sorted(paths_data, key=itemgetter(1))
  # Slice the best <num_paths> paths.
  paths_data = paths_data[:num_paths]

  return paths_data


@time_this
def _adapted_path(path,
                  path_cost,
                  saving_states,
                  dijkstra_states,
                  adj_list,
                  start,
                  goal,
                  disconnected_nodes):
  if isinstance(path[0], (list, tuple)):
    path = list(list(zip(*path))[0])
  paths_data = [[path, path_cost, []]]

  # In case of disconnected_nodes are not provided, pick a random node towards
  # the end of the path.
  if not disconnected_nodes:
    disconnected_nodes = [
      path[random.randrange(len(path) // 2, len(path) - 1)]
    ]

  [start, goal] = utils.check_nodal_connection([start, goal],
                                               adj_list,
                                               disconnected_nodes)

  if saving_states:
    visited_nodes = list(dijkstra_states.keys())
    # Retrieve the algorithm state that corresponds to the previous step.
    for i, visited_node in enumerate(visited_nodes):
      if visited_node in disconnected_nodes:
        checkpoint_node = visited_nodes[i - 1]
        break
    to_visit, dijkstra_output = dijkstra_states[checkpoint_node]

    # Disconnect the node
    del to_visit[disconnected_nodes]
  else:
    # Build a new PriorityQueue
    num_nodes = len(adj_list)
    to_visit = [[math.inf, node, node] for node in range(1, num_nodes + 1)]
    to_visit = PriorityQueue(to_visit)
    to_visit[start] = [0, start, start]

    # Disconnect the node
    del to_visit[disconnected_nodes]

    for node in disconnected_nodes:
      for neighbor in adj_list[node]:
        # NOTE: Uncomment this, instead of the loop, in case of simple graphs.
        #       In case of weights on nodes, the graph can be considered as a
        #       directed multigraph, where for each edge there is one of oppo-
        #       site direction and with different weight.
        #       Example:
        #         weight(a, b) = weight(b)
        #         weight(b, a) = weight(a)
        #
        # adj_list[neighbor[0]].remove([node, neighbor[1]])
        for ne in adj_list[neighbor[0]]:
          if ne[0] == node:
            adj_list[neighbor[0]].remove(ne)
            break
      adj_list[node].clear()

    dijkstra_output = [[0, None] for _ in range(num_nodes + 1)]
    visited_nodes = None
    checkpoint_node = None

  # Continue with the algorithm execution
  new_dijkstra_output, _ = _dijkstra(adj_list,
                                     to_visit,
                                     goal,
                                     dijkstra_output,
                                     saving_states=False,
                                     avoided_nodes=disconnected_nodes)

  new_path = utils.extract_path(new_dijkstra_output, start, goal)
  new_path_cost = new_dijkstra_output[goal][0]
  if new_path:
    paths_data.append([new_path, new_path_cost, disconnected_nodes])

  return paths_data, visited_nodes, checkpoint_node


def shortest_path(adj_list,
                  num_nodes,
                  start,
                  goal,
                  num_paths=1,
                  saving_states=True,
                  adapted_path=False,
                  disconnected_nodes=[]):
  """Finds the shortest path from start to goal, using the Dijkstra's algorithm

  Args:
    adj_list (list)                 : each entry is a list of 2-tuples for the
                                      neighbors of the corresponding node:
                                      (neighbor, weight)
    num_nodes (int)                 : the number of nodes
    start, goal (any hashable type) : the start and the goal nodes
    num_paths (int)                 : number of alternative paths to generate
                                      (defaults to 1)
    disconnected_nodes (list)       : (defaults to [])

  Returns:
    paths_data (list)               : format:
                                      [
                                        [path_1, path_1_cost, avoided_nodes_1],
                                        [path_2, path_2_cost, avoided_nodes_2],
                                        ...
                                      ]
                                      Where each path is a list of the nodes of
                                      the shortest path.
  """
  # Build the priority queue holding the current shortest path cost for each
  # node, as well as its privious node. Format:
  # [
  #   [cost_to_node_x, prev_node_id_x, node_id_of_x],
  #   [cost_to_node_y, prev_node_id_y, node_id_of_y],
  #   ...
  # ]
  # Where cost_to_node_x <= cost_to_node_y.
  to_visit = [[math.inf, node, node] for node in range(1, num_nodes + 1)]
  to_visit = PriorityQueue(to_visit)
  to_visit[start] = [0, start, start]

  # Dijkstra's algorithm will return the shortest path to all nodes expanded,
  # starting from the start-node.
  # [
  #   [0, None]
  #   [cost_to_node_1, prev_node_id_1],
  #   [cost_to_node_2, prev_node_id_2],
  #   ...
  #   [cost_to_node_n, prev_node_id_n]
  # ]
  dijkstra_output = [[0, None] for _ in range(num_nodes + 1)]

  # Find the absolute shortest path.
  dijkstra_output, dijkstra_states = _dijkstra(adj_list,
                                               to_visit,
                                               goal,
                                               dijkstra_output,
                                               saving_states)
  path_cost = dijkstra_output[goal][0]
  path = utils.extract_path(dijkstra_output,
                            start,
                            goal,
                            with_step_weights=True)

  if adapted_path:
    paths_data, visited_nodes, checkpoint_node = \
        _adapted_path(path,
                      path_cost,
                      saving_states,
                      dijkstra_states,
                      adj_list,
                      start,
                      goal,
                      disconnected_nodes)
  else:
    if num_paths > 1:
      paths_data = _alternative_paths(num_paths,
                                      path,
                                      path_cost,
                                      saving_states,
                                      dijkstra_states,
                                      adj_list,
                                      start,
                                      goal)
    else:
      paths_data = [[list(list(zip(*path))[0]), path_cost, []]]

  return paths_data
