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

import math

from pathplanning.priorityq import PriorityQueue


def _extract_path(shortest_paths, start, goal):
  """Dijkstra's method saves the shortest path cost for each node of the graph,
  as well as its previous node on the path, so as to retrieve the path by
  jumping through previous nodes, until the start node.

  Args:
    shortest_paths (2D list)        : each entry is a 2-list,
                                      [path_cost, prev_node_id]
    start, goal (any hashable type) : the ids of start and goal nodes

  Returns:
    path (list)                     : the consecutive nodes of the path
  """
  path = [goal]
  node = goal
  while node != start:
    prev_node = shortest_paths[node][1]
    path.append(prev_node)
    if node == prev_node:
      raise Exception("Start node ({}) is not connected to the".format(start)
                      + " destination node ({}).".format(goal))
    node = prev_node
  path.reverse()
  return path


def _dijkstra(adj_list,
              to_visit,
              goal,
              shortest_paths):
  """Runs the dijkstra's algorithm recursively.

  Args:
    adj_list (list)          : each entry is a list of 2-tuples for the
                               neighbors of the corresponding node:
                               (neighbor, weight)
    to_visit (PriorityQueue) : holds the nodes not yet visited by the
                               algorithm, each entry is a list:
                               [path_cost, prev_node_id, node_id]
    goal (any hashable type) : the goal node
    shortest_paths (2D list) : each entry is a 2-list:
                               (path_cost, prev_node_id)

  Returns:
    shortest_paths (2D list) : each entry is a 2-list for each node:
                               [path_cost, prev_node_id]
  """
  if to_visit.empty():
    return shortest_paths

  # u is the currently visited node
  u = to_visit.pop_low()

  # save its final shortest path (-1 denotes an unconnected node)
  shortest_paths[u[-1]][0] = u[0] if u[0] != math.inf else -1
  # save its previous node
  shortest_paths[u[-1]][1] = u[-2]

  if u[-1] == goal:
    return shortest_paths

  # v is the neighbors iterator of u
  for v in adj_list[u[-1]]:
    if ((v[0] in to_visit) and (to_visit[v[0]][0] > (u[0] + v[1]))):
      to_visit[v[0]] = [u[0] + v[1], u[-1], v[0]]
  return _dijkstra(adj_list, to_visit, goal, shortest_paths)


def shortest_path(adj_list, num_nodes, start, goal):
  """Finds the shortest path from start to goal, using the Dijkstra's algorithm

  Args:
    adj_list (list)                 : each entry is a list of 2-tuples for the
                                      neighbors of the corresponding node:
                                      (neighbor, weight)
    num_nodes (int)                 : the number of nodes
    start, goal (any hashable type) : the goal node

  Returns:
    path_cost (int)                 : the total cost of the shortest path
    path (list)                     : a list of the nodes of the shortest path
  """
  # Build the priority queue holding the current shortest path for each node,
  # as well as its privious node. Format:
  # [
  #   [path_cost, count, prev_node_id, node_id],
  #   [path_cost, count, prev_node_id, node_id],
  #   ...
  # ]
  to_visit = [[math.inf, node, node] for node in range(1, num_nodes + 1)]
  to_visit = PriorityQueue(to_visit)
  to_visit[start] = [0, start, start]

  shortest_paths = [[0, None] for _ in range(num_nodes + 1)]
  shortest_paths = _dijkstra(adj_list,
                             to_visit,
                             goal,
                             shortest_paths)
  path_cost = shortest_paths[goal][0]
  path = _extract_path(shortest_paths, start, goal)
  return path_cost, path
