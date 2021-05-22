# yen.py is part of ShortestPaths
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
"""Implementation of Yen's k-shortests paths algorithm with improvements.

  Improvements (see Brander-Sinclair 1996):
    - Not searching for deviation paths already found. (Lawler 1972)
    - Using a heap instead of list (Yen's B), to store candidate paths.
    - If at least K - k candidates with the same cost as the (k - 1)th path
      were already found, append them to the k_paths list (Yen's A) and return.
"""

import copy
import heapq

from shortestpaths import dijkstra


def fail_found_spur_edges(adj_list,
                          spur_node,
                          spur_node_path_idx,
                          base_path,
                          k_paths,
                          inverted_adj_list=None,
                          head=None):
  """Failes the edges having spur-node as tail, for each of the K - k found
  paths, that have the same root-path with the prospect-path."""
  # {head: (head, edge_cost)}
  failed_edges = dict()
  # {invertd_tail (head): (inverted_head (tail), edge_cost)}
  failed_inverted_edges = dict()
  for j in k_paths:
    if j[0][:spur_node_path_idx + 1] == base_path[:spur_node_path_idx + 1]:
      failed_edges[j[0][spur_node_path_idx + 1]] = None
      failed_inverted_edges[j[0][spur_node_path_idx + 1]] = None
  # Don't disconnect the failed edge yet, because it will be disconnected
  # in the subsequent loop.
  if head:
    del failed_edges[head]
    del failed_inverted_edges[head]

  for v, uv_weight in adj_list[spur_node]:
    if v in failed_edges.keys():
      failed_edges[v] = (v, uv_weight)
      failed_inverted_edges[v] = (spur_node, uv_weight)
  for v, edge in failed_edges.items():
    adj_list[spur_node].remove(edge)

  if inverted_adj_list:
    for u, edge in failed_inverted_edges.items():
      inverted_adj_list[u].remove(edge)
  return failed_edges, failed_inverted_edges


def reconnect_spur_edges(spur_node,
                         adj_list,
                         failed_edges,
                         inverted_adj_list=None,
                         failed_inverted_edges=None):
  for _, edge in failed_edges.items():
    adj_list[spur_node].add(edge)
  failed_edges.clear()
  if inverted_adj_list:
    for u, edge in failed_inverted_edges.items():
      inverted_adj_list[u].add(edge)
    failed_inverted_edges.clear()


def push_prospect(path,
                  path_cost,
                  path_hop_weights,
                  spur_node_idx,
                  K,
                  k,
                  prospects):
  if ((len(prospects) < K - k)
          or (path_cost < heapq.nsmallest(K - k, prospects)[-1][0])):
    # Check if the prospect is already found.
    prospect_already_found = False
    for p_cost, p, c, d in prospects:
      if (p_cost == path_cost) and (p == path):
        prospect_already_found = True
        break
    if not prospect_already_found:
      heapq.heappush(prospects,
                     (path_cost, path, path_hop_weights, spur_node_idx))
  return prospects


def push_kth_path(prospects, K, k, last_path, k_paths):
  # Check if at least K - k prospects with the same cost as the (k - 1)th
  # path were already found.
  if ((len(prospects) >= K - k)
          and heapq.nsmallest(K - k, prospects)[-1][0] == last_path[0]):
    for _ in range(K - k):
      kth_path = heapq.heappop(prospects)
      k_paths.append([kth_path[1], kth_path[0], None])
    return None, None
  kth_path = heapq.heappop(prospects)
  last_path = kth_path[1]
  k_paths.append([last_path, kth_path[0], None])
  cum_hop_weights = kth_path[2]
  parent_spur_node_idx = kth_path[3]
  return cum_hop_weights, parent_spur_node_idx


def update_prospects(sink,
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
                     lawler=False,
                     verbose=0):
  """Finds the replacement-paths of the (k-1)-th path and updates the prospects
  heap.

  - Replacement-paths search starts form the parent_spur_node_idx, as suggested
    by Lawler.
  - <k_paths> and <prospects> are Yen's a and B, respectively.
  """
  # This counter is used to print the spur-paths search difference produced
  # by Lawler's modification.
  spur_counter = 0
  if not lawler:
    parent_spur_node_idx = 0

  # Construct the deviation paths of the last found shortest path.
  # (u is the spur node)
  for i, u in enumerate(last_path[parent_spur_node_idx: -1]):
    spur_counter += 1

    u_idx = i + parent_spur_node_idx
    # Fail the (i, i + 1) edges of the found k - 1 shortest paths.
    failed_edges, _ = fail_found_spur_edges(adj_list,
                                            u,
                                            u_idx,
                                            last_path,
                                            k_paths)
    # Fail the root-path nodes from the to_visit PriorityQueue. Note that the
    # PriorityQueue should have been deepcopied.
    new_to_visit = copy.deepcopy(to_visit)
    del new_to_visit[last_path[:u_idx]]

    # Set the spur-node as source and initialize its cost to root-path-cost.
    new_to_visit[u] = [cum_hop_weights[u_idx], u, u]

    spur, prospect_cost, spur_weights, _ = dijkstra.unidirectional_dijkstra(
      adj_list,
      sink,
      new_to_visit,
      copy.deepcopy(visited),
      None,
      with_cum_hop_weights=True,
      verbose=verbose
    )

    if spur:
      prospect = last_path[:u_idx] + spur
      prospect_hop_weights = cum_hop_weights[:u_idx] + spur_weights

      push_prospect(prospect,
                    prospect_cost,
                    prospect_hop_weights,
                    u_idx,
                    K,
                    k,
                    prospects)

    # Restore the failed edges.
    reconnect_spur_edges(u, adj_list, failed_edges)

  if verbose >= 2:
    print(f"k: {k + 1:{len(str(K))}}   spur paths: {spur_counter}")
  return prospects
