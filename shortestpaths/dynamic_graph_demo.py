

@time_this
def _adapted_path(path,
                  path_cost,
                  memoizing_states,
                  tape,
                  adj_list,
                  source,
                  sink,
                  disconnected_nodes=[],
                  random_seed=None):
  if isinstance(path[0], (list, tuple)):
    path = list(list(zip(*path))[0])
  paths_data = [[path, path_cost, []]]

  # In case of disconnected_nodes are not provided, pick a random node towards
  # the sink.
  if not disconnected_nodes:
    random.seed(random_seed)
    disconnected_nodes = [
      path[random.randrange(len(path) // 2, len(path) - 1)]
    ]

  [source, sink] = utils.check_nodal_connection([source, sink],
                                                adj_list,
                                                disconnected_nodes)

  if memoizing_states:
    visited_nodes = list(tape.keys())
    # Retrieve the algorithm state that corresponds to the previous step.
    for i, visited_node in enumerate(visited_nodes):
      if visited_node in disconnected_nodes:
        checkpoint_node = visited_nodes[i - 1]
        break
    to_visit, visited = tape[checkpoint_node]

    # Disconnect the node
    del to_visit[disconnected_nodes]
  else:
    # Build a new PriorityQueue
    to_visit, visited = _dijkstra_init(len(adj_list), source)
    visited_nodes = None
    checkpoint_node = None

    # Disconnect the node
    del to_visit[disconnected_nodes]

  # Continue with the algorithm execution
  new_dijkstra_output, _ = _dijkstra(adj_list,
                                     sink,
                                     to_visit,
                                     visited,
                                     memoizing_states=False,
                                     avoided_nodes=disconnected_nodes)

  new_path = utils.extract_path(new_dijkstra_output, source, sink)
  new_path_cost = new_dijkstra_output[sink][0]
  if new_path:
    paths_data.append([new_path, new_path_cost, disconnected_nodes])

  return paths_data, visited_nodes, checkpoint_node


