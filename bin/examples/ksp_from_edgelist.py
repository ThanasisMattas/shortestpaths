#!/usr/bin/env python3
import networkx as nx
import shortestpaths as sp


def main():
  G = nx.read_weighted_edgelist(
    "example.edgelist",
    nodetype=int,
    create_using=nx.DiGraph
  )
  k_paths = sp.k_shortest_paths(G, 1, 200, k=5)
  sp.print_paths(k_paths)
  sp.plot_paths(k_paths, G)


if __name__ == '__main__':
  main()
