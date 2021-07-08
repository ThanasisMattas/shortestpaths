#!/usr/bin/env python3
"""Plots a 2d contour of graph density vs sigmoid center & initial probability.
(see shortestpaths/graph.py)
"""
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import click

from shortestpaths import post


@click.command()
@click.argument("n", type=click.INT)
@click.option('-g', "--graphs-per-measure", type=click.INT,
              default=10, show_default=True,
              help=("Number of graphs to generate for each measure, in order"
                    " to take the mean."))
@click.option("--directed", is_flag=True)
@click.option("--weights-on", default="edges-and-nodes", show_default=True,
              type=click.Choice(["edges", "nodes", "edges-and-nodes"],
                                case_sensitive=False))
def main(**kwargs):
  post.graph_density_contour(**kwargs)

if __name__ == '__main__':
  main()
