#!/usr/bin/env python3
import math
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import matplotlib.pyplot as plt

from shortestpaths.graph_generator import random_graph


def main():
  n = 100
  directed = False
  weights_on = "edges-and-nodes"
  c = 0.3
  gradient = 0.5
  random_seed = 1
  if directed:
    m_max = n * (n - 1)
  else:
    m_max = n * (n - 1) // 2
  center = c * (n - 1)

  G_1, probs_1, edge_lengths_1, edge_lengths_true_1 = random_graph(
      n,
      directed=directed,
      weights_on=weights_on,
      random_seed=random_seed,
      gradient=gradient,
      center_portion=c,
      p_0=1,
      get_probability_distribution=True)
  m_1 = G_1.number_of_edges()

  G_2, probs_2, edge_lengths_2, edge_lengths_true_2 = random_graph(
      n,
      directed=directed,
      weights_on=weights_on,
      random_seed=random_seed,
      gradient=gradient,
      center_portion=c,
      p_0=0.7,
      get_probability_distribution=True)
  m_2 = G_2.number_of_edges()

  axes_title_fontsize = 12

  fig1, ax1 = plt.subplots(1, 3, figsize=(10, 5), dpi=200)
  # a
  ax1[0].scatter(edge_lengths_1, probs_1, s=10)
  ax1[0].set_ylim(0, 1.1)
  ax1[0].set_xlim(0, 110)
  # ax1[0].set_ylabel("p(x)", rotation=False, fontsize=10)
  ax1[0].set_title("a)  $\mathregular{p_0}$: 1.0" + f", λ: {gradient}, c: {c}",  # noqa: W605
                   fontsize=axes_title_fontsize)
  # b
  ax1[1].hist(edge_lengths_1, bins=range(1, 102))
  ax1[1].set_ylim(0, 110)
  ax1[1].set_xlim(0, 110)
  # complete graph edge distance frequency
  ax1[1].set_title("b)  f'(x) = 100 - x",
                   fontsize=axes_title_fontsize)
  # c
  ax1[2].hist(edge_lengths_true_1, bins=range(1, 102))
  ax1[2].set_ylim(0, 110)
  ax1[2].set_xlim(0, 110)
  ax1[2].set_title(f"c)  {m_1}/{m_max}", fontsize=axes_title_fontsize)

  fig2, ax2 = plt.subplots(1, 3, figsize=(10, 5), dpi=200)
  # d
  ax2[0].scatter(edge_lengths_2, probs_2, s=10)
  ax2[0].set_ylim(0, 1.1)
  ax2[0].set_xlim(0, 110)
  # ax2[0].set_ylabel("p(x)", rotation=False, fontsize=10)
  ax2[0].set_title("d)  $\mathregular{p_0}$: 0.7" + f", λ: {gradient}, c: {c}",  # noqa: W605
                   fontsize=axes_title_fontsize)
  # e
  expected_edge_weight_freq = [
    0.7 * (1 - 1 / (1 + math.exp(-gradient * (i - center))))
    * edge_lengths_2.count(i)
    for i in range(1, 102)
  ]
  ax2[1].bar(range(1, 102), expected_edge_weight_freq, width=1)
  ax2[1].set_ylim(0, 110)
  ax2[1].set_xlim(0, 110)
  ax2[1].set_xlabel("x = abs(head - tail)", fontsize=axes_title_fontsize)
  ax2[1].set_title((f"e)  {int(sum(expected_edge_weight_freq))}/{m_max}"
                    "\nf(x) = p(x)f'(x)"),
                   fontsize=axes_title_fontsize, y=0.93, linespacing=1.8)
  # f
  ax2[2].hist(edge_lengths_true_2, bins=range(1, 102))
  ax2[2].set_ylim(0, 110)
  ax2[2].set_xlim(0, 110)
  ax2[2].set_title(f"f)  {m_2}/{m_max}",
                   fontsize=axes_title_fontsize)

  for ax in [ax1, ax2]:
    for j in [0, 1, 2]:
      ax[j].tick_params(axis='both', which='major', labelsize=12)

  fig1.savefig("graph_model_vis_1.png")
  fig2.savefig("graph_model_vis_2.png")


if __name__ == '__main__':
  main()
