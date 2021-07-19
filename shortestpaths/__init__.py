# __init__.py is part of ShortestPaths
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
"""Special project variables & API"""

from shortestpaths.api import k_shortest_paths, replacement_paths
from shortestpaths.post import print_paths, plot_paths
from shortestpaths.graph import random_graph, adj_list_reversed


__name__ = 'shortestpaths'
__version__ = '1.1.2'
__author__ = 'Athanasios Mattas'
__author_email__ = 'thanasismatt@gmail.gr'
__description__ = "Bidirectional replacement paths and k-shortest paths search with dynamic programming"
__url__ = 'https://github.com/ThanasisMattas/shortestpaths.git'
__license__ = 'GNU General Public License v3'
__copyright__ = 'Copyright 2020 Athanasios Mattas'
