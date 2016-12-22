import networkx as nx
from collections import defaultdict, deque
from random import choice, shuffle
import matplotlib.pyplot as plt

import snap
import parser, make_graphs

filenames = [ "0301/{}.txt".format(i) for i in range(0, 3) ]
data = parser.Data(filenames)

def make_graph(data, categories=None):
    if categories is None:
        categories = data.categories

    edges = []
    for vid, entry in data.lookup.items():
        if entry.category not in categories:
            continue
        for rid in entry.related:
            if rid in data.lookup and data.lookup[rid].category in categories:
                edges.append((data.nodeid[vid], data.nodeid[rid]))

    nodeids = set()
    for src_id, dst_id in edges:
        nodeids.add(src_id)
        nodeids.add(dst_id)

    graph = nx.DiGraph()
    for nid in data.videoid:
        graph.add_node(nid)

    for src_id, dst_id in edges:
        graph.add_edge(src_id, dst_id)

    return graph


g = make_graph(data)

nx.pagerank(g, p)
