import snap
import snap
import parser
import pickle
import collections
import logging
import numpy as np
import matplotlib.pyplot as plt

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

    graph = snap.TNGraph.New(len(nodeids), len(edges))
    for nid in nodeids:
        graph.AddNode(nid)

    for src_id, dst_id in edges:
        graph.AddEdge(src_id, dst_id)

    return graph
    

    
def save_graph_data(data, graph, prefix):
    snap.SaveEdgeList(graph, prefix + '-graph.txt')
    with open(prefix + '-data.pkl', 'wb') as f:
        pickle.dump(data, f)

def load_graph_data(prefix):
    graph = snap.LoadEdgeList(snap.PNGraph, prefix + '-graph.txt', 0, 1)
    with open(prefix + '-data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data, graph
    
    
fieldnames = [
    'videoid',
    'uploader',
    'age',
    'category',
    'length',
    'views',
    'rate',
    'ratings',
    'comments',
    'related'
]

Datum = collections.namedtuple('Datum', fieldnames)

def extract_fields(line):
    raw_fields = [ s.strip() for s in line.split('\t') ]
    if len(raw_fields) < 9:
        #log len(raw_fields), raw_fields TODO
        return None
    else:
        fields = [
            raw_fields[0],
            raw_fields[1],
            int(raw_fields[2]),
            raw_fields[3],
            int(raw_fields[4]),
            int(raw_fields[5]),
            float(raw_fields[6]),
            int(raw_fields[7]),
            int(raw_fields[8]),
            tuple(raw_fields[9:])
        ]
        d = Datum(*fields)
        return d

class Data(object):
    def __init__(self, filenames):
        parsed = []
        for filename in filenames:
            with open(filename, 'r') as f:
                lines = f.readlines()

            # careful: extract_fields can return None so we filter them out
            parsed += filter(lambda x: x is not None, map(extract_fields, lines))

        self.size = 0
        self.lookup = { x.videoid: x for x in parsed } # videoid -> fields
        self.nodeid = {} # videoid -> nodeid
        self.videoid = {} # nodeid -> videoid
        self.categories = { x.category for x in parsed }

        for x in parsed:
            vid = x.videoid
            for v in list(x.related) + [vid]:
                if v not in self.nodeid:
                    self.nodeid[v] = self.size
                    self.videoid[self.size] = v
                    self.size += 1

                    
                    
filenames = [ "0301/{}.txt".format(i) for i in range(0, 4) ]
data = Data(filenames)
graph = make_graph(data)
Graph = snap.ConvertGraph(snap.PUNGraph, graph)
save_graph_data(data, graph, "try")
data,graph = load_graph_data("try")
Graph = snap.ConvertGraph(snap.PUNGraph, graph)

GraphClustCoeff = snap.GetClustCf (Graph, -1)

print "Average clustering coefficient of the graph is ", GraphClustCoeff

for category in data.categories:
    graph1 = make_graph(data,[category])
    save_graph_data(data, graph1, "temp")
    data,graph1 = load_graph_data("temp")
    Graph1 = snap.ConvertGraph(snap.PUNGraph, graph1)
    print category, Graph1.GetNodes(), Graph1.GetEdges()
    GraphClustCoeff1 = snap.GetClustCf (Graph1, -1)
    print "Average clustering coefficient of the " +category + " graph is ", GraphClustCoeff1


V = Graph.GetNodes()
E = Graph.GetEdges()
print V,E
Erdos = snap.GenRndGnm(snap.PNGraph, V, E)
print "Erdos CC", snap.GetClustCf(Erdos,-1)

Rnd = snap.TRnd()
UGraph = snap.GenPrefAttach(V, 20, Rnd)
print "Pref Attachment CC", snap.GetClustCf(UGraph,-1)