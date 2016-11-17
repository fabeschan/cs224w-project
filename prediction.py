import snap
import snap
import parser
import pickle
import collections
import logging
import numpy as np
import matplotlib.pyplot as plt
from sets import Set

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

    graph = snap.TNGraph.New(len(data.nodeid), len(edges))
    for nid in data.videoid:
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

def getScore(Graph,Node1,Node2,key):


    neigh1 = Set(Node1.GetOutEdges())
    neigh2 = Set(Node2.GetOutEdges())
    if key==1:
        return len(neigh1.intersection(neigh2))
    if key==2:
        return (len(neigh1.intersection(neigh2)) * 1.0)/len(neigh1.union(neigh2))
    if key==3:
        return len(neigh1)*len(neigh2)
    if key==4:
        score = 0
        for z in neigh1.intersection(neigh2):
            Node3 = Graph.GetNodeById(z)
            score += 1.0/np.log(len(Node3.GetOutEdges))
        return score
     
    return 0
        

TrainGraph = Graph
TestGraph = Graph

def PredictKey(Graph,key,limit):
    for EI in Graph.Edges():
        src = EI.GetSrcNId()
        dest = EI.GetDstNId()
        p = np.random.uniform()
        if p<0.2:
            TrainGraph.DelEdge(src,dest)
        else:
            TestGraph.DelEdge(src,dest)

    pairs = []
    n = 0
    lowest = 0
    scores = []
    print Graph.GetNodes(), Graph.GetEdges()
    for Node1 in TrainGraph.Nodes():
        n1 = Node1.GetId()
        for Node2 in TrainGraph.Nodes():
            n2 = Node2.GetId()
            
            if n1>=n2:
                continue
            if TrainGraph.IsEdge(n1,n2):
                continue
            #print n1,n2
            score = getScore(TrainGraph,Node1,Node2,key)
            if n < limit:
                pairs.append( [score,(n1,n2)] )
                scores.append(score)
                lowest = min(lowest,score)
                n += 1
            else:
                if score > lowest:
                    lowestloc = np.argmin(scores)
                    pairs[lowestloc] = [score,(n1,n2)]
                    scores[lowestloc] = score
                    lowest = min(scores)
                    
                

    pairs.sort(key=lambda x: -x[0])

    nTrue = 0
    for i in range(100):

        print pairs[i]
        if TestGraph.IsEdge(pairs[i][1][0],pairs[i][1][1]):
            nTrue += 1
            
    return nTrue



print PredictKey(snap.GetRndESubGraph(Graph, 10000),1,100)
#print PredictKey(Graph,2,100)
print PredictKey(Graph,3,100)
print PredictKey(Graph,4,100)
    
    
    

