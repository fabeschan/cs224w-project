import snap
import snap
import parser
import pickle
import collections
import logging
import numpy as np
import matplotlib.pyplot as plt
from sets import Set
import time

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

                    
                    
#filenames = [ "0301/{}.txt".format(i) for i in range(0, 4) ]
#data = Data(filenames)
#graph = make_graph(data)
#save_graph_data(data, graph, "try")
data,graph = load_graph_data("try")
Graph = snap.ConvertGraph(snap.PUNGraph, graph)

shortestDist = {}
# print Graph.GetNodes()

def getWeight(Graph,n1,n2):
    return 1.0

def propflow(Graph, root, l):
    scores = {}
    
    n1 = root
    found = [n1]
    newSearch = [n1]
    scores[n1]=1.0
    
    for currentDegree in range(0,l+1):
        oldSearch = list(newSearch)
        newSearch = []
        while len(oldSearch) != 0:
            n2 = oldSearch.pop()
            nodeInput = scores[n2]
            sumOutput = 0.0
            Node2 = Graph.GetNI(n2)
            for n3 in Node2.GetOutEdges():
                sumOutput += getWeight(Graph,n2,n3)
            flow = 0.0
            for n3 in Node2.GetOutEdges():
                wij = getWeight(Graph,n2,n3)
                flow = nodeInput * (wij*1.0/sumOutput)
                if n3 not in scores:
                    scores[n3]=0.0
                scores[n3] += flow
            if n2 not in found:
                found.append(n2)
                newSearch.append(n2)
    return scores
    
    
#print propflow(Graph,1,2)

def copy_graph(graph):
    tmpfile = '.copy.bin'

    # Saving to tmp file
    FOut = snap.TFOut(tmpfile)
    graph.Save(FOut)
    FOut.Flush()

    # Loading to new graph
    FIn = snap.TFIn(tmpfile)
    graphtype = type(graph)
    new_graph = graphtype.New()
    new_graph = new_graph.Load(FIn)

    return new_graph

def PredictKey(Graph,limit,ktrain,ktest,l):
    NId1 = snap.GetMxDegNId(Graph)
    Node3 = Graph.GetNI(NId1)
    TrainGraph = copy_graph(Graph)
    TestGraph = copy_graph(Graph)

    
    for EI in Graph.Edges():
        src = EI.GetSrcNId()
        dest = EI.GetDstNId()
        p = np.random.uniform()
        #if src!=NId1 and dest!=NId1:
        #    continue
        if p<0.2:
            TrainGraph.DelEdge(src,dest)
        else:
            TestGraph.DelEdge(src,dest)
            
    core = []

    for Node1 in TrainGraph.Nodes():
        neigh1 = len(Set(Node1.GetOutEdges()))
        if neigh1 >= ktrain:
            core.append(Node1.GetId())
    
    for node in core:
        Node1 = TestGraph.GetNI(node)
        neigh1 = len(Set(Node1.GetOutEdges()))
        if neigh1 < ktest:
            core.remove(node)

    limit = 0
    for Node1 in TrainGraph.Nodes():
        n1 = Node1.GetId()
        if n1 not in core:
            continue
        for Node2 in TrainGraph.Nodes():
            n2 = Node2.GetId()    
            if n2 not in core:
                continue
            if n1>=n2:
                continue
            if TestGraph.IsEdge(n1,n2):
                limit += 1
    #print "Number of true core edges: ", limit
    #print "Original Graph: ", Graph.GetNodes(), Graph.GetEdges()
    #print "Train Graph: ", TrainGraph.GetNodes(), TrainGraph.GetEdges()
    #print "Test Graph: ", TestGraph.GetNodes(), TestGraph.GetEdges()
    #print "Core Nodes: ", len(core)
    global shortestDist
    shortestDist = {}
    pairs = []
    n = 0
    lowest = 0
    scores = []
    propflows = {}
    for Node1 in TrainGraph.Nodes():
        n1 = Node1.GetId()
        if n1 not in core:
            continue
        for Node2 in TrainGraph.Nodes():
            n2 = Node2.GetId()    
            if n2 not in core:
                continue
            if n1>=n2:
                continue
            if TrainGraph.IsEdge(n1,n2):
                continue
            #print n1,n2
            if n1 in propflows:
                propscores = propflows[n1]
            else:
                propscores = propflow(TrainGraph,n1,l)
                #print propscores
                propflows[n1] = propscores
            score = 0
            if n2 in propscores:
                score = propscores[n2]
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
    print pairs
    nTrue = 0
    for i in range(limit):

        #print pairs[i]
        if TestGraph.IsEdge(pairs[i][1][0],pairs[i][1][1]):
            nTrue += 1
            
    return [nTrue*100.0/limit,limit]

for ktrain in range(60,20,-5):
    #ktrain = 60
    #ktest = 60
    limit = 100
    ktest = ktrain
    print "kTrain/kTest", ktrain,ktest
    print PredictKey(Graph,limit,ktrain,ktest,20)
    
