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

NId1 = snap.GetMxDegNId(Graph)
NIdToDistH = snap.TIntH()
shortestPath = snap.GetShortPath(Graph, NId1, NIdToDistH)
shortestDist = {}
for item in NIdToDistH:
    shortestDist[item]=NIdToDistH[item]

PRankH = snap.TIntFltH()
snap.GetPageRank(Graph, PRankH)
    
simRanks = {}    
    
def simRank(Graph, nIters, gamma):

    for Node1 in Graph.Nodes(): 
        n1 = Node1.GetId()
        for Node2 in Graph.Nodes(): 
            n2 = Node1.GetId()
            if n2 < n1:
                continue
            simRanks[(n1,n2)] = 1.0
            
    for i in range(nIters):
        for Node1 in Graph.Nodes(): 
            n1 = Node1.GetId()
            for Node2 in Graph.Nodes(): 
                n2 = Node1.GetId()
                if n2 < n1:
                    continue
                runsum = 0.0
                for n3 in Node1.GetOutEdges():
                    for n4 in Node2.GetOutEdges():
                        if n4 < n3:
                            continue
                        runsum += simRanks[(n3,n4)]
                l1 = len(Set(Node1.GetOutEdges()))
                l2 = len(Set(Node2.GetOutEdges()))
                simRanks[(n1,n2)] = (2.0 * runsum)/(l1 * l2)
                
                
def rootedPageRank(Graph, nIters, root, beta):

    rPRs = {}
    
    for Node1 in Graph.Nodes(): 
        n1 = Node1.GetId()
        rPRs[n1] = 1.0
            
    for i in range(nIters):
        print i
        for Node1 in Graph.Nodes(): 
            n1 = Node1.GetId()
            rPRs[n1] = 0.0
            if n1 == root:
                rPRs[n1] += 1 - beta
            for n2 in Node1.GetOutEdges():
                Node2 = Graph.GetNI(n2)
                #l1 = len(Set(Node1.GetOutEdges()))
                l2 = len(Set(Node2.GetOutEdges()))
                rPRs[n1] += beta * (rPRs[n2]/l2) 
    s = 0
    for Node1 in Graph.Nodes(): 
        n1 = Node1.GetId()
        s += rPRs[n1]
    for Node1 in Graph.Nodes(): 
        n1 = Node1.GetId()    
        rPRs[n1] = rPRs[n1]/s
    return rPRs        
#simRank(Graph,100,0.8)
#print "Sims done"

start = time.time()
rPRs = rootedPageRank(Graph,10,1,0.85)
end = time.time()
print "done", end-start

def getScore(Graph,Node1,Node2,key):


    neigh1 = Set(Node1.GetOutEdges())
    neigh2 = Set(Node2.GetOutEdges())
    if key==1: # Common number of neighbours
        return len(neigh1.intersection(neigh2))
    if key==2: # Jaccard of sets of neighbours
        if len(neigh1.union(neigh2)) == 0:
            return 0
        return (len(neigh1.intersection(neigh2)) * 1.0)/len(neigh1.union(neigh2))
    if key==3: # Preferential Attachment
        return len(neigh1)*len(neigh2)
    if key==4: # Adamic/Adar score
        score = 0.0
        for z in neigh1.intersection(neigh2):
            Node3 = Graph.GetNI(z)
            score += 1.0/np.log(len(Set(Node3.GetOutEdges())))
        return score
    if key==5: # Graph Distance
        return -shortestDist[Node2.GetId()]
    if key==6: # PageRank
        return PRankH[Node1.GetId()] + PRankH[Node2.GetId()]
    if key==7: # Random
        return np.random.uniform()
    return 0
        
        
        
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
    
def PredictKey(Graph,key,limit):
    NId1 = snap.GetMxDegNId(Graph)
    Node3 = Graph.GetNI(NId1)
    print NId1, len(Set(Node3.GetOutEdges()))
    TrainGraph = copy_graph(Graph)
    TestGraph = copy_graph(Graph)
    print Graph.GetNodes(), Graph.GetEdges()

    for EI in Graph.Edges():
        src = EI.GetSrcNId()
        dest = EI.GetDstNId()
        p = np.random.uniform()
        if src!=NId1 and dest!=NId1:
            continue
        if p<0.2:
            TrainGraph.DelEdge(src,dest)
        else:
            TestGraph.DelEdge(src,dest)
    print Graph.GetNodes(), Graph.GetEdges()
    print TrainGraph.GetNodes(), TrainGraph.GetEdges()
    print TestGraph.GetNodes(), TestGraph.GetEdges()
    pairs = []
    n = 0
    lowest = 0
    scores = []
    for Node1 in TrainGraph.Nodes():
        n1 = Node1.GetId()
        if n1!=NId1:
            continue
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
    for i in range(limit):

        #print pairs[i]
        if TestGraph.IsEdge(pairs[i][1][0],pairs[i][1][1]):
            nTrue += 1
            
    return nTrue


#print PredictKey(Graph,1,100)
#print PredictKey(Graph,2,100)
#print PredictKey(Graph,3,100)
#print PredictKey(Graph,4,100)
#print PredictKey(Graph,5,1000)
#print PredictKey(snap.GetRndSubGraph(Graph, 10000),1,1000)
#print PredictKey(Graph,2,100)
#print PredictKey(snap.GetRndSubGraph(Graph, 10000),3,1000)
#print PredictKey(snap.GetRndSubGraph(Graph, 10000),4,1000)
    


def PredictKey1(Graph,key,limit,ktrain,ktest):
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
        if p<0.5:
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
            
    
    
    #print "Original Graph: ", Graph.GetNodes(), Graph.GetEdges()
    #print "Train Graph: ", TrainGraph.GetNodes(), TrainGraph.GetEdges()
    #print "Test Graph: ", TestGraph.GetNodes(), TestGraph.GetEdges()
    #print "Core Nodes: ", len(core)
    
    pairs = []
    n = 0
    lowest = 0
    scores = []
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
    for i in range(limit):

        #print pairs[i]
        if TestGraph.IsEdge(pairs[i][1][0],pairs[i][1][1]):
            nTrue += 1
            
    return nTrue
    
    
ktrain = 50
ktest = 50
limit = 100

print "Common number of neighbours Accuracy:", PredictKey1(Graph,1,limit,ktrain,ktest)
print "Jaccard of sets of neighbours Accuracy:", PredictKey1(Graph,2,limit,ktrain,ktest)
print "Preferential Attachment Accuracy:", PredictKey1(Graph,3,limit,ktrain,ktest)
print "Adamic/Adar score Accuracy:", PredictKey1(Graph,4,limit,ktrain,ktest)
print "Graph Distance Accuracy:", PredictKey1(Graph,5,limit,ktrain,ktest)
print "Random Accuracy:", PredictKey1(Graph,7,limit,ktrain,ktest)
