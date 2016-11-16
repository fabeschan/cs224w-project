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

                    
                    
filenames = [ "0301/{}.txt".format(i) for i in range(0, 3) ]
data = Data(filenames)
graph = make_graph(data)
Graph = snap.ConvertGraph(snap.PUNGraph, graph)


# PageRanks

PRankH = snap.TIntFltH()
snap.GetPageRank(Graph, PRankH)


def getTop(PRankH):
    
    sorted_PRankH = sorted(PRankH, key = lambda key: PRankH[key], reverse = True)
    return sorted_PRankH

topPRs = getTop(PRankH)

#print "Top 10 most central nodes are", topPRs, [data.lookup[data.videoid[i]] for i in topPRs]

NIdHubH = snap.TIntFltH()
NIdAuthH = snap.TIntFltH()
snap.GetHits(Graph, NIdHubH, NIdAuthH)

topH = getTop(NIdHubH)

topA = getTop(NIdAuthH)
#print "4. Top 3 hubs are", getTop(NIdHubH),
#print "and top 3 authorities are", getTop(NIdAuthH)

d1 = data.lookup[data.videoid[i]]

#print data._fields()
#fields = uploader='EA', age=742, category='Gadgets & Games', length=61, views=1128, rate=4.67, ratings=9, comments=6

def convertKeyToThing(t,key):
    if key=="age":
        return t.age
    if key=="categorgy":
        return t.category
    if key=="length":
        return t.length
    if key=="views":
        return t.views
    if key=="rate":
        return t.rate
    if key=="ratings":
        return t.ratings
    if key=="comments":
        return t.comments
    return None
    
def plot(tops, pageranks, key, filename,xlabel):
    x = []
    y = []
    for i in tops:
        if i not in data.videoid:
            continue
        if data.videoid[i]=='QQvBfH3ZcKQ':
            print data.videoid[i]
        d1 = data.lookup[data.videoid[i]]
        #print d1
        
        if convertKeyToThing(d1,key) > 10000000:
            continue
        x.append(pageranks[i])
        y.append(convertKeyToThing(d1,key))
        #print x,y
    plt.figure()

    #print x[1:10], y[1:10]
    plt.xlabel(xlabel)
    plt.ylabel(key)
    plt.title(key + " vs. " +xlabel)
    plt.plot(x,y, 'ro')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig("plots-links/log"+filename+".png")
    
    
plot(topPRs, PRankH, "age", "PR-age","PageRank")
plot(topPRs, PRankH, "length", "PR-length","PageRank")
plot(topPRs, PRankH, "views", "PR-views","PageRank")
plot(topPRs, PRankH, "rate", "PR-rate","PageRank")
plot(topPRs, PRankH, "ratings", "PR-ratings","PageRank")
plot(topPRs, PRankH, "comments", "PR-comments","PageRank")


plot(topH, NIdHubH, "age", "H-age","H")
plot(topH, NIdHubH, "length", "H-length","H")
plot(topH, NIdHubH, "views", "H-views","H")
plot(topH, NIdHubH, "rate", "H-rate","H")
plot(topH, NIdHubH, "ratings", "H-ratings","H")
plot(topH, NIdHubH, "comments", "H-comments","H")

plot(topA, NIdAuthH, "age", "A-age","A")
plot(topA, NIdAuthH, "length", "A-length","A")
plot(topA, NIdAuthH, "views", "A-views","A")
plot(topA, NIdAuthH, "rate", "A-rate","A")
plot(topA, NIdAuthH, "ratings", "A-ratings","A")
plot(topA, NIdAuthH, "comments", "A-comments","A")



    