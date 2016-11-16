import snap
import parser
import pickle

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

'''
if __name__ == '__main__':
    filenames = [ "0301/{}.txt".format(i) for i in range(0, 3) ]
    data = parser.Data(filenames)
    graph = make_graph(data)

    # example save and load
    #save_graph_data(data, graph, "test")
    #data, graph = load_graph_data("test")

    # example reading and processing
    for n in graph.Nodes():
        nid = n.GetId()
        vid = data.videoid[nid]
        if vid in data.lookup:
            print data.lookup[vid]
'''


filenames = [ "0301/{}.txt".format(i) for i in range(0, 3) ]
data = parser.Data(filenames)
graph = make_graph(data)
Graph = snap.ConvertGraph(snap.PUNGraph, graph)

GraphClustCoeff = snap.GetClustCf (Graph, -1)

print "Average clustering coefficient of the graph is ", GraphClustCoeff

for category in data.categories:
    graph1 = make_graph(data,[category])
    Graph1 = snap.ConvertGraph(snap.PUNGraph, graph1)
    print category, Graph1.GetNodes(), Graph1.GetEdges()
    GraphClustCoeff1 = snap.GetClustCf (Graph1, -1)
    print "Average clustering coefficient of the " +category + " graph is ", GraphClustCoeff1

