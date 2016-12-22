import snap
import parser, make_graphs

filenames = [ "0301/{}.txt".format(i) for i in range(0, 3) ]
data = parser.Data(filenames)
graph = make_graphs.make_graph(data)
ugraph = snap.ConvertGraph(snap.PUNGraph, graph)
mxwcc = snap.GetMxWcc(graph)
umxwcc = snap.GetMxWcc(ugraph)
N = 20

# === GetDegreeCentr ===
s = []
for NI in umxwcc.Nodes():
    DegCentr = snap.GetDegreeCentr(umxwcc, NI.GetId())
    s.append((NI.GetId(), DegCentr))
s.sort(key=lambda x: x[1], reverse=True) # sort with max centrality at front
print '=== GetDegreeCentr ==='
with open("GetDegreeCentr-0-2.txt", 'w') as f:
    for x in s:
        f.write("{} {}\n".format(*x))

# === GetBetweennessCentr ===
Nodes = snap.TIntFltH()
Edges = snap.TIntPrFltH()
snap.GetBetweennessCentr(mxwcc, Nodes, Edges, 1.0)
s = [ (node, Nodes[node]) for node in Nodes ]
s.sort(key=lambda x: x[1], reverse=True) # sort with max centrality at front
print '=== GetBetweennessCentr ==='
with open("GetBetweennessCentr-0-2.txt", 'w') as f:
    for x in s:
        f.write("{} {}\n".format(*x))

# === GetClosenessCentr ===
s = []
for NI in mxwcc.Nodes():
    CloseCentr = snap.GetClosenessCentr(mxwcc, NI.GetId())
    s.append((NI.GetId(), CloseCentr))
s.sort(key=lambda x: x[1], reverse=True) # sort with max centrality at front
print '=== GetClosenessCentr ==='
with open("GetClosenessCentr-0-2.txt", 'w') as f:
    for x in s:
        f.write("{} {}\n".format(*x))

