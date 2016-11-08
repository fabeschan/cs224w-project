import snap

graph = snap.LoadEdgeList(snap.PUNGraph, "imdb_actor_edges.tsv", 0, 1, '\t')
mxwcc = snap.GetMxWcc(graph)
N = 20

# === 2.1 ===
s = []
for NI in mxwcc.Nodes():
    DegCentr = snap.GetDegreeCentr(mxwcc, NI.GetId())
    s.append((NI.GetId(), DegCentr))
s.sort(key=lambda x: x[1], reverse=True) # sort with max centrality at front
print '=== 2.1 ==='
for x in s[:N]:
    print x

# === 2.2 ===
Nodes = snap.TIntFltH()
Edges = snap.TIntPrFltH()
snap.GetBetweennessCentr(mxwcc, Nodes, Edges, 1.0)
s = [ (node, Nodes[node]) for node in Nodes ]
s.sort(key=lambda x: x[1], reverse=True) # sort with max centrality at front
print '=== 2.2 ==='
for x in s[:N]:
    print x

# === 2.3 ===
s = []
for NI in mxwcc.Nodes():
    CloseCentr = snap.GetClosenessCentr(mxwcc, NI.GetId())
    s.append((NI.GetId(), CloseCentr))
s.sort(key=lambda x: x[1], reverse=True) # sort with max centrality at front
print '=== 2.3 ==='
for x in s[:N]:
    print x

