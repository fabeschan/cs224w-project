import snap
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import math
import numpy as np

G = snap.LoadEdgeList(snap.PNGraph, "test-graph.txt", 0, 1)

print "Num nodes: %d; Num Edges: %d"%(G.GetNodes(), G.GetEdges())
print "getting graph diamaters..."
# Get full diameter for directed and undirected
print "Approx. full diameter (directed): %d"%(snap.GetBfsFullDiam(G, 100, True))
print "Approx. full diameter (undirected): %d"%(snap.GetBfsFullDiam(G, 100, False))

print "getting WCC size distribution..."
# Get WCC size distribution
ComponentDist = snap.TIntPrV()
snap.GetWccSzCnt(G, ComponentDist)
size = []
counts = []
print "WCC counts"
for comp in ComponentDist:
	size.append(comp.GetVal1())
	counts.append(comp.GetVal2())
	print "Size: %d Count: %d"%(comp.GetVal1(), comp.GetVal2())

plt.clf()
plt.figure()
plt.plot(size, counts, '.')

ComponentDist2 = snap.TIntPrV()
snap.GetWccSzCnt(G, ComponentDist2)
print "SCC counts"
for comp in ComponentDist2:
	print "Size: %d Count: %d"%(comp.GetVal1(), comp.GetVal2())

plt.title("Youtube Video WCC Size Distribution")
plt.xlabel("WCC Size")
plt.ylabel("Number of WCC of given size")
plt.savefig("wcc-distr3.pdf")

snap.PlotWccDistr(G, "wcc-distr3", "Directed Related Video Graph - WCC distribution")

print "getting SCC size distribution..."
snap.PlotSccDistr(G, "scc-distr3", "Directed Related Video Graph - SCC distribution")