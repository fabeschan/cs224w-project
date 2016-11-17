import snap
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import math
import numpy as np

# will return out degree frequencies if outdeg is True, (will do in-deg otherwise)
def getInDegDistr(G, outdeg):
	degHistogram = snap.TIntPrV()
	if outdeg:
		snap.GetOutDegCnt(G, degHistogram)
	else:
		snap.GetInDegCnt(G, degHistogram)
	degDistr = [(pair.GetVal1(), pair.GetVal2()) for pair in degHistogram]
	degDistr = sorted(degDistr, key=lambda pair: pair[0], reverse=False)
	degrees = []
	counts = []
	for pair in degDistr:
		#first = degree
		degrees.append(pair[0])
		#second = #nodes of degree - normalize by total nodes to get proportion of nodes
		counts.append(1.0*pair[1]/G.GetNodes())
	return (degrees, counts)

G = snap.LoadEdgeList(snap.PNGraph, "test-graph.txt", 0, 1)


print "plotting in-degree distributions..."
# Plot degree distribution of youtube networks on log-log scale.
plt.clf()
plt.figure()
x1, y1 = getInDegDistr(G, False)
plt.plot(x1, y1, '.', label="Youtube")
plt.title("Youtube Video In-Degree Distribution")
plt.legend()
plt.xlabel("Degree")
plt.ylabel("Proportion of nodes with given degree")
plt.savefig("in-degree-distr.pdf")

plt.clf()
plt.figure()
plt.loglog(x1, y1, '.', label="Youtube (Log)")

plt.title("Youtube Video In-Degree Distribution (Log)")
plt.legend()
plt.xlabel("Degree in log scale")
plt.ylabel("Proportion of nodes with given degree in log scale")
plt.savefig("in-degree-distr-log.pdf")

print "plotting out-degree distributions..."
# Plot degree distribution of youtube networks on log-log scale.
plt.clf()
plt.figure()
x1, y1 = getInDegDistr(G, True)
plt.plot(x1, y1, '.', label="Youtube")
plt.title("Youtube Video Out-Degree Distribution")
plt.legend()
plt.xlabel("Degree")
plt.ylabel("Proportion of nodes with given degree")
plt.savefig("out-degree-distr.pdf")

plt.clf()
plt.figure()
plt.loglog(x1, y1, '.', label="Youtube (Log)")

plt.title("Youtube Video Out-Degree Distribution (Log)")
plt.legend()
plt.xlabel("Degree in log scale")
plt.ylabel("Proportion of nodes with given degree in log scale")
plt.savefig("out-degree-distr-log.pdf")