import snap
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import math
import numpy as np

# returns the average shortest path for all pairs of nodes
# assumes a directed graph
def avgShortestPath(G):
	avgPathDir = 0
	avgPathUndir = 0
	numDirPath = 0
	numUndirPath = 0
	for src in G.Nodes():
		NIdToDistH = snap.TIntH()
		shortestPathUndir = snap.GetShortPath(G, src.GetId(), NIdToDistH, False)
		numUndirPath += len(NIdToDistH)
		for item in NIdToDistH:
			avgPathUndir += 1.0*NIdToDistH[item]#/len(NIdToDistH)

		shortestPathDir = snap.GetShortPath(G, src.GetId(), NIdToDistH, True)
		numDirPath += len(NIdToDistH)
		for item in NIdToDistH:
			avgPathDir += 1.0*NIdToDistH[item]#/len(NIdToDistH)
	print "Avg. Shortest Path (directed): %f"%(1.0*avgPathDir/numDirPath)
	print "Avg. Shortest Path (undirected): %f"%(1.0*avgPathUndir/numUndirPath)

G = snap.LoadEdgeList(snap.PNGraph, "test-graph.txt", 0, 1)
print "getting shortest paths..."
avgShortestPath(G)