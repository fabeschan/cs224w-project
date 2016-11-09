import snap
import parser, make_graphs

filenames = [ "0301/{}.txt".format(i) for i in range(0, 2) ]
data = parser.Data(filenames)
graph = make_graphs.make_graph(data)
ugraph = snap.ConvertGraph(snap.PUNGraph, graph)


# PageRanks

PRankH = snap.TIntFltH()
snap.GetPageRank(Graph, PRankH)


def getTop(PRankH):
    
    sorted_PRankH = sorted(PRankH, key = lambda key: PRankH[key], reverse = True)
    
    return sorted_PRankH[0:3]


print "3. Top 3 most central nodes are", getTop(PRankH)

NIdHubH = snap.TIntFltH()
NIdAuthH = snap.TIntFltH()
snap.GetHits(Graph, NIdHubH, NIdAuthH)

print "4. Top 3 hubs are", getTop(NIdHubH),
print "and top 3 authorities are", getTop(NIdAuthH)