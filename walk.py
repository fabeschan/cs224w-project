import snap
import parser
import numpy as np
import random
from srw import transition_matrix, features
from srw.adjacency_matrix import adjacency_matrix
from make_graphs import save_graph_data, load_graph_data, make_graph

def bfs(graph, root, depth):
    nodes = set()
    for i in range(depth):
        NodeVec = snap.TIntV()
        snap.GetNodesAtHop(graph, root, i, NodeVec, False)
        for x in NodeVec:
            nodes.add(x)
    return nodes

def subgraph(g, root, depth):
    NIdV = snap.TIntV()
    for i in bfs(g, root, depth):
        NIdV.Add(i)
    sg = snap.GetSubGraph(g, NIdV)
    return sg

def relabel_subgraph(subgraph):
    '''
    construct new subgraph with nodeIDs from 0 to subgraph.GetNodes()
    return new subgraph and dict[old_nodeID->new_nodeID]
    '''

    new_graph = snap.TNGraph.New(subgraph.GetNodes(), subgraph.GetEdges())

    d = {}
    q = {} #reverse d
    size = 0
    for n in subgraph.Nodes():
        nid = n.GetId()
        if nid not in d:
            new_id = size
            size += 1
            d[new_id] = nid
            q[nid] = new_id
            new_graph.AddNode(new_id)

    for e in subgraph.Edges():
        src, dst = e.GetSrcNId(), e.GetDstNId()
        new_graph.AddEdge(q[src], q[dst])

    return new_graph, d

def remove_edges(graph, nid, proportion, removeout=True, removein=False):
    ''' remove a portion of nid's out edges and return the removed edges '''
    out_neighbors = [ id for id in graph.GetNI(nid).GetOutEdges() ]
    out_removed = random.sample(out_neighbors, int(proportion * len(out_neighbors)))

    in_neighbors = [ id for id in graph.GetNI(nid).GetInEdges() ]
    in_removed = random.sample(in_neighbors, int(proportion * len(in_neighbors)))

    removed = []
    if removeout:
        for r in out_removed:
            graph.DelEdge(nid, r)
        removed += out_removed
        print 'removed [{}] out edges'.format(len(out_removed))
    if removein:
        for r in in_removed:
            graph.DelEdge(r, nid)
        removed += in_removed
        print 'removed [{}] in edges'.format(len(in_removed))
    return removed

def score(predicted, truth):
    count = 0.0
    for p in predicted:
        if p in truth:
            count += 1.0
    return count / len(truth)

if __name__ == '__main__':
    filenames = [ "0301/{}.txt".format(i) for i in range(0, 4) ]
    data = parser.Data(filenames)
    graph = make_graph(data)

    ROOT = 8

    subg = subgraph(graph, root=ROOT, depth=4)
    print 'num nodes and edges of subgraph:', subg.GetNodes(), subg.GetEdges()

    # extract subgraph
    graph, labels = relabel_subgraph(subg)

    # remove a portion of ROOT's edges
    removed_edges = remove_edges(graph, ROOT, 0.1, True, True)

    # set up feature extractor
    fx = features.FeatureExtractor(labels, data)

    # set up adj matrix and trans matrix
    am = adjacency_matrix(graph)
    fm = transition_matrix.feature_matrix(graph, fx)

    # set up initial p and w
    p = np.zeros([graph.GetNodes()])
    p[ROOT] = 1.0
    w = np.random.normal(size=[fx.NUM_FEATURES])

    for i in range(200):
        p_new = transition_matrix.pagerank_one_iter(p, w, fm, am)
        w_new = transition_matrix.gradient_descent_step(p, w, fm, am)

        if np.sum(np.abs(w_new - w)) < 10 ** -10 and np.sum(np.abs(p_new - p)) < 10 ** -8:
            p = p_new
            w = w_new
            break

        p = p_new
        w = w_new
    print 'ran [{}] iterations'.format(i)

    print 'p:', p
    print 'w:', w

    p_enum = [ (p[i], i) for i in range(len(p)) ]
    p_enum.sort(key=lambda x: x[0], reverse=True)
    print p_enum
    print 'compare'
    p_1 = [ p[1] for p in p_enum ]
    p_2 = [ p for p in p_1 if not graph.IsEdge(ROOT, p) ]
    print 'len p_2:', len(p_2)
    p_3 = p_2[:len(removed_edges)]
    print 'predicted:', p_3
    print 'truth:', removed_edges
    print 'score:', score(p_3, removed_edges)
