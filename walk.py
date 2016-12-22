import snap
import parser
import numpy as np
import random
from srw import transition_matrix, features, loss_function
from srw.adjacency_matrix import adjacency_matrix
from make_graphs import save_graph_data, load_graph_data, make_graph

def bfs(graph, roots, depth):
    nodes = set()
    for r in roots:
        for i in range(depth+1):
            NodeVec = snap.TIntV()
            snap.GetNodesAtHop(graph, r, i, NodeVec, False)
            for x in NodeVec:
                nodes.add(x)
    return nodes

def subgraph(g, roots, depth):
    NIdV = snap.TIntV()
    for i in bfs(g, roots, depth):
        NIdV.Add(i)
    sg = snap.GetSubGraph(g, NIdV)
    return sg

def relabel_subgraph(subgraph):
    '''
    construct new subgraph with nodeIDs from 0 to subgraph.GetNodes()
    return new subgraph and dict[new_nodeID->old_nodeID] and dict[old_nodeID->new_nodeID]
    '''

    new_graph = snap.TNGraph.New(subgraph.GetNodes(), subgraph.GetEdges())

    d = {}
    q = {} #reverse d
    size = 0
    for n in subgraph.Nodes():
        nid = n.GetId()
        if nid not in q:
            new_id = size
            size += 1
            d[new_id] = nid
            q[nid] = new_id
            new_graph.AddNode(new_id)

    for e in subgraph.Edges():
        src, dst = e.GetSrcNId(), e.GetDstNId()
        new_graph.AddEdge(q[src], q[dst])

    return new_graph, d, q

def remove_edges(graph, p_node, p_edge, min_outdeg=20, override=[]):
    '''
    remove a proportion p_edge of out edges from each of proportion p_node nodes
    and return the removed edges as a dict[nid->removed]

    '''
    q = [ n.GetId() for n in graph.Nodes() if n.GetOutDeg() >= min_outdeg ]
    removed = {}
    nodes = random.sample(q, int(p_node * len(q)))
    if override:
        nodes = override

    for nid in nodes:
        out_neighbors = [ id for id in graph.GetNI(nid).GetOutEdges() ]
        out_removed = random.sample(out_neighbors, int(p_edge * len(out_neighbors)))
        for r in out_removed:
            graph.DelEdge(nid, r)
        removed[nid] = out_removed
    #print 'removed:', removed
    return removed

def eval_score(predicted, truth, total):
    count = 0.0
    for p in predicted:
        if p in truth:
            count += 1.0
    score = count / len(truth)
    precision = count / len(predicted)
    recall = count / len(truth)
    return score, precision, recall

def evaluate(data, graph):
    ROOT = random.sample([ n.GetId() for n in graph.Nodes() if n.GetOutDeg() >= 20 ], 1)

    subg = subgraph(graph, roots=ROOT, depth=2)
    #print 'num nodes and edges of subgraph:', subg.GetNodes(), subg.GetEdges()

    # extract subgraph
    graph, labels, rlabels = relabel_subgraph(subg)

    # remove a portion of ROOT's edges
    newROOT = rlabels[ROOT[0]]
    #print [ (n.GetId(), n, n.GetOutDeg()) for n in graph.Nodes() ]
    removed = remove_edges(graph, p_node=0.05, p_edge=0.2, override=[newROOT])
    if len(removed[newROOT]) == 0: return None

    # set up feature extractor
    fx = features.FeatureExtractor(labels, data)

    # set up adj matrix and trans matrix
    am = adjacency_matrix(graph)
    fm = transition_matrix.feature_matrix(graph, fx)

    # set up initial p and w
    p = np.zeros([graph.GetNodes()])
    p[newROOT] = 1.0
    w = np.random.normal(size=[fx.NUM_FEATURES])

    brk = False
    for i in range(200):
        #print '=== iter {} ==='.format(i)
        p_new = transition_matrix.pagerank_one_iter(p, w, fm, am)
        w_new = transition_matrix.gradient_descent_step(p, w, fm, am, removed, newROOT)

        if np.sum(np.abs(w_new - w)) < 10 ** -120 and np.sum(np.abs(p_new - p)) < 10 ** -80:
            brk = True

        p = p_new
        w = w_new
        #print 'ran [{}] iterations'.format(i)

        #print 'total loss:', loss_function.total_loss(p, removed, newROOT)
        #print 'w:', w

        p_enum = [ (p[i], i) for i in range(len(p)) ]
        p_enum.sort(key=lambda x: x[0], reverse=True)
        p_1 = [ e[1] for e in p_enum ]
        p_2 = [ e for e in p_1 if not graph.IsEdge(newROOT, e) ]
        p_3 = p_2[:len(removed[newROOT])]
        #print 'predicted:', p_3
        #print 'truth:', removed[newROOT]
        score, precision, recall = eval_score(p_3, removed[newROOT], len(p_2))

        if brk:
            break
    return score, precision, recall

if __name__ == '__main__':
    filenames = [ "0301/{}.txt".format(i) for i in range(0, 4) ]
    data = parser.Data(filenames)
    graph = make_graph(data)
    #data, graph = load_graph_data(prefix='test')

    for k in range(1):
        s, p, r = evaluate(data, graph)
        print "{} {} {}".format(s, p, r)
