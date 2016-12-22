import parser, make_graphs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


if __name__ == '__main__':
    filenames = [ "0301/{}.txt".format(i) for i in range(0, 3) ]
    data = parser.Data(filenames)
    graph = make_graphs.make_graph(data)
    print graph.GetNodes()

    #centr_files = ['GetBetweennessCentr-0-2.txt', 'GetClosenessCentr-0-2.txt', 'GetDegreeCentr-0-2.txt']
    centr_files = ['GetDegreeCentr-0-2.txt']
    for centr in centr_files:
        with open(centr) as f:
            contents = f.readlines()
        split = [ l.strip().split(' ') for l in contents ]
        parsed = [ (int(nid), float(centr)) for nid, centr in split ]
        results = [ (data.lookup[data.videoid[nid]].age, centr) for nid, centr in parsed ]
        x, y = zip(*results)
        print y[:20]
        print x[:20]

        #plt.loglog(x, y, 'o')
        plt.title('Degree Centrality vs Age')
        plt.xlabel('age')
        plt.ylabel('centrality score')
        plt.scatter(x, y)
        plt.show()
