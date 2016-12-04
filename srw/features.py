import numpy as np

def indicator(x):
    return 1.0 if x else 0.0

class FeatureExtractor(object):
    def __init__(self, subgraph_labels, data):
        self.subgraph_labels = subgraph_labels
        self.data = data
        self.extractor = self.define_extractor()
        self.NUM_FEATURES = len(self.extractor)

    def define_extractor(self):
        ''' return a list of functions that correspond to a feature vector '''
        x = [
                lambda udata, vdata: 1,
                lambda udata, vdata: indicator(udata.uploader == vdata.uploader),
                lambda udata, vdata: indicator(udata.category == vdata.category),
                lambda udata, vdata: 100.0 / max(1, abs(udata.age - vdata.age)),
                lambda udata, vdata: 100.0 / max(1, abs(udata.ratings - vdata.ratings)),
                lambda udata, vdata: 100.0 / max(1, abs(udata.views - vdata.views)),
                lambda udata, vdata: 100.0 / max(1, abs(udata.length - vdata.length)),
            ]
        return x

    def feature_vector(self, u, v):
        uvid = self.data.videoid[self.subgraph_labels[u]]
        vvid = self.data.videoid[self.subgraph_labels[v]]

        udata = self.data.lookup[uvid]
        vdata = self.data.lookup[vvid]

        return np.array([ f(udata, vdata) for f in self.extractor ])

