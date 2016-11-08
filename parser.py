import collections
import logging

fieldnames = [
    'videoid',
    'uploader',
    'age',
    'category',
    'length',
    'views',
    'rate',
    'ratings',
    'comments',
    'related'
]

Datum = collections.namedtuple('Datum', fieldnames)

def extract_fields(line):
    raw_fields = [ s.strip() for s in line.split('\t') ]
    if len(raw_fields) < 9:
        #log len(raw_fields), raw_fields TODO
        return None
    else:
        fields = [
            raw_fields[0],
            raw_fields[1],
            int(raw_fields[2]),
            raw_fields[3],
            int(raw_fields[4]),
            int(raw_fields[5]),
            float(raw_fields[6]),
            int(raw_fields[7]),
            int(raw_fields[8]),
            tuple(raw_fields[9:])
        ]
        d = Datum(*fields)
        return d

class Data(object):
    def __init__(self, filenames):
        parsed = []
        for filename in filenames:
            with open(filename, 'r') as f:
                lines = f.readlines()

            # careful: extract_fields can return None so we filter them out
            parsed += filter(lambda x: x is not None, map(extract_fields, lines))

        self.size = 0
        self.lookup = { x.videoid: x for x in parsed } # videoid -> fields
        self.nodeid = {} # videoid -> nodeid
        self.videoid = {} # nodeid -> videoid
        self.categories = { x.category for x in parsed }

        for x in parsed:
            vid = x.videoid
            for v in list(x.related) + [vid]:
                if v not in self.nodeid:
                    self.nodeid[v] = self.size
                    self.videoid[self.size] = v
                    self.size += 1


if __name__ == '__main__':
    filenames = [ "0301/{}.txt".format(i) for i in range(0, 3) ]
    data = Data(filenames)

