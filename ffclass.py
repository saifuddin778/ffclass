import sys
sys.dont_write_bytecode = True
import random
import math
from methods_ import Functions
import numpy
funcs_ = Functions()

class ffclass(object):
    def __init__(self, data):
        self.data = data
        self.tdata = zip(*self.data)
        self.labels = self.tdata[-1]
        self.scorer = {}
        self.process()

    def get_histo(self, key, index):
        gq = {}
        column = self.tdata[index]
        for i in range(0, len(column)):
            if column[i] == key:
                if gq.has_key(self.labels[i]):
                    gq[self.labels[i]] += 1
                else:
                    gq[self.labels[i]] = 1
        return gq
    
    def process(self):
        for i, a in enumerate(self.tdata):
            if i < len(self.tdata)-1:
                self.scorer[i] = {}
                for j in self.tdata[i]:
                    self.scorer[i][j] = self.get_histo(j, i)

    def query(self, item):
        counter = dict([(l, 0) for l in self.labels])
        for i in range(0, len(item)):
            result = self.scorer[i][sorted(self.scorer[i].keys(), key=lambda n: abs(n-item[i]))[0]]
            counter[max(result, key=result.get)] += 1
        
        return max(counter, key=counter.get)
        
class forest_fclass(object):
    def __init__(self, data):
        self.data = data
        random.shuffle(data, random=numpy.random.rand)
        self.labels_ = [a[-1] for a in data]
        self.sets = self.get_chunks(self.data, len(self.data)/20)
        self.classifiers = {}
        self.build_classifiers()
    
    def get_chunks(self, set_, n):
        if n < 1:
            n = 1
        return [set_[i:i + n] for i in xrange(0, len(set_), n)]
    
    def build_classifiers(self):
        distribution = {}
        for a in self.labels_:
            if distribution.has_key(a):
                distribution[a] += 1
            else:
                distribution[a] = 1

        print "Class Distribution: %s" % distribution
        print "To have an optimal performance, equal number of samples should be provided for each class.."
        print "processing.."
        
        for i in range(0, len(self.sets)):
            self.classifiers[i] = ffclass(self.sets[i])
        return self.classifiers

    def predict(self, v):
        results = {}
        for a in self.classifiers:
            result = self.classifiers[a].query(v)
            if results.has_key(result):
                results[result] += 1
            else:
                results[result] = 1
        return max(results, key=results.get)
