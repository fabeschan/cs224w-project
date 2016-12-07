import snap
import numpy as np
from sklearn import linear_model
import parser # have local parse copy

# Constants
numExamples = 30 # min num samples in train and test to be a core node
depth = 3
#numSamples = 75000 # per class; subset the training examples

# get features
userId = {} # maps username to unique id
userIdNext = 0

def getFeatureVector(Gtrain, data, n1, n2, userId):
        global userIdNext
	srcVid = data.videoid[n1]
	destVid = data.videoid[n2]
	if srcVid not in data.lookup or destVid not in data.lookup or data.lookup[srcVid] == None or data.lookup[destVid] == None:
		return None #continue # no valid data

	newRow = []
	if data.lookup[srcVid][0] not in userId:
		userId[data.lookup[srcVid][0]] = userIdNext
		userIdNext += 1
	if data.lookup[destVid][0] not in userId:
		userId[data.lookup[destVid][0]] = userIdNext
		userIdNext += 1

	newRow.append(data.lookup[srcVid][2]) # data.lookup[videoid][field index]
	newRow.append(data.lookup[destVid][2]) # age
	newRow.append(categoryId[data.lookup[srcVid][3]])
	newRow.append(categoryId[data.lookup[destVid][3]]) # category
	newRow.append(data.lookup[srcVid][4])
	newRow.append(data.lookup[destVid][4]) # length
	newRow.append(data.lookup[srcVid][6])
	newRow.append(data.lookup[destVid][6]) # rate
	newRow.append(Gtrain.GetNI(src).GetDeg())
	newRow.append(Gtrain.GetNI(dest).GetDeg()) # degree
	#uploader as feature - tendency for related videos between same user
	newRow.append(userId[data.lookup[srcVid][0]])
	newRow.append(userId[data.lookup[destVid][0]])
	return newRow

def getGoldLabel(G, n1, n2):
	if G.IsEdge(n1, n2):
		 return 1
	return 0

# read in graph twice
# one will be the gold graph and the other the train graph
print "Parsing text data..."
labels = [] #maps row number to gold label
filenames = [ "0301/{}.txt".format(i) for i in range(0, depth+1) ] # change this for which data files to parse
data = parser.Data(filenames)
print "Reading in edge list for depth %d..."%(depth)
G = snap.LoadEdgeList(snap.PUNGraph, "test-graph.txt", 0, 1) # change this for which graph edges to load
Gtrain = snap.LoadEdgeList(snap.PUNGraph, "test-graph.txt", 0, 1)

categoryId = {} # maps category name to unique integer
counter = 0
print "Generating category numbers..."
for category in data.categories:
	categoryId[category] = counter
	counter += 1

userIdNext = 0
userId = {}


# get train and test split
trainFeatures = []
testFeatures = []
trainLabels = []
testLabels = []

# for every edge, 15% time remove edge from train graph (and add to test set)
print "Generating training graph..."
for e in G.Edges():
	src = e.GetSrcNId()
	dest = e.GetDstNId()
	p = np.random.uniform()
	if p < 0.2:
		Gtrain.DelEdge(src, dest)

# find the core nodes
core = []
print "Finding core nodes... (k >= %d)"%(numExamples)
for node in Gtrain.Nodes():
	if node.GetDeg() >= numExamples:
		core.append(node.GetId())
for node in core:
	ni = G.GetNI(node)
	if ni.GetDeg() < numExamples:
		core.remove(node)
print "Found %d core nodes."%(len(core))

# with 70% chance to use a negative train example (and some % chance of being in test set)
# maybe consider only looking at edges and no-edge examples between core nodes
print "Generating feature matrix for training graph..."
trainNegSet = set() #check if example in pos training set if is edge
testNegSet = set()
print "Num edges = num pos/neg examples: %d"%(Gtrain.GetEdges())
#print "Num samples: %d"%(numSamples)
numTotalPos = 0 # all pos examples in gold graph core nodes
numTotalNeg = 0 # all neg examples in gold graph core nodes
numTestPos = 0 # num true pos test cases
numTestNeg = 0 # num true neg test cases
numTrainPos = 0 # num true pos train cases
numTrainNeg = 0 # num true neg train cases

for src in core:
	for dest in G.GetNI(src).GetOutEdges():
		n1 = src
		n2 = dest
		if src > dest and dest in core:
			continue
		elif src > dest and dest not in core:
			n1 = dest
			n2 = src
		if Gtrain.IsEdge(n1, n2):
			# add to train set
			newRow = getFeatureVector(Gtrain, data, n1, n2, userId)
			if newRow == None:
                                continue
			trainFeatures.append(newRow)
			trainLabels.append(getGoldLabel(G, n1, n2))
			numTotalPos += 1
			numTrainPos += 1
			# add a negative training example
			rnd = Gtrain.GetRndNId()
			tries = 0
			while G.IsEdge(src, rnd) or (src, rnd) in trainNegSet or (rnd, src) in trainNegSet or (src, rnd) in testNegSet or (rnd, src) in testNegSet or tries < 100:
				rnd = Gtrain.GetRndNId()
				tries += 1
			if tries > 100:
				continue
			#numTrainNeg += 1
			if src < rnd:
				newRow = getFeatureVector(Gtrain, data, src, rnd, userId)
                                if newRow == None:
                                        continue
				trainNegSet.add((src, rnd))
				trainFeatures.append(newRow)
				trainLabels.append(getGoldLabel(G, src, rnd))
				numTrainNeg += 1
			else:
                                newRow = getFeatureVector(Gtrain, data, rnd, src, userId)
                                if newRow == None:
                                        continue
				trainNegSet.add((rnd,src))
				trainFeatures.append(newRow)
				trainLabels.append(getGoldLabel(G, rnd, src))
		elif G.IsEdge(n1, n2):
			# add to test set
			testFeatures.append(getFeatureVector(Gtrain, data, n1, n2, userId))
			testLabels.append(getGoldLabel(G, n1, n2))
			numTotalPos += 1
			numTestPos += 1 #numTestPos should = numTestPosEx
			# add a negative test example
			rnd = Gtrain.GetRndNId()
			tries = 0
			while G.IsEdge(src, rnd) or (src, rnd) in trainNegSet or (rnd, src) in trainNegSet or (src, rnd) in testNegSet or (rnd, src) in testNegSet or tries < 100:
				rnd = Gtrain.GetRndNId()
				tries += 1
			if tries > 100:
				continue
			if src < rnd:
				newRow = getFeatureVector(Gtrain, data, src, rnd, userId)
                                if newRow == None:
                                        continue
				testNegSet.add((src, rnd))
				testFeatures.append(newRow)
				testLabels.append(getGoldLabel(G, src, rnd))
				numTestNeg += 1
			else:
                                newRow = getFeatureVector(Gtrain, data, rnd, src, userId)
                                if newRow == None:
                                        continue
				testNegSet.add((rnd,src))
				testFeatures.append(newRow)
				testLabels.append(getGoldLabel(G, n1, n2))
				numTestNeg += 1

print "Num pos training examples: %d"%(numTrainPos)
print "Num neg training examples: %d"%(numTrainNeg)

# convert to numpy
print "Converting to numpy arrays..."
featureMatrix = np.array(trainFeatures)
labelVector = np.array(trainLabels)

# train on every edge in train graph (using train graph features and gold graph label)
regr = linear_model.LogisticRegression(class_weight='balanced')
print "Training logistic regression model on depth %d..."%(depth)
regr.fit(featureMatrix, labelVector)
print('Coefficients: \n', regr.coef_)

# using predict, predict_proba, calculate precision and accuracy on train and test sets
# make sure score()'s accuracy is the same as the predict, manual calculated accuracy
testFeatureMatrix = np.array(testFeatures)
testLabelVector = np.array(testLabels)
print('Mean Accuracy Score (Train group): %f' % regr.score(featureMatrix, labelVector))
print('Mean Accuracy Score (Test group): %f' % regr.score(testFeatureMatrix, testLabelVector))

numCorrectPos = 0 # num true pos classified as pos test cases; false positives = numClassifiedPos - numCorrectPos
numCorrectNeg = 0 # num true neg classified as neg test cases
numClassifiedPos = 0 # num test cases classified as pos
numClassifiedNeg = 0 # num test cases classified as neg
numUnknownClassification = 0
testPrediction = regr.predict(testFeatureMatrix)
for i in xrange(len(testPrediction)):
	if testPrediction[i] == 0:
		numClassifiedNeg += 1
		if testLabelVector[i] == 0:
			numCorrectNeg += 1
	elif testPrediction[i] == 1:
		numClassifiedPos += 1
		if testLabelVector[i] == 1:
			numCorrectPos += 1
	else:
		numUnknownClassification += 1

print "Num correctly classified pos: %d"%(numCorrectPos)
print "Num classified pos in test: %d"%(numClassifiedPos)
print "Num true positives in test: %d"%(numTestPos)
print "Num correctly classified neg: %d"%(numCorrectNeg)
print "Num classified neg in test: %d"%(numClassifiedNeg)
print "Num true negatives in test: %d"%(numTestNeg)
print "Num unknown classifications: %d"%(numUnknownClassification)

print "------------------------------"
print "Precision (+): %f"%(1.0*numCorrectPos/numClassifiedPos)
print "Recall (+): %f"%(1.0*numCorrectPos/numTestPos)
print "Precision (-): %f"%(1.0*numCorrectNeg/numClassifiedNeg)
print "Recall (-): %f"%(1.0*numCorrectNeg/numTestNeg)
