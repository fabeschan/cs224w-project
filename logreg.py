import snap
import numpy as np
from sklearn import linear_model
import parser # have local parse copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def indicator(x):
	return 1.0 if x else 0.0

def getFeatureVector(Gtrain, data, n1, n2):
	srcVid = data.videoid[n1]
	destVid = data.videoid[n2]
	if srcVid not in data.lookup or destVid not in data.lookup or data.lookup[srcVid] == None or data.lookup[destVid] == None:
		return None #continue # no valid data

	newRow = []
	newRow.append(100.0/max(1, abs(data.lookup[srcVid][2] - data.lookup[destVid][2]))) #age
	newRow.append(indicator(data.lookup[srcVid][3] == data.lookup[destVid][3])) #category
	newRow.append(100.0/max(1, abs(data.lookup[srcVid][4] - data.lookup[destVid][4]))) #length
	newRow.append(100.0/max(1, abs(data.lookup[srcVid][5] - data.lookup[destVid][5]))) #views
	newRow.append(100.0/max(0.0001, abs(data.lookup[srcVid][6] - data.lookup[destVid][6]))) #rate (float)
	newRow.append(100.0/max(1, abs(data.lookup[srcVid][7] - data.lookup[destVid][7]))) #ratings
	newRow.append(100.0/max(1, abs(data.lookup[srcVid][8] - data.lookup[destVid][8]))) #comments
	newRow.append(indicator(data.lookup[srcVid][1] == data.lookup[destVid][1])) #uploader
	newRow.append(100.0/max(1, abs(Gtrain.GetNI(n1).GetDeg() - Gtrain.GetNI(n2).GetDeg()))) #training degree
	return newRow

def getGoldLabel(G, n1, n2):
	if G.IsEdge(n1, n2):
		 return 1
	return 0

def logreg(G, Gtrain, data, numExamples): # min num samples in train and test to be a core node)
	# get train and test split
	trainFeatures = []
	testFeatures = []
	trainLabels = []
	testLabels = []

	# find the core nodes
	core = []
	#print "Finding core nodes... (k >= %d)"%(numExamples)
	for node in Gtrain.Nodes():
		if node.GetDeg() >= numExamples:
			core.append(node.GetId())
	for node in core:
		ni = G.GetNI(node)
		if ni.GetDeg() < numExamples:
			core.remove(node)
        #print G.GetNodes()
        #print G.GetEdges()
        #print Gtrain.GetNodes()
        #print Gtrain.GetEdges()
        #print numExamples
	#print "Found %d core nodes."%(len(core))
	#if len(core) < 1:
 #               return -1, -1, -1, -1

	# with 70% chance to use a negative train example (and some % chance of being in test set)
	# maybe consider only looking at edges and no-edge examples between core nodes
	#print "Generating feature matrix for training graph..."
	trainNegSet = set() #check if example in pos training set if is edge
	testNegSet = set()
	#print "Num edges = num pos/neg examples: %d"%(Gtrain.GetEdges())
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
				newRow = getFeatureVector(Gtrain, data, n1, n2)
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
				if tries > 1000:
					continue
				#numTrainNeg += 1
				if src < rnd:
					newRow = getFeatureVector(Gtrain, data, src, rnd)
					if newRow == None:
						continue
					trainNegSet.add((src, rnd))
					trainFeatures.append(newRow)
					trainLabels.append(getGoldLabel(G, src, rnd))
					numTrainNeg += 1
				else:
					newRow = getFeatureVector(Gtrain, data, rnd, src)
					if newRow == None:
						continue
					trainNegSet.add((rnd,src))
					trainFeatures.append(newRow)
					trainLabels.append(getGoldLabel(G, rnd, src))
			elif G.IsEdge(n1, n2):
				# add to test set
				testFeatures.append(getFeatureVector(Gtrain, data, n1, n2))
				testLabels.append(getGoldLabel(G, n1, n2))
				numTotalPos += 1
				numTestPos += 1 #numTestPos should = numTestPosEx
				# add a negative test example
				rnd = Gtrain.GetRndNId()
				tries = 0
				while G.IsEdge(src, rnd) or (src, rnd) in trainNegSet or (rnd, src) in trainNegSet or (src, rnd) in testNegSet or (rnd, src) in testNegSet or tries < 100:
					rnd = Gtrain.GetRndNId()
					tries += 1
				if tries > 1000:
					continue
				if src < rnd:
					newRow = getFeatureVector(Gtrain, data, src, rnd)
					if newRow == None:
						continue
					testNegSet.add((src, rnd))
					testFeatures.append(newRow)
					testLabels.append(getGoldLabel(G, src, rnd))
					numTestNeg += 1
				else:
					newRow = getFeatureVector(Gtrain, data, rnd, src)
					if newRow == None:
						continue
					testNegSet.add((rnd,src))
					testFeatures.append(newRow)
					testLabels.append(getGoldLabel(G, n1, n2))
					numTestNeg += 1

	# add some more negative training examples
	if numTrainNeg < numTrainPos:
		for n1 in G.Nodes():
			if numTrainNeg >= numTrainPos:
				break
				src = n1.GetId()
				for n2 in G.Nodes():
					dest = n2.GetId()
					if src >= dest:
						continue
						if src not in core and dest not in core:
							continue
							if not G.IsEdge(src, dest) and (src, dest) not in trainNegSet and (src, dest) not in testNegSet:
								p = np.random.uniform()
								if p < 0.2:
									newRow = getFeatureVector(Gtrain, data, src, dest)
									if newRow == None:
										continue
										trainNegSet.add((src, dest))
										trainFeatures.append(newRow)
										trainLabels.append(getGoldLabel(G, src, dest))
										numTrainNeg += 1
										if numTrainNeg >= numTrainPos:
											break

	#print "Num pos training examples: %d"%(numTrainPos)
	#print "Num neg training examples: %d"%(numTrainNeg)

	# convert to numpy
	#print "Converting to numpy arrays..."
	featureMatrix = np.array(trainFeatures)
	labelVector = np.array(trainLabels)

	# train on every edge in train graph (using train graph features and gold graph label)
	regr = linear_model.LogisticRegression(class_weight='balanced')
	#print "Training logistic regression model on depth %d..."%(depth)
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
	return (1.0*numCorrectPos/numClassifiedPos), (1.0*numCorrectPos/numTestPos), (1.0*numCorrectNeg/numClassifiedNeg), (1.0*numCorrectNeg/numTestNeg)

def generateTrainingGraph(testingPercent):
	print "Reading in edge list for depth %d..."%(depth)
	G = snap.LoadEdgeList(snap.PUNGraph, "test-graph.txt", 0, 1) # change this for which graph edges to load
	Gtrain = snap.LoadEdgeList(snap.PUNGraph, "test-graph.txt", 0, 1)

	# for every edge, testingPercent time remove edge from train graph (and add to test set)
	print "Generating training graph..."
	for e in G.Edges():
		src = e.GetSrcNId()
		dest = e.GetDstNId()
		p = np.random.uniform()
		if p < testingPercent:
			Gtrain.DelEdge(src, dest)
	return G, Gtrain

# Constants
depth = 3

# read in graph twice
# one will be the gold graph and the other the train graph
print "Parsing text data..."
filenames = [ "0301/{}.txt".format(i) for i in range(0, depth+1) ] # change this for which data files to parse
data = parser.Data(filenames)

avgPosPrecs = []
avgPosRecs = []
avgNegPrecs = []
avgNegRecs = []

trialXaxis = []
posPrecs = []
posRecs = []
negPrecs = []
negRecs = []

avgNegRecC = {}
avgNegPrecC = {}
avgPosRecC = {}
avgPosPrecC = {}
k = [25, 30, 35, 40, 45, 50]
for i in k:       
        avgNegRecC[i] = 0.0
        avgNegPrecC[i] = 0.0
        avgPosRecC[i] = 0.0
        avgPosPrecC[i] = 0.0
	
for trial in xrange(10):
	G, Gtrain = generateTrainingGraph(0.2)
	for i in k:
		print "---------------------------------> running logreg on core nodes k = %d on trial %d"%(i, trial+1)
		posPrec, posRec, negPrec, negRec = logreg(G, Gtrain, data, i)
		trialXaxis.append(i)
		posPrecs.append(posPrec)
		posRecs.append(posRec)
		negPrecs.append(negPrec)
		negRecs.append(negRec)
		avgPosPrecC[i] += posPrec
		avgPosRecC[i] += posRec
		avgNegPrecC[i] += negPrec
		avgNegRecC[i] += negRec

avgNegRecs = []
avgNegPrecs = []
avgPosRecs = []
avgPosPrecs = []
for i in k:
	avgPosPrecs.append(avgPosPrecC[i]/len(k))
	avgPosRecs.append(avgPosRecC[i]/len(k))
	avgNegPrecs.append(avgNegPrecC[i]/len(k))
	avgNegRecs.append(avgNegRecC[i]/len(k))

print k
print avgPosPrecs
print avgPosRecs
print avgNegPrecs
print avgNegRecs
print trialXaxis
print posPrecs
print posRecs
print negPrecs
print negRecs

plt.clf()
plt.figure()
plt.plot(k, avgPosPrecs, "o", label="avg pos precision")
plt.plot(k, avgPosRecs, "o", label="avg pos recall")
plt.title("Avg. Positive Precision + Recall")
plt.legend()
plt.xlabel("k values (for core node)")
plt.ylabel("Precision/Recall")
plt.savefig("avgPosResults.pdf")

plt.clf()
plt.figure()
plt.plot(k, avgNegPrecs, "o", label="avg neg precision")
plt.plot(k, avgNegRecs, "o", label="avg neg recall")
plt.title("Avg. Negative Precision + Recall")
plt.legend()
plt.xlabel("k values (for core node)")
plt.ylabel("Precision/Recall")
plt.savefig("avgNegResults.pdf")


plt.clf()
plt.figure()
plt.plot(trialXaxis, posPrecs, "o", label="indv pos precision")
plt.plot(trialXaxis, posRecs, "o", label="indv pos recall")
plt.title("Positive Precision + Recall by Trial")
plt.legend()
plt.xlabel("k values (for core node)")
plt.ylabel("Precision/Recall")
plt.savefig("indvPosResults.pdf")


plt.clf()
plt.figure()
plt.plot(trialXaxis, negPrecs, "o", label="indv neg precision")
plt.plot(trialXaxis, negRecs, "o", label="indv neg recall")
plt.title("Avg. Positive Precision + Recall")
plt.legend()
plt.xlabel("k values (for core node)")
plt.ylabel("Precision/Recall")
plt.savefig("indvNegResults.pdf")
