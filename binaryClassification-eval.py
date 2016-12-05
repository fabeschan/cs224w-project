'''
use scikit learn
the training data will be every pair of nodes, each labeled as edge or no edge (ignore test case data)
the testing data will be select pairs of nodes (maybe 10% of the top 10 highest degree nodes to other nodes)
use metadata as training features? (i.e. each possible feature (in category or not) will be 0/1 present or not in feature vector)
'''
import snap
import numpy as np
from sklearn import linear_model
import parser # have local parse copy


regr = linear_model.LogisticRegression(class_weight='balanced')
#print "class order"
#print regr.classes_

# track each possible edge (a, b) where a < b with a unique identifier in some dicitonary
# for each edge, create numpy array vector of 0/1 for features:
# uploader(?), age, category, length, views, rate, ratings, comments for each end point

goldLabels = {} # maps edge to -1 if not present or 1 if present
labels = [] #maps row number to gold label

print "Parsing text data..."
filenames = [ "0301/{}.txt".format(i) for i in range(0, 3) ] # change this for which data files to parse
data = parser.Data(filenames)
print "Reading in edge list for depth 2..."
G = snap.LoadEdgeList(snap.PUNGraph, "test-graph-d2.txt", 0, 1) # change this for which graph edges to load
Gtrain = G
Gtest = G
for e in G.Edges():
	src = e.GetSrcNId()
	dest = e.GetDstNId()
	p = np.random.uniform()
	if p < 0.2:
		Gtrain.DelEdge(src, dest)
	else:
		Gtest.DelEdge(src,dest)

categoryId = {} # maps category name to unique integer
counter = 0
print "Generating category numbers..."
for category in data.categories:
	categoryId[category] = counter
	counter += 1

# get Gold labels and generate data set feature vectors
A = [] # 2D list that will be converted to a 2D numpy array
numSkipped = 0
numNonEdgeExamples = 0

userId = {} # maps username to unique id
userIdNext = 0

# TODO: random subgraph and train on that instead (in 0-4 depth)
# get probabilities of top 100 edges likely to appear? then check how many of those edges are actually in the test graph? (can do this without iterating over all node pairs?) --> node pairs where nodes have k train in and k test in data set?
# 		test edges for nodes that have high degree (top 100 high degree nodes (>50 edges in test and train sets))
# SPlit graph into training/testing data before hand? don't test on any pair that is trained on

# take whole graph and take half edges in test graph and half graph in train graph
#
# create own test and train matrix

# Gprime = snap.GetRndESubgraph(G, 1000)

print "Generating feature matrix for training graph..."
trainNegSet = set() #check if example in pos training set if is edge
for src in Gtrain.Nodes():
	#if len(trainNegSet) > 2 and len(A) > 2:
	#	break
	n1 = src.GetId()
	for dest in Gtrain.Nodes():
		#if len(trainNegSet) > 2 and len(A) > 2:
		#	break
		n2 = dest.GetId()
		if n1 > n2:
			# enforce src < dest in all edge considerations
			continue
		newRow = []
		srcVid = data.videoid[n1]
		destVid = data.videoid[n2]
		if srcVid not in data.lookup or destVid not in data.lookup or data.lookup[srcVid] == None or data.lookup[destVid] == None:
			numSkipped += 1
			continue # no valid data

		if data.lookup[srcVid][0] not in userId:
			userId[data.lookup[srcVid][0]] = userIdNext
			userIdNext += 1
		if data.lookup[destVid][0] not in userId:
			userId[data.lookup[destVid][0]] = userIdNext
			userIdNext += 1

		if Gtrain.IsEdge(n1, n2):
			goldLabels[(n1, n2)] = 1
			labels.append(1)
		elif numNonEdgeExamples < Gtrain.GetEdges(): #force balance the classes
		#else:
			# randomly decide if this negative example will be trained on
			goldLabels[(n1, n2)] = 0#-1
			p = np.random.uniform()
			if p < 0.7:
				labels.append(0)#-1)
				trainNegSet.add((n1, n2))
				numNonEdgeExamples += 1
			else:
				continue
		else:
			goldLabels[(n1, n2)] = 0#-1
			continue # too many non edge examples?

		newRow.append(data.lookup[srcVid][2]) # data.lookup[videoid][field index]
		newRow.append(data.lookup[destVid][2]) # age
		newRow.append(categoryId[data.lookup[srcVid][3]])
		newRow.append(categoryId[data.lookup[destVid][3]]) # category
		newRow.append(data.lookup[srcVid][4])
		newRow.append(data.lookup[destVid][4]) # length
		newRow.append(data.lookup[srcVid][6])
		newRow.append(data.lookup[destVid][6]) # rate
		newRow.append(src.GetDeg())
		newRow.append(dest.GetDeg()) # degree
		#uploader as feature - tendency for related videos between same user
		newRow.append(userId[data.lookup[srcVid][0]])
		newRow.append(userId[data.lookup[destVid][0]])

		A.append(newRow)
print "Skipped %d possible node pairs."%(numSkipped)

print "Converting to numpy arrays..."
# convert 2D list into 2D numpy array
featureMatrix = np.array(A)
# convert label vector into numpy array
labelVector = np.array(labels)

# run logreg
print "Training logistic regression model on depth 2..."
regr.fit(featureMatrix, labelVector)

print "Generating test examples where at least k examples in the train set for each node and not trained on exactly"
top100Edges = []
top100EdgeScores = []
lowestScore = 0
for src in Gtrain.Nodes():
	n1 = src.GetId()
	for dest in Gtrain.Nodes():
		n2 = dest.GetId()
		if n1 > n2:
			# enforce src < dest in all edge considerations
			continue
		if Gtrain.IsEdge(n1, n2):
			continue
		if (n1, n2) in trainNegSet:
			continue

		# check if enough edges
		if src.GetDeg() < 40 or Gtest.GetNI(n1).GetDeg() < 40 or dest.GetDeg() < 40 or Gtest.GetNI(n2).GetDeg() < 40:
			continue

		newRow = []
		srcVid = data.videoid[n1]
		destVid = data.videoid[n2]
		if srcVid not in data.lookup or destVid not in data.lookup or data.lookup[srcVid] == None or data.lookup[destVid] == None:
			continue # no valid data

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
		newRow.append(src.GetDeg())
		newRow.append(dest.GetDeg()) # degree
		#uploader as feature - tendency for related videos between same user
		newRow.append(userId[data.lookup[srcVid][0]])
		newRow.append(userId[data.lookup[destVid][0]])

		X = np.array(newRow)
		classProbs = regr.predict_proba(X)
		yesEdgeProb = classProbs[0,1]
		if len(scores) < 100:
			top100Edges.append( [yesEdgeProb, (n1, n2)] )
			top100EdgeScores.append(yesEdgeProb)
			lowestScore = min(lowestScore, yesEdgeProb)
		else:
			if yesEdgeProb > lowestScore:
				lowestLoc = np.argmin(top100EdgeScores)
				top100Edges[lowestLoc] = [ yesEdgeProb,(n1, n2) ]
				top100EdgeScores[lowestLoc] = yesEdgeProb
				lowestScore = min(top100EdgeScores)

top100Edges.sort(key=lambda x:-x[0])

numTrue = 0
for i in xrange(len(top100Edges)):
	if Gtest.IsEdge(top100Edges[i][1][0],top100Edges[i][1][1]):
		numTrue += 1

print "Test Accuracy: %d"%(numTrue)
print "Qualifying Test Cases: %d"%(len(top100EdgeScores))

'''
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error (Train): %f"
      % np.mean((regr.predict(featureMatrix_train) - labelVector_train) ** 2))
print('Mean Accuracy Score (Train): %f' % regr.score(featureMatrix_train, labelVector_train))

print("Mean squared error (Test): %f"
      % np.mean((regr.predict(featureMatrix_test) - labelVector_test) ** 2))
# Explained variance score: 1 is perfect prediction
# A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.
print('Mean Accuracy Score (Test): %f' % regr.score(featureMatrix_test, labelVector_test))
'''