import snap
import numpy as np
from sklearn import linear_model
import parser # have local parse copy


regr = linear_model.LogisticRegression(class_weight='balanced')

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
top100EdgesNeg = []
top100EdgeScoresNeg = []
lowestScoreNeg = 0
numCorrectPredictions = 0
numTotalPredctions = 0
for src in Gtrain.Nodes():
	#if len(top100Edges) > 2 and len(top100EdgeScores) > 2:
	#	break
	n1 = src.GetId()
	for dest in Gtrain.Nodes():
		#if len(top100Edges) > 2 and len(top100EdgeScores) > 2:
		#	break
		n2 = dest.GetId()
		if n1 > n2:
			# enforce src < dest in all edge considerations
			continue
		if Gtrain.IsEdge(n1, n2):
			continue
		if (n1, n2) in trainNegSet:
			continue

		# check if enough edges
		if src.GetDeg() < 35 or Gtest.GetNI(n1).GetDeg() < 35 or dest.GetDeg() < 35 or Gtest.GetNI(n2).GetDeg() < 35:
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
		result = regr.predict(X.reshape(1,-1))
		numTotalPredctions += 1
		if (result[0] == 0 and not G.IsEdge(n1,n2)) or (result[0] == 1 and G.IsEdge(n1,n2)):
			numCorrectPredictions += 1

		classProbs = regr.predict_proba(X.reshape(1,-1))
		yesEdgeProb = classProbs[0,1]
		noEdgeProb = classProbs[0,0]

		if len(top100EdgeScores) < 100:
			top100Edges.append( [yesEdgeProb, (n1, n2)] )
			top100EdgeScores.append(yesEdgeProb)
			lowestScore = min(lowestScore, yesEdgeProb)
		else:
			if yesEdgeProb > lowestScore:
				lowestLoc = np.argmin(top100EdgeScores)
				top100Edges[lowestLoc] = [ yesEdgeProb,(n1, n2) ]
				top100EdgeScores[lowestLoc] = yesEdgeProb
				lowestScore = min(top100EdgeScores)

		if len(top100EdgeScoresNeg) < 100:
			top100EdgesNeg.append( [noEdgeProb, (n1, n2)] )
			top100EdgeScoresNeg.append(noEdgeProb)
			lowestScore = min(lowestScoreNeg, noEdgeProb)
		else:
			if noEdgeProb > lowestScoreNeg:
				lowestLoc = np.argmin(top100EdgeScoresNeg)
				top100EdgesNeg[lowestLoc] = [ noEdgeProb,(n1, n2) ]
				top100EdgeScoresNeg[lowestLoc] = noEdgeProb
				lowestScoreNeg = min(top100EdgeScoresNeg)

top100Edges.sort(key=lambda x:-x[0])
top100EdgesNeg.sort(key=lambda x:-x[0])

numTrue = 0
for i in xrange(len(top100Edges)):
	if Gtest.IsEdge(top100Edges[i][1][0],top100Edges[i][1][1]):
		numTrue += 1
numTrueNeg = 0
for i in xrange(len(top100EdgesNeg)):
	if Gtest.IsEdge(top100EdgesNeg[i][1][0],top100EdgesNeg[i][1][1]):
		numTrueNeg += 1

print top100EdgeScores
print top100EdgeScoresNeg
print "Test Accuracy (class1): %d"%(numTrue)
print "Test Accuracy (class0): %d"%(numTrueNeg)
print "Qualifying Test Cases: %d"%(len(top100EdgeScores))

print "All correct predictions: %d"%(numCorrectPredictions)
print "Total predictions made: %d"%(numTotalPredctions)

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