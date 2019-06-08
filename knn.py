import numpy as np
from sklearn import datasets
import operator
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def loadData():
    iris = datasets.load_iris()
    X = np.array(iris.data[:, 3 : 4])
    y = np.array(iris.target)
    return X, y


'''
    --euclideanDistance--

    Input: 
        inst1 - vector of size 
        inst2 - vector of size
        length - the length of the two vectors.

    Output: 
       the euclidean distance between the two instances.
''' 
def euclideanDistance(inst1, inst2, length):
	distance = 0
	for x in range(length):
		distance += np.power((inst1[x] - inst2[x]), 2)
	return np.sqrt(distance)


'''
    --getNeighbors--

    Input: 
        TrainingSet - List of samples of the training set (with the tagg collum (y)).
        testInstance - chosen instance to be look at.
        k - the number of instances to find as the "best" neighbors.

    Output: 
        neighbors - the k nearest neighbors.
''' 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance) - 1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors


'''
    --getDecision--

    Input: 
        neighbors - neighbors of some instance

    Output:
        returns the id of the most common class of all neighbors.
'''
def getDecision(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		decision = neighbors[x][-1]
		if decision in classVotes:
			classVotes[decision] += 1
		else:
			classVotes[decision] = 1

	return max(classVotes.items(), key=operator.itemgetter(1))[0]


'''
    --getAccuracy--

    Input:
        testSet - instances(with tags) ([[1,2,3,0], [1,4,5,0]]).
        predictions - predictions for the given testSet by KNN ([0, 0]).

    Output:
        returns the prediction accuracy in percents (100%).
''' 
def getAccuracy(X, predictions):
	correct = 0
	for x in range(len(X)):
		if X[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(X))) * 100.0

'''
    --KNN--

    Input:
        X - training set with the y appended as the last collum ( [1, 2, 3, 'a'])
        k - k of the algorithm (how many neighbors to search for each instance).

    Output:
        accuracy - percents of right predictions.
        predictions - the predictions for each sample.
'''
def KNN(X, k):
    predictions = []
    for x in range(len(X)):
        neighbors = getNeighbors(X, X[x], k)
        result = getDecision(neighbors)
        predictions.append(result)
        print('predicted=' + repr(result) + ', actual=' + repr(X[x][-1]))
    accuracy = getAccuracy(X, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    return accuracy, predictions


if __name__ == '__main__':
    k = 5
    X, y = loadData()
    X = np.concatenate((X, y[:,None]), axis=1)
    acc, pred = KNN(X, k)
