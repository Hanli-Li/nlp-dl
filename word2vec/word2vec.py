#!/usr/bin/env python

import numpy as np
import random
from softmax import softmax
from gradcheck import gradcheck
from sigmoid import sigmoid, sigmoid_grad


def normalizeRows(x):
    """ 
    Row normalization function

    A function that normalizes each row of a matrix to have
    unit length.
    """
    norm = np.sqrt(np.sum(x * x, 1))
    norm = norm.reshape(x.shape[0], 1)
    return x * (1.0 / norm)


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ 
    Softmax cost function for word2vec models

    The cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v})
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word vector
    grad -- the gradient with respect to all the other word vectors
    """
    y = softmax(np.dot(outputVectors, predicted))
    cost = -np.log(y[target])

    y[target] -= 1
    grad_pred = np.dot(y, outputVectors)

    V = outputVectors.shape[0]  # vocabulary size
    d = outputVectors.shape[1]  # vector dimension
    grad = y.reshape((V, 1)) * predicted.reshape((1, d))

    return cost, grad_pred, grad


def getNegativeSamples(target, dataset, K):
    """ 
    Samples K indexes which are not the target
    """
    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    """
    Negative sampling cost function for word2vec models

    Implements the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    subset = outputVectors[indices, :]
    f = sigmoid(-1 * np.dot(subset, predicted))

    f[0] = 1 - f[0]
    cost = -np.sum(np.log(f))

    f = 1 - f
    f[0] *= -1
    grad_pred = np.dot(f, subset)

    grad = np.zeros(outputVectors.shape)
    # grad[indices, :] = f.reshape((K + 1, 1)) * predicted.reshape((1, outputVectors.shape[1]))
    # sampling with replacement, cannot assign to grad directly
    gradtemp = f.reshape((K + 1, 1)) * predicted.reshape((1, outputVectors.shape[1]))
    for k in xrange(K+1):
        grad[indices[k], :] += gradtemp[k,:]

    return cost, grad_pred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ 
    Skip-gram model in word2vec

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    totalcost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    pred_idx = tokens[currentWord]
    predicted = inputVectors[pred_idx, :]
    for contextword in contextWords:
        target = tokens[contextword]
        cost, grad_pred, grad = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
        totalcost += cost
        gradIn[pred_idx, :] += grad_pred
        gradOut += grad

    return totalcost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """
    CBOW model in word2vec

    This function implements the continuous bag-of-words model.

    Arguments/Return specifications: same as the skip-gram model
    """
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    indices = []
    predicted = np.zeros(inputVectors.shape[1])
    for word in contextWords:
        idx = tokens[word]
        indices.append(idx)
        predicted += inputVectors[idx, :]  # duplicate indices possible, cannot submatrix and np.sum

    target = tokens[currentWord]
    cost, grad_pred, grad = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
    
    for idx in indices:
        gradIn[idx, :] += grad_pred
    gradOut += grad
    
    return cost, gradIn, gradOut


def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom
        
    return cost, grad


#############################################
# Testing functions                         #
#############################################
def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def test_word2vec():
    """
    Interface to the dataset for negative sampling
    """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW ===="
    gradcheck(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()