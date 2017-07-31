#!/usr/bin/env python

import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid function for the input x.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(s):
    """
    Compute the gradient for the sigmoid function.

    Arguments:
    s -- A scalar or numpy array.

    Return:
    ds -- computed gradient.
    """
    return s * (1 - s)


def test_sigmoid():
    """
    Non-exhaustive tests for sigmoid
    """
    print "Running basic tests for sigmoid..."
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print g
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print "Tests passed!\n"


if __name__ == "__main__":
    test_sigmoid()
