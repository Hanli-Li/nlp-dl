import numpy as np


def softmax(x):
    """
    Compute the softmax function for each row of the input x, where
    x is either a single N-dim row vector or an M x N matrix

    For optimization, see numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    Arguments:
    x -- An N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- To modify x in-place
    """
    orig_shape = x.shape
    if len(orig_shape) == 1:
        x = x.reshape(1, orig_shape[0])

    row_max = np.max(x, 1)
    x = np.exp(x - row_max[:, np.newaxis])
    partitions = 1.0 / np.sum(x, 1)
    x = x * partitions[:, np.newaxis]
    x = x.reshape(orig_shape)

    assert x.shape == orig_shape
    return x


def test_softmax():
    """
    Non-exhaustive simple tests for softmax function
    """
    print "Running basic tests for softmax function..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "Tests passed!\n"


if __name__ == "__main__":
    test_softmax()
