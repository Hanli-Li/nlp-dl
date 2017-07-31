#!/usr/bin/env python

import numpy as np
import random


def gradcheck(f, x):
    """ 
    Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not modify h

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # Modify x[ix] with h defined above to compute numerical gradients.
        # Call random.setstate(rndstate) before calling f(x) each time.
        # This makes it possible to test cost functions with built in randomness later.
        x[ix] += h
        random.setstate(rndstate)
        fx_1, g = f(x)

        x[ix] -= 2 * h
        random.setstate(rndstate)
        fx_2, g = f(x)

        numgrad = (fx_1 - fx_2) / (2 * h)
        x[ix] += h

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return

        it.iternext() # Step to next dimension

    print "Gradient check passed!"


def test_gradcheck():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running gradcheck tests..."
    gradcheck(quad, np.array(123.456))      # scalar test
    gradcheck(quad, np.random.randn(3,))    # 1-D test
    gradcheck(quad, np.random.randn(4,5))   # 2-D test
    print ""


if __name__ == "__main__":
    test_gradcheck()
