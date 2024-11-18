import numpy as np


def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''

    assert isinstance(x, np.ndarray)
    assert x.dtype == float

    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.allclose(orig_x, x, atol=tol), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    for idx in np.ndindex(x.shape):
        analytic_grad_at_ix = analytic_grad[idx]

        arg_array_minus = np.copy(x)
        arg_array_plus = np.copy(x)
        arg_array_minus[idx] -= delta
        arg_array_plus[idx] += delta

        fx_plus, _ = f(arg_array_plus)
        fx_minus, _ = f(arg_array_minus)

        numeric_grad_at_ix = (fx_plus - fx_minus) / (2 * delta)

        # TODO compute value of numeric gradient of f to idx
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (idx, analytic_grad_at_ix, numeric_grad_at_ix))
            return False


    print("Gradient check passed!")
    return True




