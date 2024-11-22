import numpy as np

from assignments.assignment1.linear_classifer import softmax, cross_entropy_loss


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.sum(W ** 2)

    grad = 2 * reg_strength * W

    return loss, grad

def softmax_with_cross_entropy(preds, target_index):
    predictions_c = preds.copy()
    predictions_c = softmax(predictions_c)
    loss = cross_entropy_loss(predictions_c, target_index)
    dprediction = predictions_c.copy()

    if predictions_c.ndim == 1:
        dprediction[target_index] -= 1
    else:
        batch_size = predictions_c.shape[0]
        dprediction[np.arange(batch_size), target_index.ravel()] -= 1
        dprediction /= batch_size

    return loss, dprediction



class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, X):

        self.mask = (X > 0).astype(int)  # Сохраняем маску, где X > 0
        return np.maximum(X, 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """

        d_result = d_out * self.mask

        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X  # Сохраняем входные данные для backward-pass
        res = X @ self.W.value + self.B.value  # Линейное преобразование
        return res

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """

        d_input = d_out @ self.W.value.T

        d_W = self.X.T @ d_out
        self.W.grad += d_W

        d_B = np.sum(d_out, axis=0, keepdims=True)
        self.B.grad += d_B

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
