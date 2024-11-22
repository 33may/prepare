import numpy as np

from assignments.assignment1.linear_classifer import softmax
from layers import FullyConnectedLayer, ReLULayer, l2_regularization, softmax_with_cross_entropy


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg

        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)



    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        for param in self.params().values():
            param.grad = np.zeros_like(param.grad)


        # 2. Прямой проход
        fc1_out = self.fc1.forward(X)
        relu1_out = self.relu1.forward(fc1_out)
        fc2_out = self.fc2.forward(relu1_out)

        # Вычисляем loss и d_loss
        loss, d_loss = softmax_with_cross_entropy(fc2_out, y)

        # 3. Обратный проход
        d_relu1 = self.fc2.backward(d_loss)
        d_fc1 = self.relu1.backward(d_relu1)
        self.fc1.backward(d_fc1)

        for param in self.params().values():
            l2_loss, l2_grad = l2_regularization(param.value, self.reg)
            loss += l2_loss
            param.grad += l2_grad


        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """

        fc1_result = self.fc1.forward(X)

        relu1_result = self.relu1.forward(fc1_result)

        fc2_result = self.fc2.forward(relu1_result)

        pred = softmax(fc2_result)

        return np.argmax(pred, axis=1)

    def params(self):
        result =  {
            'fc1_W': self.fc1.W,
            'fc1_B': self.fc1.B,
            'fc2_W': self.fc2.W,
            'fc2_B': self.fc2.B
        }

        return result
