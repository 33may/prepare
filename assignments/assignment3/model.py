import numpy as np

from assignments.assignment1.linear_classifer import softmax
from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """


        self.layer1 = ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1)
        self.relu1 = ReLULayer()
        self.max_pool1 = MaxPoolingLayer(pool_size=4, stride=1)
        self.layer2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.relu2 = ReLULayer()
        self.max_pool2 = MaxPoolingLayer(pool_size=4, stride=1)
        self.flatten = Flattener()
        self.fully_connected = FullyConnectedLayer(1352, n_output_classes)
        # TODO Create necessary layers



    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Очистка градиентов
        params = self.params()
        for key in params:
            params[key].grad = np.zeros_like(params[key].grad)

        # Прямой проход
        preds = self.forward(X)

        # Вычисление потери и градиента
        loss, dprediction = softmax_with_cross_entropy(preds, y)

        # Обратный проход
        d_fc = self.fully_connected.backward(dprediction)
        d_flat = self.flatten.backward(d_fc)
        d_maxpool2 = self.max_pool2.backward(d_flat)
        d_relu2 = self.relu2.backward(d_maxpool2)      # Добавлен ReLU2.backward
        d_conv2 = self.layer2.backward(d_relu2)
        d_maxpool1 = self.max_pool1.backward(d_conv2)
        d_relu1 = self.relu1.backward(d_maxpool1)
        d_conv1 = self.layer1.backward(d_relu1)

        return loss


    def forward(self, X):
        l1_out = self.layer1.forward(X)
        relu1_out = self.relu1.forward(l1_out)
        maxpool1_out = self.max_pool1.forward(relu1_out)
        layer2_out = self.layer2.forward(maxpool1_out)
        relu2_out = self.relu2.forward(layer2_out)
        maxpool2_out = self.max_pool2.forward(relu2_out)
        flat_out = self.flatten.forward(maxpool2_out)
        fc1_out = self.fully_connected.forward(flat_out)

        return fc1_out

    def predict(self, X):

        logits = self.forward(X)

        probs = softmax(logits)

        return np.argmax(probs, axis=1)

    def params(self):
        result = {}

        l1_params = self.layer1.params()
        l2_params = self.layer2.params()


        result['conv1_w'] = l1_params['W']
        result['conv1_b'] = l1_params['B']
        result['conv2_w'] = l2_params['W']
        result['conv2_b'] = l2_params['B']
        fc_params = self.fully_connected.params()

        result['fc1_w'] = fc_params['W']
        result['fc1_b'] = fc_params['B']

        params = result

        return params
