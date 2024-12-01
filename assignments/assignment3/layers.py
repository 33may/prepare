import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    raise Exception("Not implemented!")

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    raise Exception("Not implemented!")
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO copy from the previous assignment
        raise Exception("Not implemented!")
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO copy from the previous assignment

        raise Exception("Not implemented!")
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

        self.last_X = None


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height + 2 * self.padding - self.filter_size + 1
        out_width = width + 2 * self.padding - self.filter_size + 1

        result = np.zeros((batch_size, out_height, out_width , self.out_channels))

        if self.padding > 0:
            padded_input = np.zeros((batch_size, height + 2 * self.padding, width + 2 * self.padding, channels))
            padded_input[:, self.padding:height + self.padding, self.padding:width + self.padding, :] = X
            X = padded_input

        self.last_X = X

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                input = X[:, y:y+self.filter_size , x:x+self.filter_size , :]

                input = input.reshape(batch_size, self.filter_size * self.filter_size * channels)

                weights = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)

                res = input @ weights

                res += self.B.value

                result[:, y , x, : ] = res

        return result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.last_X.shape

        if self.padding:
            height -= self.padding * 2
            width -= self.padding * 2
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        weights = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)

        d_input = np.zeros_like(self.last_X)

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                input_patch = self.last_X[:, y:y+self.filter_size , x:x+self.filter_size , :]

                input_reshaped = input_patch.reshape(batch_size, self.filter_size * self.filter_size * self.in_channels)

                d_out_patch = d_out[:, y, x, :]

                dw = input_reshaped.T @ d_out_patch

                db = np.sum(d_out_patch, axis=0)

                dX_patch = d_out_patch @ weights.T

                d_input[:, y:y+self.filter_size, x:x+self.filter_size, :] += dX_patch.reshape(batch_size, self.filter_size, self.filter_size, self.in_channels)

                dw_original = dw.reshape(self.filter_size, self.filter_size, self.in_channels, self.out_channels)

                self.W.grad += dw_original

                self.B.grad += db

        return d_input if not self.padding else d_input[:, self.padding:height + self.padding, self.padding:width + self.padding, :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension

        self.X = X

        out_width = (width - self.pool_size ) // self.stride + 1
        out_height = (height - self.pool_size ) // self.stride + 1

        result = np.zeros((batch_size, out_height, out_width, channels))

        for y in range(out_height):
            for x in range(out_width):

                region = X[:, y * self.stride:y * self.stride +self.pool_size, x * self.stride:x * self.stride +self.pool_size, :]

                max = np.max(region, axis=(1,2))

                result[:,y,x,:] = max

        return result


    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape

        dX = np.zeros_like(self.X)

        out_width = (width - self.pool_size ) // self.stride + 1
        out_height = (height - self.pool_size ) // self.stride + 1

        for y in range(out_height):
            for x in range(out_width):
                out_region = d_out[:, y, x, :]

                input_region = self.X[:, y * self.stride:y * self.stride +self.pool_size, x * self.stride:x * self.stride +self.pool_size, :]

                input_region_flat = input_region.reshape(batch_size, -1, channels)

                max_indices = np.argmax(input_region_flat, axis=1)

                row_indices_region, col_indices_region = np.unravel_index(max_indices, (self.pool_size, self.pool_size))

                row_indices_global = row_indices_region + y * self.stride
                col_indices_global = col_indices_region + x * self.stride

                batch_indices = np.arange(batch_size)[:, None]
                channel_indices = np.arange(channels)

                dX[batch_indices, row_indices_global, col_indices_global, channel_indices] += out_region

        return dX


    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
