import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''

    if predictions.ndim == 1:
        predictions -= np.max(predictions)
    else:
        predictions -= np.max(predictions, axis=1, keepdims=True)

    exponents = np.exp(predictions)

    # TODO implement softmax
    # Your final implementation shouldn't have any loops

    if predictions.ndim == 1:
        sum = np.sum(exponents)
    else:
        sum = np.sum(exponents, axis=1, keepdims=True)

    probs = exponents / sum

    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: int or np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if probs.ndim == 1:
        loss = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        correct_logprobs = -np.log(probs[np.arange(batch_size), target_index])
        loss = np.mean(correct_logprobs)
    return loss


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
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops

    predictions_c = predictions.copy()

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

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops

    loss = reg_strength * np.sum(W ** 2)

    grad = 2 * reg_strength * W

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops

    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)

    dW = X.transpose().dot(dprediction)

    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):

            # Перемешать индексы тренировочных данных
            shuffled_indices = np.arange(num_train)

            # Разбить индексы на мини-батчи
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)


            # Инициализировать список для потерь текущей эпохи
            epoch_loss = []

            total_loss = 0
            # Цикл по батчам
            for batch_idxs in batches_indices:
                # Извлечь данные и метки для текущего батча
                train_X, train_y = X[batch_idxs], y[batch_idxs]
                # Вычислить предсказания и градиенты по весам с использованием softmax и cross-entropy loss
                loss_back, back_dW = linear_softmax(train_X, W=self.W, target_index=train_y)

                # Вычислить регуляризационную потерю и её градиент
                loss_reg, regularization_dw = l2_regularization(self.W, reg)

                # Сложить кросс-энтропийную потерю и регуляризацию

                total_dw = back_dW + regularization_dw

                # Обновить веса с использованием градиентного спуска

                self.W -= learning_rate * total_dw

                # Сохранить потерю для текущего батча

                batch_loss = loss_back + loss_reg

                epoch_loss.append(batch_loss)
            # Вычислить среднюю потерю для эпохи



            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            # end
            # Напечатать информацию об эпохе и значении функции потерь


            # print("Epoch %i, loss: %f" % (epoch, np.mean(epoch_loss)))

            loss_history.append(np.mean(epoch_loss))

        # Вернуть историю потерь

        # print(f"Loss: {loss_history[-1]}")

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=int)

        predictions = np.dot(X, self.W)

        probs = softmax(predictions)

        y_pred = np.argmax(probs, axis=1)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops

        return y_pred



                
                                                          

            

                
