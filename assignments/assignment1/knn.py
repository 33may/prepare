import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):

                train_vector = self.train_X[i_train]
                test_vector = X[i_test]

                distance = train_vector - test_vector

                norm = np.sum(np.abs(distance), axis=0)

                dists[i_test, i_train] = norm

        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            test_vector = X[i_test]

            # Use broadcasting to compute the L1 distance
            # between one test sample and all train samples
            dists[i_test, :] = np.sum(np.abs(self.train_X - test_vector), axis=1)

        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)

        train_tensor = self.train_X[np.newaxis, :, :]
        test_tensor = X[:, np.newaxis, :]

        distances = np.sum(np.abs(train_tensor - test_tensor), axis=2)

        dists = distances

        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)

        for i in range(num_test):
            cur_dist = dists[i]

            # Сортируем индексы по возрастанию расстояний
            sorted_indices = np.argsort(cur_dist)

            # Берем k ближайших соседей
            k_nearest_indices = sorted_indices[:self.k]

            # Получаем метки ближайших соседей
            k_nearest_labels = self.train_y[k_nearest_indices]

            # Определяем метку на основе голосования
            majority_label = np.bincount(k_nearest_labels).argmax()

            # Сохраняем предсказание
            pred[i] = majority_label

        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, int)
        for i in range(num_test):
            cur_dist = dists[i]

            sorted_indices = np.argsort(cur_dist)

            # Берем k ближайших соседей
            k_nearest_indices = sorted_indices[:self.k]

            # Получаем метки ближайших соседей
            k_nearest_labels = self.train_y[k_nearest_indices]

            majority_label = np.bincount(k_nearest_labels).argmax()

            # Сохраняем предсказание
            pred[i] = majority_label

        return pred
