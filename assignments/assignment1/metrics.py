def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(prediction)):
        if ground_truth[i] == prediction[i]:
            if ground_truth[i]:
                TP += 1  # Истинно положительные: и предсказание, и метка истинно положительны
            else:
                TN += 1  # Истинно отрицательные: и предсказание, и метка истинно отрицательны
        else:
            if ground_truth[i]:
                FN += 1  # Ложно отрицательные: пример положительный, но предсказание отрицательное
            else:
                FP += 1  # Ложно положительные: пример отрицательный, но предсказание положительное

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    precision = TP / (TP + FP)

    recall = TP / (TP + FN)

    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''

    TP = 0
    size = prediction.shape[0]

    for i in range(len(prediction)):
        if ground_truth[i] == prediction[i]:
            TP += 1


    return TP / (TP + size)
