def accuracy_score(y_true, y_pred):
    count = 0
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            count = count + 1
    totalaccuracy = count / len(y_true)
    return totalaccuracy
