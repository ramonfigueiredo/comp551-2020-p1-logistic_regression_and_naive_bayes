import time

import numpy as np

from metrics.accuracy_score import accuracy_score


def cross_validation(estimator, X, y, k, randomize=False, verbose=False):
    if verbose:
        print('\nX', X)
        print('\ny', y)

    sample_indices = np.arange(X.shape[0])
    if verbose:
        print('\nSample indexes', sample_indices)

    if randomize:
        np.random.shuffle(sample_indices)
        if verbose:
            print('\nShuffle sample indexes', sample_indices)

    folds = [sample_indices[i::k] for i in range(k)] # folds = np.hsplit(sample_indices, k)

    scores = []
    fit_times = []
    predict_times = []
    model_accuracy_times = []
    processing_times_of_folds = []
    for i in range(k):
        start_fold = time.time()
        test_indices = folds[i]
        training_indices = [item
                    for s in folds if s is not test_indices
                    for item in s]

        X_test = X[test_indices]
        y_test = y[test_indices]
        X_train = X[training_indices]
        y_train = y[training_indices]

        if verbose:
            print('\n==> Fold {}:'.format(i + 1))
            print('\nTest set indices', test_indices)
            print('\nX_test', X_test)
            print('\ny_test', y_test)
            print('\nTraining set indices', training_indices)
            print('\nX_train', X_train)
            print('\ny_train', y_train)

        # Fit according to X, y
        start_fit = time.time()
        estimator.fit(X_train, y_train)
        fit_times.append(time.time() - start_fit)

        # Predicting the Test set results
        start_predict = time.time()
        y_pred = estimator.predict(X_test)
        predict_times.append(time.time() - start_predict)

        # Model accuracy
        start_model_accuracy = time.time()
        model_accuracy = accuracy_score(y_test, y_pred)
        model_accuracy_times.append(time.time() - start_model_accuracy)

        scores.append(model_accuracy)

        if verbose:
            print('\ny_pred', y_pred)
            print("\nModel accuracy score: ", model_accuracy)

        processing_times_of_folds.append(time.time() - start_fold)

    return np.asarray(scores), np.asarray(fit_times), np.asarray(predict_times), np.asarray(model_accuracy_times), \
           np.asarray(processing_times_of_folds)
