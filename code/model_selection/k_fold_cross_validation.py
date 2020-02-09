import time

import numpy as np

from metrics.accuracy_score import evaluate_acc


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
        validation_indices = folds[i]
        training_indices = [item
                    for s in folds if s is not validation_indices
                    for item in s]

        X_validation = X[validation_indices]
        y_validation = y[validation_indices]
        X_train = X[training_indices]
        y_train = y[training_indices]

        if verbose:
            print('\n==> Fold {}:'.format(i + 1))
            print('\nValidation set indices', validation_indices)
            print('\nX_validation', X_validation)
            print('\ny_validation', y_validation)
            print('\nTraining set indices', training_indices)
            print('\nX_train', X_train)
            print('\ny_train', y_train)

        # Fit according to X, y
        start_fit = time.time()
        estimator.fit(X_train, y_train)
        fit_times.append(time.time() - start_fit)

        # Predicting the Validation set results
        start_predict = time.time()
        y_pred = estimator.predict(X_validation)
        predict_times.append(time.time() - start_predict)

        # Model accuracy
        start_model_accuracy = time.time()
        model_accuracy = evaluate_acc(y_validation, y_pred)
        model_accuracy_times.append(time.time() - start_model_accuracy)

        scores.append(model_accuracy)

        if verbose:
            print('\ny_pred', y_pred)
            print("\nModel accuracy score: ", model_accuracy)

        processing_times_of_folds.append(time.time() - start_fold)

    return np.asarray(scores), np.asarray(fit_times), np.asarray(predict_times), np.asarray(model_accuracy_times), \
           np.asarray(processing_times_of_folds)
