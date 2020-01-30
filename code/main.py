from datasets.load_dataset import get_dataset, Datasets
import numpy as np

def run_logistic_regression(dataset):
    print('Dataset: {}'.format(dataset.name))
    X, y = get_dataset(dataset)
    print(X, y)

    # TODO: Do without use scikit-learn
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # TODO: Do without use scikit-learn
    # TODO: Change according selected dataset
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # TODO: Do without use scikit-learn
    # Fitting Logistic Regression to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    # TODO: Do without use scikit-learn
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print("Predicting the Test set results\n", y_pred)

    # TODO: Do without use scikit-learn
    # Creating the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix\n", cm)

    # TODO: Do without use scikit-learn
    # Calculating metrics using the confusion matrix
    TP = cm[0][0]
    FN = cm[0][1]
    TN = cm[1][0]
    FP = cm[1][1]
    print("True Positive (TP):", TP)
    print("False Negative (FN):", FN)
    print("True Negative (TN):", TN)
    print("False Positive (FP):", FP)

    print("\n")

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("Accuracy = (TP + TN) / (TP + TN + FP + FN): %.2f %%" % (accuracy * 100))

    recall = TP / (TP + FN)
    print("Recall = TP / (TP + FN): %.2f %%" % (recall * 100))

    precision = TP / (TP + FP)
    print("Precision = TP / (TP + FP): %.2f %%" % (precision * 100))

    Fmeasure = (2 * recall * precision) / (recall + precision)
    print("Fmeasure = (2 * recall * precision) / (recall + precision): %.2f %%" % (Fmeasure * 100))

    # TODO: Do without use scikit-learn
    # K-fold cross validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(classifier, X, y, cv=5)
    print("K-fold cross validation (k=5). Scores: ", scores)

    # TODO: Do without use scikit-learn
    # Model accuracy
    from sklearn.metrics import accuracy_score
    model_accuracy = accuracy_score(y_test, y_pred)
    print("Model accuracy (accuracy score): ", model_accuracy)


def run_naive_bayes(dataset):
    print('Dataset: {}'.format(dataset.name))
    X, y = get_dataset(dataset)
    print(X, y)

    # TODO: Do without use scikit-learn
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # TODO: Do without use scikit-learn
    # TODO: Change according selected dataset
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # TODO: Do without use scikit-learn
    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # TODO: Do without use scikit-learn
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print("Predicting the Test set results\n", y_pred)

    # TODO: Do without use scikit-learn
    # Creating the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix\n", cm)

    # TODO: Do without use scikit-learn
    # Calculating metrics using the confusion matrix
    TP = cm[0][0]
    FN = cm[0][1]
    TN = cm[1][0]
    FP = cm[1][1]
    print("True Positive (TP):", TP)
    print("False Negative (FN):", FN)
    print("True Negative (TN):", TN)
    print("False Positive (FP):", FP)

    print("\n")

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("Accuracy = (TP + TN) / (TP + TN + FP + FN): %.2f %%" % (accuracy * 100))

    recall = TP / (TP + FN)
    print("Recall = TP / (TP + FN): %.2f %%" % (recall * 100))

    precision = TP / (TP + FP)
    print("Precision = TP / (TP + FP): %.2f %%" % (precision * 100))

    Fmeasure = (2 * recall * precision) / (recall + precision)
    print("Fmeasure = (2 * recall * precision) / (recall + precision): %.2f %%" % (Fmeasure * 100))

    # TODO: Do without use scikit-learn
    # K-fold cross validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(classifier, X, y, cv=5)
    print("K-fold cross validation (k=5). Scores: ", scores)

    # TODO: Do without use scikit-learn
    # Model accuracy
    from sklearn.metrics import accuracy_score
    model_accuracy = accuracy_score(y_test, y_pred)
    print("Model accuracy (accuracy score): ", model_accuracy)


if __name__ == '__main__':
    print('\n\n==> Logistic Regression')
    # run_logistic_regression(Datasets.IONOSPHERE)
    # run_logistic_regression(Datasets.ADULT)
    # run_logistic_regression(Datasets.WINE_QUALITY)
    run_logistic_regression(Datasets.BREAST_CANCER_DIAGNOSIS)

    print('\n\n==> Naive Bayes')
    # run_naive_bayes(Datasets.IONOSPHERE)
    # run_naive_bayes(Datasets.ADULT)
    # run_naive_bayes(Datasets.WINE_QUALITY)
    run_naive_bayes(Datasets.BREAST_CANCER_DIAGNOSIS)

    print('DONE!')
