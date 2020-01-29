from datasets.load_dataset import get_dataset, Datasets

def run_logistic_regression(dataset):
    print('Dataset: {}'.format(dataset.name))
    X, y = get_dataset(dataset)
    print(X, y)


def run_naive_bayes(dataset):
    print('Dataset: {}'.format(dataset.name))
    X, y = get_dataset(dataset)
    print(X, y)


if __name__ == '__main__':
    print('\n\n==> Logistic Regression')
    run_logistic_regression(Datasets.IONOSPHERE)
    run_logistic_regression(Datasets.ADULT)
    run_logistic_regression(Datasets.WINE_QUALITY)
    run_logistic_regression(Datasets.BREAST_CANCER_DIAGNOSIS)

    print('\n\n==> Naive Bayes')
    run_naive_bayes(Datasets.IONOSPHERE)
    run_naive_bayes(Datasets.ADULT)
    run_naive_bayes(Datasets.WINE_QUALITY)
    run_naive_bayes(Datasets.BREAST_CANCER_DIAGNOSIS)

    print('DONE!')
