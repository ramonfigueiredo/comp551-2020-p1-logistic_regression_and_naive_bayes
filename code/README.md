

## How to run the Python program?
1. Install [virtualenv](https://virtualenv.pypa.io/en/latest/)
	* To activate the virtualenv on Linux or MacOS: ```source venv/bin/activate```
	* To activate the virtualenv on Windows: ```\venv\Script\activate.bat```

2. Run the program

```sh
cd <folder_name>/

virtualenv venv -p python3 or python3 -m venv env  if you are using Mac



source venv/bin/activate

pip install -r requirements.txt

python main.py
```

**Note**: To desactivate the virtual environment

```sh
deactivate
```

For more help you can type python ```main.py -h``` and get the arguments to run specific methods on specific datasets. 

```
usage: main.py [-h] [-c CLASSIFIER] [-tsize TRAINING_SET_SIZE] [-d DATASET]
               [-plot_cost] [-lr LEARNING_RATES_LIST] [-heatmap] [-save_logs]
               [-v]

MiniProject 1: Logistic Regression and Naive Bayes. Authors: Ramon Figueiredo
Pessoa, Rafael Gomes Braga, Ege Odaci

optional arguments:
  -h, --help            show this help message and exit
  -c CLASSIFIER, --classifier CLASSIFIER
                        Classifier used (Options: all,
                        logistic_regression_sklearn OR lrskl,
                        logistic_regression OR lr, naive_bayes_sklearn OR
                        nbskl naive_bayes OR nb).
  -tsize TRAINING_SET_SIZE, --train_size TRAINING_SET_SIZE
                        Training set size (percentage). Should be between 0.0
                        and 1.0 and represent the proportion of the dataset to
                        include in the training split
  -d DATASET, --dataset DATASET
                        Database used (Options: all, ionosphere OR i adult OR
                        a wine_quality OR wqbreast_cancer_diagnosis OR bcd).
  -plot_cost, --plot_cost_vs_iterations
                        Plot different learning rates for gradient descent
                        applied to logistic regression. Use a threshold for
                        change in the value of the cost function as
                        termination criteria, and plot the accuracy on
                        train/validation set as a function of iterations of
                        gradient descent.
  -lr LEARNING_RATES_LIST, --learning_rates_list LEARNING_RATES_LIST
                        Learning rates list used to plot cost versus
                        iterations. For example: python main --classifier
                        logistic_regression --dataset adult -plot_cost -lr
                        0.001 -lr 0.01 -lr 0.05 -lr 1
  -heatmap, --plot_heatmap
                        Plot heatmaps for all datasets. Show the correlations
                        between the datasets features (X). For example: python
                        main.py --classifier naive_bayes --dataset
                        wine_quality -heatmap
  -save_logs, --save_logs_in_file
                        Save logs in a file
  -v, --version         show program's version number and exit

COMP 551 (001/002), Applied Machine Learning, Winter 2020, McGill University.
```


