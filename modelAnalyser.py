from argparse import ArgumentParser
from nltk.tbl import feature
import os
from time import time

from matplotlib.pyplot import savefig, rc
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection.from_model import SelectFromModel
from sklearn.feature_selection.univariate_selection import SelectPercentile, \
    chi2
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection._search import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm.classes import SVC, LinearSVC

from file import load_json, load_sparse, load_array, \
    FILE_DATA_X_TRAIN, FILE_DATA_X_TEST, FILE_DATA_Y_TRAIN, FILE_DATA_Y_TEST, \
    FILE_DATA_TARGETS, DIR_RESULTS, save, load_json, \
    FILE_DATA_VOC, DIR_VECTORIZED_DATA, load_json_unicode
import matplotlib.pyplot as plt
from models.model import ModelClf, ModelSelectionGrid, ModelSelectionRandSearch
import numpy as np
from text.sdCountVectorizer import get_feature_names


FN_BASE = "%s_%s" # <model name>_<file type>

FILE_REPORT = "rep"

def analyse(X_train, X_test, y_train, y_test, voc, targets, chi, scoring, output_path, label):
    
    print("#" * 50)
    print("# Analyse %s" % label)
    print("#" * 50)
    
    feature_names = get_feature_names(voc)
    
    """ FEATURE SELECTION """
    if chi is not None:
        ch2 = SelectPercentile(chi2, percentile=chi)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
    
    """ INIT CLASSIFIER """
    models = []
    
    max_iter=1000
    tol=1e-3
    
    ######################################## Passive Aggressive Classifier ########################################
    #models.append(ModelClf("Passive-Aggressive", PassiveAggressiveClassifier(max_iter=max_iter, tol=tol), targets, feature_names))
    
    ######################################## RidgeClassifier ########################################
    #models.append(ModelClf("Ridge Classifier", RidgeClassifier(max_iter=max_iter, tol=tol, solver='sag'), targets, feature_names))    
    
    ######################################## Nearest Centroid ########################################
    #models.append(ModelClf("Nearest Centroid", NearestCentroid(), targets, feature_names))    
    
    ######################################## Linear SVC ########################################
    models.append(ModelClf("Linear SVC", LinearSVC(penalty='l2', dual=False, tol=tol), targets, feature_names))
    
    ######################################## Naive Bayes ########################################
    pipe = Pipeline([('clf', None)])
    
    param_grid = [
        {
            'clf' : [MultinomialNB(), BernoulliNB()],
            'clf__alpha' : np.linspace(.1, 1.0, 5)
        }
    ]
    
    grid = GridSearchCV(pipe, param_grid, scoring = scoring, n_jobs=-1)
    models.append(ModelSelectionGrid("Naive Bayes Grid Search", grid, targets, feature_names))
    
    ######################################## Random Forest ########################################
    pipe = Pipeline([('clf', RandomForestClassifier())])
    
    param_grid = {"clf__max_depth": [3, 5, 8, 13, None],
                  "clf__max_features": ['auto', 'sqrt', 'log2'],
                  "clf__min_samples_split": [2, 3, 10],
                  "clf__min_samples_leaf": [1, 3, 10],
                  "clf__bootstrap": [True, False],
                  "clf__criterion": ["gini", "entropy"]}
        
    rand_search = RandomizedSearchCV(pipe, param_grid, scoring = scoring, n_iter=15, n_jobs=-1)
    models.append(ModelSelectionRandSearch("Random forest", rand_search, targets, feature_names)) 
    
    ######################################## SGD ########################################
    pipe = Pipeline([('clf', SGDClassifier(max_iter=max_iter, tol=tol))])
    
    param_grid = {"clf__alpha": [1e-4, .1, 1.0],
                  "clf__penalty": ['l2', 'elasticnet']}
    
    grid = GridSearchCV(pipe, param_grid, scoring = scoring, n_jobs=-1)
    #models.append(ModelSelectionGrid("SGD", grid, targets, feature_names))
    
    ######################################## logistic regression aka Maximum Entropy ########################################
    pipe = Pipeline([('clf', LogisticRegression(tol=tol))])
    
    param_grid = {"clf__C": [1e-4, .1, 1.0, 1e3],
                  "clf__solver": ['newton-cg', 'lbfgs', 'sag', 'saga']}
    
    grid = GridSearchCV(pipe, param_grid, scoring = scoring, n_jobs=-1)
    #models.append(ModelSelectionGrid("Maximum Entropy", grid, targets, feature_names))
    

    """ EXECUTE """        
    for model in models:
        print("Execute %s" % model.getName())
        
        t_start = time()
        
        # train + test
        model.fit(X_train, y_train)
        model.predict_and_analyse(X_test, y_test)
        
        # print report
        fn = FN_BASE % (model.getName(), FILE_REPORT)
        save(output_path, fn, model.getReport().getText().encode('utf8'))
        
        print("Time passed: %0.3f" % (time() - t_start))
        
    """ PLOTS """
    def save_plt(fn):
        savefig(os.path.join(output_path, fn))
    indices = np.arange(len(models))
    
    model_names = []
    f1_scores = []
    accuracies = []
    training_times = []
    test_times = []
    
    for m in models:
        model_names.append(m.name)
        f1_scores.append(m.f1_weighted)
        accuracies.append(m.accuracy)
        training_times.append(m.training_time)
        test_times.append(m.prediction_time)
    
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size' : '14'}
    
    rc('font', **font)
    
    plt.figure(figsize=(12, 10))
    plt.title("Score")
    plt.barh(indices, f1_scores, .2, label=scoring, color='navy')
    plt.barh(indices + .3, accuracies, .2, label="accuracy", color='green')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.35)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)
    
    for i, c in zip(indices, model_names):
        plt.text(-.2, i, c)
        
    save_plt('score.png')
    """
    plt.figure(figsize=(12, 10))
    plt.title("Training/Test time")
    plt.barh(indices, training_times, .2, label="training time", color='c')
    plt.barh(indices + .3, test_times, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.3)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)
    
    for i, c in zip(indices, model_names):
        plt.text(-.027, i, c)
    
    save_plt('training_test_times.png')
    """
    plt.show()
    
def load(path):
    """ LOAD DATA """
    X_train = load_sparse(os.path.join(path, FILE_DATA_X_TRAIN))
    X_test = load_sparse(os.path.join(path, FILE_DATA_X_TEST))
    y_train = load_array(os.path.join(path, FILE_DATA_Y_TRAIN))
    y_test = load_array(os.path.join(path, FILE_DATA_Y_TEST))
    voc = load_json_unicode(os.path.join(path, FILE_DATA_VOC))
    targets = load_json(os.path.join(path, FILE_DATA_TARGETS))
    
    return X_train, X_test, y_train, y_test, voc, targets

def main():
    
    """ PARSE ARGUMENTS """
    parser = ArgumentParser(description = "Analyse various models")
    
    parser.add_argument('--vec_data_dir', help = 'Name of the directory where the input data is stored.')
    
    parser.add_argument('--result_dir', help = 'Name of the directory where the results are saved')
        
    parser.add_argument('--scoring', default = 'f1_macro')
    
    parser.add_argument('--chi2', default = None, help='percentage of selected features, using chi2')
            
    args = parser.parse_args()
    
    scoring = args.scoring
    chi = args.chi2
    
    if chi is not None:
        chi = int(chi)
    
    input_paths = []
    output_paths = []
    labels = []
    
    if args.vec_data_dir == 'all':
        for name in os.listdir(DIR_VECTORIZED_DATA):
            sub_dir = os.path.join(DIR_VECTORIZED_DATA, name)
            if os.path.isdir(sub_dir):
                input_paths.append(sub_dir)
                output_paths.append(os.path.join(DIR_RESULTS, *[args.result_dir, name]))
                labels.append(name)
    else:
        input_paths.append(os.path.join(DIR_VECTORIZED_DATA, args.vec_data_dir))
        output_paths.append(os.path.join(DIR_RESULTS, args.result_dir))
        labels.append(args.vec_data_dir)

    for path_in, path_out, label in zip(input_paths, output_paths, labels):
        X_train, X_test, y_train, y_test, voc, targets = load(path_in)
        analyse(X_train, X_test, y_train, y_test, voc, targets, chi, scoring, path_out, label)
        
    
if __name__ == '__main__':
    main()