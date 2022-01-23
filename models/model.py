from time import time

from sklearn import metrics
from sklearn.base import clone

from analytics.Report import Report
import numpy as np


TOP_N_FEATURES_TO_REPORT = 20

class ModelBase():        
    def __init__(self, name, clf, targets, feature_names):
        
        self.name = name
        self.clf = clf
        self.targets = targets
        self.feature_names = feature_names
        
        self.report = Report()
        
        self.f1_weighted = None
        self.accuracy = None
        self.training_time = None
        self.prediction_time = None
                
    def fit(self, X, y):
        self.report.addSeperator()   
        
        name = self.name
        
        self.report.addLn("Fit model %s" % name)

        t_start = time()
        
        self.report.addLn("Train classifier")
        
        """ EXECUTE """
        self.clf.fit(X,y)
        
        t_train = time() - t_start
        
        self.training_time = t_train
        
        self.report.addLn("Train time: %0.3f" % t_train)
        
        self.report_top_n_features()
        
        
    def predict_and_analyse(self, X, y):
        clf = self.clf
        
        t_start = time()
        
        y_pred = clf.predict(X)
        
        self.prediction_time = time() - t_start
        self.accuracy = metrics.accuracy_score(y, y_pred)
        self.f1_weighted = metrics.f1_score(y, y_pred, average='weighted')
        
        self.report.addSeperator()
        self.report.addLn("Prediction time: %0.3f" % self.prediction_time)
        self.report.addLn("accuracy:   %0.3f" % self.accuracy)
        self.report.addLn("f1_weighted:   %0.3f" % self.f1_weighted)
        self.report.addLn("classification report:")
        self.report.addLn(metrics.classification_report(y, y_pred, target_names=self.targets.keys()))
        self.report.addLn("confusion matrix:")
        self.report.addLn(str(metrics.confusion_matrix(y, y_pred)))
        
    def report_top_n_features(self, n=TOP_N_FEATURES_TO_REPORT):
        
        coefs = self.getCoefs()
        
        if coefs is None:
            self.report.addLn('The classifier does not expose '
                               '"coef_" or "feature_importances_" '
                               'attributes')
        else:
            self.report.addLn("Top %d features: " % n)
            if coefs.shape[0] == 1:
                top_n = np.argsort(np.squeeze(coefs))[-n:]
                self.report.addLn("%s" % (", ".join(self.feature_names[top_n])))
            else:
                for name, value in self.targets.iteritems():
                    top_n = np.argsort(coefs[value])[-n:]
                    self.report.addLn("%s: %s" % (name, ", ".join(self.feature_names[top_n])))
            
    def getCoefs(self):
        clf = self.clf
        coefs = None
        
        if hasattr(clf, 'coef_'):
            coefs = clf.coef_
        else:
            coefs = getattr(clf, 'feature_importances_', None)
        
        return coefs

    def getReport(self):
        return self.report
    
    def getName(self):
        return self.name
    
class ModelClf(ModelBase):
    def __init__(self, name, clf, targets, feature_names):
        ModelBase.__init__(self, name, clf, targets, feature_names)

    def fit(self, X, y):
        ModelBase.fit(self, X, y)
            
class ModelSelectionBase(ModelBase):
    def __init__(self, name, clf, targets, feature_names):
        ModelBase.__init__(self, name, None, targets, feature_names)
        
        self.model_selector = clf
        
    def fit(self, X, y):
        model_selector = self.model_selector
        
        model_selector.fit(X,y)
        
        clf_best = model_selector.best_estimator_._final_estimator
        
        #refit for training time analysis
        clf = clone(clf_best)
        
        t_start = time()
        clf.fit(X,y)
        training_time = time() - t_start
        
        self.report.addLn("Best model:")
        self.report.addLn(str(self.clf))
        self.report.addLn("Training time: %d" % training_time)
        
        self.clf = clf
        self.training_time = training_time
        
        self.report_top_n_features()
                    
class ModelSelectionGrid(ModelSelectionBase):
    def __init__(self, name, clf, targets, feature_names):
        ModelSelectionBase.__init__(self, name, clf, targets, feature_names)

    def fit(self, X, y):
        self.report.addSeperator()
        self.report.addLn("Using grid search for model selection")
        self.report.addLn("Param grid:")
        self.report.addLn(str(self.model_selector.param_grid))
        
        ModelSelectionBase.fit(self, X, y)
    
class ModelSelectionRandSearch(ModelSelectionBase):
    def __init__(self, name, clf, targets, feature_names):
        ModelSelectionBase.__init__(self, name, clf, targets, feature_names)

    def fit(self, X, y):
        self.report.addSeperator()
        self.report.addLn("Using randomized Search for model selection")
        self.report.addLn("Param distribution:")
        self.report.addLn(str(self.model_selector.param_distributions))
                
        ModelSelectionBase.fit(self, X, y)
        
        
        
        
        
        
        
        
        
        