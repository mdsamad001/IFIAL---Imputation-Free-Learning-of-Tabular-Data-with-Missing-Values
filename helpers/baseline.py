from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.base import clone as sk_clone

from joblib import Parallel, delayed

from itertools import product

import numpy as np


common_classifiers = {
    'Logistic Regression': {
        'model': LogisticRegression (random_state = 42, class_weight = "balanced", max_iter=300, solver='liblinear'),
        'params': {
            'C': [0.8, 0.5, 1, 5, 0.01,0.05], 
            'penalty': ['l1', 'l2']
        },
    },
    'Random Forest': {
        'model': RandomForestClassifier(n_estimators= 5,  class_weight = "balanced",random_state=42),
        'params': {
            'n_estimators': list(range(10, 120, 20))
            },
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'params': {
              "min_samples_split": [2, 10, 20],
              "max_depth": [2, 5, 10]
            },
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'params': {
            "n_estimators":[50, 80, 110],
            # "min_samples_split": [2, 5, 10, 15, 20],
            # "learning_rate":[0.1, 0.3, 1],
            "max_depth": [2, 5, 10, 15]
        },
    },
    'SVM Linear': {
        'model': SVC(kernel='linear', random_state=42, class_weight = "balanced", probability= True),
        'params': {
            'C': [0.8, 0.5, 0.1, 0.05, 0.01]
        },
    },
    'SVM RBF': {
        'model': SVC(kernel='rbf', random_state=42,cache_size=20000, class_weight = "balanced", probability= True),
        'params': {
            'C': [0.8, 0.5, 0.1, 0.05, 0.01], 
            'gamma': [0.1, 0.02, 0.3, 0.5, 0.05, 0.01]
            },
    },

}

def set_random_seeds():
    np.random.seed(0)


def train_and_test(clf, params_range, train, valid, test, pbar=False, n_jobs=4):
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x, test_y = test
    
    best_model = False
    best_params = False
    best_val = 0

    val_score = []
    param_combinations = list(product(*params_range.values()))
    
    pbar and pbar.set_description(f'Validating {clf.__class__.__name__} : {len(param_combinations)} param combinations')
    
    def check_param(params):
        params = {k:v for k,v in zip(params_range.keys(), params)}

        # clone to get the unfitted yet a true copy of the classifier
        # didn't change the output
        set_random_seeds()
        model =  sk_clone(clf)
        model.set_params(**params)


        model.fit(train_x, train_y)


        y_pred = model.predict(valid_x)
        score = f1_score(valid_y, y_pred, average='weighted')
        
        return score

    val_score = Parallel(n_jobs=n_jobs)(delayed(check_param)(p) for p in param_combinations)
    best_id = np.argmax(val_score)
    
    best_params = {k:v for k,v in zip(params_range.keys(), param_combinations[best_id])}
    
    best_model = sk_clone(clf)
    best_model.set_params(**best_params)
    best_model.fit(train_x, train_y)
    y_pred = best_model.predict(test_x)

    score = f1_score(test_y, y_pred, average='weighted')
    
    return score, best_params, y_pred 