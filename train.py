from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score
import pickle
import time



pipelines = {
    'lr' : make_pipeline(StandardScaler(), LogisticRegression()),
    'rc' : make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf' : make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb' : make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

def train(loading_label, file_name):

    df = pd.read_csv(file_name)
    X = df.drop('action', axis=1) #feature
    y = df['action'] #target 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    print(X_train, y_train)

    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model

    with open('body_languageTEST.pkl', 'wb') as f:
        pickle.dump(fit_models['lr'], f)
    
    loading_label.config(text="")