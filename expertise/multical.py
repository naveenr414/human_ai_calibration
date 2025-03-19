from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def compute_accuracy_cluster(train_dataframe,test_dataframe,human_column,reference_column):
    """Compute the accuracy for a dataframe between
        the maximum predictor from a human column -> reference column"""

    X_train = train_dataframe[[human_column]].to_numpy()
    y_train = train_dataframe[reference_column].to_numpy()
    X_test = test_dataframe[[human_column]].to_numpy()
    y_test = test_dataframe[reference_column].to_numpy() 

    if len(set(y_train)) == 1:
        predictions = np.array([y_train[0] for i in range(len(X_test))])
    else:
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

def assess_callibration(dataframe,human_column,cluster_column,reference_column):
    """Assess a given multicallibration = clustering in a given dataframe
    
    Arguments:
        dataframe: Pandas Dataframe with all the information
        human_column: String, which column contains human assessments
        cluster_column: Which column contains the clusters we use
        reference_column: True label column"""

    dataframe[reference_column] = pd.to_numeric(dataframe[reference_column], errors='coerce')
    dataframe[human_column] = pd.to_numeric(dataframe[human_column], errors='coerce')

    train_dataframe, test_dataframe = train_test_split(dataframe, test_size=0.5, random_state=42)

    avg_acc = 0
    cluster_vals = set(train_dataframe[cluster_column])
    for val in cluster_vals:
        clean_train_data = train_dataframe[train_dataframe[cluster_column]==val][[reference_column, human_column]].dropna()
        clean_test_data = test_dataframe[test_dataframe[cluster_column]==val][[reference_column, human_column]].dropna()
        avg_acc += len(clean_test_data)/len(test_dataframe)*compute_accuracy_cluster(clean_train_data,clean_test_data,human_column,reference_column)

    return avg_acc