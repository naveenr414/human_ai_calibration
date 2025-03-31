import expertise.LSBoost as LSBoost

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def train_boosted(x_train,y_train,x_test,max_depth,global_gamma=0.01):
    """Train a boosted Decision Tree using x_train and y_train values
    
    Arguments:
        x_train: DataFrame of training points
        y_train: Labels for each training point
        max_depth: Integer, maximum depth of the DecisionTree
    
    Returns: List of [0,1] predictions from the LSBoost object"""

    T = 100
    num_bins = 10
    min_group_size = 5
    weak_learner = DecisionTreeRegressor(max_depth=max_depth)
    bin_type = 'default'
    learning_rate = 1
    initial_model = None
    final_round = False
    center_mean = False

    LSBoostReg = LSBoost.LSBoostingRegressor(
                                    T = T, 
                                    num_bins = num_bins, 
                                    min_group_size = min_group_size, 
                                    global_gamma = global_gamma, 
                                    weak_learner= weak_learner, 
                                    bin_type = bin_type, 
                                    learning_rate = learning_rate, 
                                    initial_model = initial_model,  
                                    final_round = final_round, 
                                    center_mean=center_mean)
    LSBoostReg.fit(x_train.values, y_train.values)
    
    return LSBoostReg.predict(x_test.values)

def train_adaboost(x_train,y_train,x_test,max_depth):
    """Train a boosted Decision Tree using x_train and y_train values
    
    Arguments:
        x_train: DataFrame of training points
        y_train: Labels for each training point
        max_depth: Integer, maximum depth of the DecisionTree
    
    Returns: List of [0,1] predictions from the LSBoost object"""

    weak_learner = DecisionTreeRegressor(max_depth=max_depth)
    model = AdaBoostRegressor(base_estimator=weak_learner, n_estimators=50)
    model.fit(x_train.values, y_train.values)

    return model.predict(x_test.values)


def train_regression_tree(x_train,y_train,x_test,random_state):
    """Get predictions for a regression tree
    
    Arguments:
        x_train: List of X training data points
        y_train: List of labels for the training dataset
        x_test: Features for the test
        random_state: Seed so we can vary randomness for Decision Trees

    Returns: 0-1 list of predictions, pred"""

    tree = DecisionTreeRegressor(random_state=random_state)
    tree.fit(x_train, y_train)
    pred = tree.predict(x_test)
    return pred 

def train_regression_tree(x_train,y_train,x_test,random_state):
    """Get predictions for a regression tree
    
    Arguments:
        x_train: List of X training data points
        y_train: List of labels for the training dataset
        x_test: Features for the test
        random_state: Seed so we can vary randomness for Decision Trees

    Returns: 0-1 list of predictions, pred"""

    tree = DecisionTreeRegressor(random_state=random_state)
    tree.fit(x_train, y_train)
    pred = tree.predict(x_test)
    return pred 

def train_random_predictor(y_test,accuracy):
    """Make predictions that match y_test with test_accuracy%
    
    Arguments:
        y_test: 0-1 predictions on the test dataset
        accuracy: Target accuracy of the predictor"""

    n = len(y_test)
    n_correct = int(accuracy * n)
    n_incorrect = n - n_correct
    
    indices = np.random.permutation(n)
    correct_indices = indices[:n_correct]  
    incorrect_indices = indices[n_correct:]
    
    rand_predictions = np.zeros(n, dtype=int)
    rand_predictions[correct_indices] = y_test[correct_indices]  # Correct predictions
    rand_predictions[incorrect_indices] = 1 - y_test[incorrect_indices]  # Incorrect predictions
    
    return rand_predictions

def predict_aggregate(prediction_list,num_clusters):
    """Use KMeans to aggregate multiple predictors
        into different clusters 
    
    Argumnets:
        prediction_list: Numpy array of 0-1 predictions, where each column
            is a different predictor
    
    Returns: List of different clusters"""

    kmeans = KMeans(n_clusters=num_clusters)
    return kmeans.fit_predict(prediction_list)


def create_predicted_dataframe(test_predictions,x_test_id,num_bins):
    """Turn a list of predictions into a dataframe + set of bins
        which can be used for mutlicallibration
    
    Arguments:
        test_predictions: [0,1] predictions for each file
        x_test_id: List of corresponding files/data points
        num_bins: How to chunk up the test_predictions
    
    Returns: Dataframe with file_id, mc_pred, and bin"""

    bins = np.ceil(test_predictions * num_bins).astype(int)

    m = pd.DataFrame({'file_id': x_test_id, 'mc_pred': test_predictions, 'bin': bins})
    return m
