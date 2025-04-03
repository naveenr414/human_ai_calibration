import pandas as pd 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np

def subsample_rows(predictions,subsampled_data,all_test_ids):
    """Take the rows from predictions that correspond to file_ids 
        in subsampled data, where all_test_ids contains the 
        file_ids for each prediction
    
    Arguments:
        predictions: List of 0-1 predictions
        subsampled_data: Dataframe with a column file_ids
        all_test_ids: All file_ids corresponding to predictions
    
    Returns: List of 0-1 predictions"""

    df = pd.DataFrame({'x_test_id': all_test_ids, 'y': predictions})
    return pd.DataFrame({'file_id': subsampled_data["file_id"]}).merge(df, left_on='file_id', right_on='x_test_id', how='left')['y'].to_numpy()

def generate_side_information_data(train_N,test_N,x_d,side_d,true_predictor_weights=None,human_predictor_weights=None):
    """Generate data with dimension train_N x (x_d+side_d),
        where x_d is the known features, and side_d is the unknown side features
    
    Arguments:
        train_N: Integer, number of training data points
        test_N: Integer, number of testing data points
        x_d: Integer, dimension of the training data that's known
        side_d: Integer, dimension of the training data that's side info
        true_predictor: None by default, but can be a list of size x_d+side_d+1
        human_predictor: None by default, but can be a list of size x_d+side_d+1"""


    X_train = np.random.random((train_N,x_d))
    X_test = np.random.random((test_N,x_d))

    c_train = np.random.random((train_N,side_d))
    c_test = np.random.random((test_N,side_d))

    bias_train = np.ones((train_N,1))
    bias_test = np.ones((test_N,1))

    if true_predictor_weights is None:
        true_predictor_weights = np.random.random(x_d+side_d+1)
    
    y_train = np.concatenate([X_train.T,c_train.T,bias_train.T]).T.dot(true_predictor_weights)
    y_test = np.concatenate([X_test.T,c_test.T,bias_test.T]).T.dot(true_predictor_weights)

    if human_predictor_weights is None:
        human_predictor_weights = np.random.random(x_d+side_d+1)

    human_train = np.concatenate([X_train.T,c_train.T,bias_train.T]).T.dot(human_predictor_weights)
    human_test = np.concatenate([X_test.T,c_test.T,bias_test.T]).T.dot(human_predictor_weights)
    
    return X_train, human_train, y_train, X_test, human_test, y_test