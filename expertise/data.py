import pandas as pd
from functools import reduce

def load_feature_data():
    """Load the Train and Test X, Y, and ID data

    Arguments: None 

    Returns: Tuple with 6 things, x_train, y_train, x_train_id, x_test, y_test, y_test_id"""

    h = pd.read_csv('../../data/visual_prediction/human_responses.csv')

    X = pd.read_csv('../../data/visual_prediction/features.csv')
    data = X.drop(['escaped', 'location', 'room'], axis=1)
    escaped = X['escaped'].map({'Y': 1, 'N': 0})

    x_train = data[~data['file_id'].isin(h['Img'])]
    y_train = escaped[~data['file_id'].isin(h['Img'])]

    x_test = data[data['file_id'].isin(h['Img'])]
    y_test = escaped[data['file_id'].isin(h['Img'])]


    x_train_id = x_train['file_id']
    x_train = x_train.drop(['file_id'], axis = 1)
    x_test_id = x_test['file_id']
    x_test = x_test.drop('file_id', axis = 1)

    return x_train, y_train, x_train_id, x_test, y_test, x_test_id

def load_all_predictors():
    """Load all the human + machine
    
    Arguments: None
    
    Returns: Pandas Dataframe with columns for the prediction from each predictor"""


    h = pd.read_csv('../../data/visual_prediction/human_responses.csv')
    
    input_data = (
    h.drop(columns=['TurkerID'])  
    .pivot(index='Img', columns='Condition', values='Esc') 
    .dropna()
    .reset_index()  
    .rename(columns={
        'Img': 'file_id',
        'control': 'human.pred.c',
        'training4': 'human.pred.t4',
        'training8': 'human.pred.t8',
        'training12': 'human.pred.t12'
    }) 
    .rename(columns={'true_esc': 'y'})
    )
    input_data.iloc[:, 1:] = (input_data.iloc[:, 1:] == 'Y').astype(int)

    model_dfs = [pd.read_csv("../../data/visual_prediction/{}.csv".format(file)) for file in ["random_forest","naive_bayes","logistic_regression","linear_svm","gradient_boosting"]]
    models = reduce(lambda left, right: pd.merge(left, right, on="file_id", how="left",suffixes=("", "_dup")), model_dfs)
    models = pd.DataFrame(models)
    models.iloc[:, 1:] = models.iloc[:, 1:].applymap(lambda y: 1 if y == 'Y' else 0)
    models.columns = (["file_id","pred_esc_random_forest","true_esc","pred_esc_naive bayes","true_esc","pred_esc_logistic_regression","true_esc","pred_esc_linear_svm","true_esc","pred_esc_gradient_boosting","true_esc"])
    models = models.T.drop_duplicates().T
    tidy = pd.merge(input_data,models,on="file_id")

    return tidy