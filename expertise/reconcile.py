import numpy as np 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def compute_disagreement(X,model_1,human_preds,epsilon,patches):
    """Compute the fraction of points models disagree on
    
    Arguments:
        X: All available features, numpy array
        model_1: f_hat, Sklearn model
        human_preds: Numpy array of predictions
        epsilon: Error tolerance, float
        patches: 2xN numpy list, patches for AI and for humans
    
    Returns: Float, rate of disagreement between the two predictions"""

    y_pred_1 = model_1.predict(X)+patches[0]
    y_pred_2 = human_preds + patches[1] 
    return np.sum(np.abs(y_pred_1-y_pred_2)>epsilon)/len(X)

def get_valid_points(X,model_1,human_preds,epsilon,y_hat_low,y_hat_high,greater,patches):
    """Compute the set of training data points where we disagree > epsilon
        and where model_1 > human (or vice versa) AND y_hat_low <= human_preds <= y_hat_high
    
    Arguments:
        X: All available features, numpy array
        model_1: f_hat, Sklearn model
        human_preds: Numpy array of predictions
        epsilon: Error tolerance, float
        y_hat_low: Float, lower bound for the human prediction
        y_hat_high: Float, upper bound for the human prediction
        patches: 2xN numpy list, patches for AI and for humans
    
    Returns: Float, rate of disagreement between the two predictions"""

    y_pred_1 = model_1.predict(X)+patches[0]
    y_pred_2 = human_preds+patches[1]
    valid_points = (np.abs(y_pred_1-y_pred_2)>epsilon)&(y_hat_low <= human_preds) &(y_hat_high >= human_preds)
    if greater:
        valid_points &= (y_pred_1 > y_pred_2)
    else:
        valid_points &= (y_pred_1 < y_pred_2)
    return valid_points


def v_star(X,Y,model_1,human_preds,epsilon,y_pred_low,y_pred_high,greater,patches):
    """Compute the average among all points in get_valid_points, for the optimal label Y
    
    Arguments:
        X: All available features, numpy array
        model_1: f_hat, Sklearn model
        human_preds: Numpy array of predictions
        epsilon: Error tolerance, float
        y_hat_low: Float, lower bound for the human prediction
        y_hat_high: Float, upper bound for the human prediction
        patches: 2xN numpy list, patches for AI and for humans
    
    Returns: Float, rate of disagreement between the two predictions"""

    return np.mean(Y[get_valid_points(X,model_1,human_preds,epsilon,y_pred_low,y_pred_high,greater,patches)])

def v_i(X,Y,model_1,human_preds,epsilon,y_pred_low,y_pred_high,greater,model_num,patches):
    """Compute the average among all points in get_valid_points, for the generated model
        Either human or AI
    
    Arguments:
        X: All available features, numpy array
        model_1: f_hat, Sklearn model
        human_preds: Numpy array of predictions
        epsilon: Error tolerance, float
        y_hat_low: Float, lower bound for the human prediction
        y_hat_high: Float, upper bound for the human prediction
        patches: 2xN numpy list, patches for AI and for humans
    
    Returns: Float, rate of disagreement between the two predictions"""
    
    if model_num == 0:
        y_pred = model_1.predict(X)+patches[0]
    else:
        y_pred =  human_preds+patches[1]
    return np.mean(y_pred[get_valid_points(X,model_1,human_preds,epsilon,y_pred_low,y_pred_high,greater,patches)])

def get_diff(X,y,model_1,human_preds,epsilon,y_hat_low,y_hat_high,greater,model_num,patches):
    """Compute the average among all points in get_valid_points, for the generated model
        Either human or AI
    
    Arguments:
        X: All available features, numpy array
        model_1: f_hat, Sklearn model
        human_preds: Numpy array of predictions
        epsilon: Error tolerance, float
        y_hat_low: Float, lower bound for the human prediction
        y_hat_high: Float, upper bound for the human prediction
        greater: Boolean, whether model_1 > human_preds or vice versa
        model_num: {0,1} either looking at AI (0) or human (1) predictions
        patches: 2xN numpy list, patches for AI and for humans
    
    Returns: Float, rate of disagreement between the two predictions"""
    

    valid_points = get_valid_points(X,model_1,human_preds,epsilon,y_hat_low,y_hat_high,greater,patches)
    if model_num == 0:
        y_pred = model_1.predict(X)+patches[0]
    else:
        y_pred = human_preds+patches[1]
    return np.mean(y[valid_points])-np.mean(y_pred[valid_points])

def compute_train_patches(X_train,human_train,y_train,f_hat,epsilon,T,y_divisions,m = 10):
    """Run the reconcile procedure over a training dataset to learn patches
    
    Arguments:
        X_train: All available features, numpy array
        human_train: Human predictions/labels
        y_train: True labels, numpy array
        f_hat: Sklearn model, AI prediction
        epsilon: Float, error tolerance
        T: Integer, number of iteartions to run for
        y_divisions: List of tuples, how to divide up y when running with unknown human labels
    
    Returns: Float, rate of disagreement between the two predictions"""
    
    patches = np.array([[0.0 for i in range(len(X_train))] for j in range(2)])

    for i in range(T):
        all_vals = []
        corresponding_pairs = [(g,i,y[0],y[1]) for g in [True,False] for i in [0,1] for y in y_divisions]
        for greater in [True,False]:
            for i in [0,1]:
                for y_low,y_high in y_divisions:
                    val = sum(get_valid_points(X_train,f_hat,human_train,epsilon,y_low,y_high,greater,patches))/len(X_train)
                    if np.isnan(val):
                        val = 0
                    if val>0:
                        val *= (v_star(X_train,y_train,f_hat,human_train,epsilon,y_low,y_high,greater,patches)-v_i(X_train,y_train,f_hat,human_train,epsilon,y_low,y_high,greater,i,patches))**2
                    if np.isnan(val):
                        val = 0
                    all_vals.append(val)
        g,i,y_low,y_high = corresponding_pairs[np.argmax(all_vals)]
        valid_points = get_valid_points(X_train,f_hat,human_train,epsilon,y_low,y_high,g,patches)
        diff = get_diff(X_train,y_train,f_hat,human_train,epsilon,y_low,y_high,g,i,patches)
        if not np.isnan(diff):
            patches[i][valid_points == True] += round(diff*m)/m
        else:
            break
    
    return patches 

def evaluate_test_patches(X_train,X_test,human_test,y_test,f_hat,patches):
    """Compute the performance on a test dataset given a set of patches
    
    Arguments:
        X_train: Features from the training data, numpy array
        X_test: Features from the testing data, numpy array
        human_test: Numpy array, list of human labels
        y_test: Numpy array, true labels
        y_test: True labels at test time
        patches: 2xN with patches for each data point
    
    Returns: Mean squared error on the test set"""

    fit_0 = GradientBoostingRegressor().fit(X_train,patches[0])
    fit_1 = GradientBoostingRegressor().fit(X_train,patches[1])

    test_patches = np.array([fit_0.predict(X_test),fit_1.predict(X_test)])
    predictions = np.mean([f_hat.predict(X_test)+test_patches[0],human_test+test_patches[1]],axis=0)
    return mean_squared_error(predictions,y_test)

def evaluate_test_patches_with_human(X_train,human_train,X_test,human_test,y_test,f_hat,patches):
    """Compute the performance on a test dataset given a set of patches
    
    Arguments:
        X_train: Features from the training data, numpy array
        human_train: Numpy array, list of human train labels
        X_test: Features from the testing data, numpy array
        human_test: Numpy array, list of human labels
        y_test: Numpy array, true labels
        y_test: True labels at test time
        patches: 2xN with patches for each data point
    
    Returns: Mean squared error on the test set"""
    

    fit_0 = GradientBoostingRegressor().fit(np.hstack([X_train,human_train.reshape(-1,1)]),patches[0])
    fit_1 = GradientBoostingRegressor().fit(np.hstack([X_train,human_train.reshape(-1,1)]),patches[1])

    test_patches = np.array([fit_0.predict(np.hstack([X_test,human_test.reshape(-1,1)])),fit_1.predict(np.hstack([X_test,human_test.reshape(-1,1)]))])
    predictions = np.mean([f_hat.predict(X_test)+test_patches[0],human_test+test_patches[1]],axis=0)
    return mean_squared_error(predictions,y_test)