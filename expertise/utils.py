import pandas as pd 

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