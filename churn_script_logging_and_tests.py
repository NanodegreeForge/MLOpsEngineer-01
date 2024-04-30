'''
Using features exported from churn_library. Test, and Run the Customer Churn analysis. 

Author: Nguyen Chi Bach
Creation Date: 04/20/2024
'''

import logging
import pytest
from churn_library import *

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.mark.parametrize("file", ["data/bank_data.csv"])
def test_import(file):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = import_data(file)
        logging.info("Testing import_data: SUCCESS")
      
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.mark.parametrize("file", ["data/bank_data.csv"])
def test_eda(file):
    '''
    test perform eda function
    '''
    try:
        dataframe = import_data(file)
        perform_eda(dataframe)
        # Successfully exported images
        assert os.path.exists('./images/eda/Churn.png')
        assert os.path.exists('./images/eda/Customer_Age.png')
        assert os.path.exists('./images/eda/Heatmap.png')
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: FAILED")
        raise err


@pytest.mark.parametrize("file", ["data/bank_data.csv"])
def test_encoder_helper(file):
    '''
    test encoder helper
    '''
    try:
        dataframe = import_data(file)
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        out_dataframe = encoder_helper(dataframe, cat_columns)
        assert len(out_dataframe['Gender_Churn']) > 0 
        logging.info("Testing encoder_helper: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing encoder_helper: FAILED")
        raise err


@pytest.mark.parametrize("file", ["data/bank_data.csv"])
def test_perform_feature_engineering(file):
    '''
    test perform_feature_engineering
    '''
    try:
        dataframe = import_data(file)
        x_train, x_test, y_train, y_test = perform_feature_engineering(dataframe, 'Churn')
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_feature_engineering: FAILED")
        raise err


@pytest.mark.parametrize("file", ["data/bank_data.csv"])
def test_train_models(file):
    '''
    test train_models
    '''
    try:
        dataframe = import_data(file)
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataframe, 'Churn')
        train_models(x_train, x_test, y_train, y_test)
        assert os.path.exists('models/logistic_model.pkl')
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: FAILED")
        raise err


if __name__ == "__main__":
    logging.info("Importing data")
    g_dataframe = import_data("data/bank_data.csv")
    logging.info("Performing EDA")
    perform_eda(g_dataframe)
    logging.info("Performing Feature Engineering")
    g_x_train, g_x_test, g_y_train, g_y_test = perform_feature_engineering(
        g_dataframe, 'Churn')
    logging.info("Training Model")
    train_models(g_x_train, g_x_test, g_y_train, g_y_test)
    logging.info("Training Completed!")
