# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Optimizing the analysis and ML model training of Customer Churn Prediction. Including the following steps. 
1. EDA
2. Feature Engineering (including encoding of categorical variables)
3. Model Training
4. Prediction
5. Model Evaluation

## Files and data description
Overview of the files and data present in the root directory. 
- data/ 
    - bank_data.csv (Dataset)
- images/
    - eda (Contains images output from the EDA process).
    - results (Contains classifications, feature importances, and ROC output from model training)
- logs/
    - churn_library.log (Log file output when run the script)
- models/ (Contains pickle files for the trained models)
- churn_library.py (Export Functions used to do each of the data analysis and model training process)
- churn_script_loging_and_tests.py (Test and Run the project)
- requirements_py3.6.txt (python dependencies definition)


## Running Files
Run on Python 3.6.3 or later and install the requirements.
```
python -m pip install -r requirements_py3.6.txt
```

To run the script with output logs to `logs/churn_library.log`
```
ipython churn_script_logging_and_tests.py
```

To run the tests using pytest
```
pytest churn_script_logging_and_tests.py
```


