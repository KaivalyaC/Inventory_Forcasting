import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error

# Performance parameter matrix for train and test
def trainTestMetrics(X_train,X_test,y_train,y_test,model):
    # In the funtion we are considering MAE as the performance parameter. Along with MAE we can use r2, adj r2, MAPE etc

    names=['MAE'] 
    
    # Training Metrics
    y_hat = model.predict(X_train)
    
    train_metrics = [mean_absolute_error(y_train,y_hat).round(3)]
    train_metrics = pd.DataFrame({'Train':train_metrics},index=names)
    
    # Testing Metrics
    y_hat = model.predict(X_test)
    test_metrics = [mean_absolute_error(y_test,y_hat).round(3)]
    test_metrics = pd.DataFrame({'Test':test_metrics},index=names)
    
    all_metrics = train_metrics.merge(test_metrics,left_index=True,right_index=True)
    # The matrix above will contain the MAE for train as well as test
    print(all_metrics)
    
# Load the data
def loadData(path):
    # load the data from given file path in df
    df = pd.read_csv(path)
    # drop the column titled 'Unnamed :0'
    df.drop(['Unnamed: 0'], inplace=True, axis = 1)
    
# Create target variable and predictor variables
def create_target_and_predictors(df,target):
    X = data.drop(columns=[target])
    y = data[target]
    return X, y
    
# Train algorithm
def trainLogisticRegModel(X,y):
    # Split the dataset into train and test 
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=2)
    # Create a regression model and fit the train data
    model = LinearRegression()
    model.fit(X_train,y_train)
    # Check the model performance for train and test data
    trainTestMetrics(X_train,X_test,y_train,y_test,model)
    
# Execute training pipeline
def run():
    # Load the data first
    df = load_data('sales.csv')

    # Now split the data into predictors and target variables
    X, y = create_target_and_predictors(data=df)

    # Finally, train the machine learning model
    train_algorithm_with_cross_validation(X=X, y=y)