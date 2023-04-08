import argparse
import os
import datetime as dt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pandas as pd
import joblib
from io import StringIO

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model

def predict_fn(input_object, model):
    y_pred = model.predict_proba(input_object)[0][1]
    return y_pred

def input_fn(request_body, request_content_type):
    print(request_body)
   
    df = pd.read_csv(StringIO(request_body), header=None)
   
    print(df)
   
    return df.to_numpy()

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    args, _ = parser.parse_known_args()
    train = pd.read_csv('{}/train.csv'.format(args.train), header=None)
    test = pd.read_csv('{}/test.csv'.format(args.test), header=None)
    
    
    
    X_train = train.iloc[:, 1:]
    y_train = train.iloc[:, 0]
    
    X_test = test.iloc[:, 1:]
    y_test = test.iloc[:, 0]
    
    
    
    X_train = pd.to_datetime(X_train)
    X_train = X_train.map(dt.datetime.toordinal)
    
    X_test = pd.to_datetime(X_test)
    X_test = X_test.map(dt.datetime.toordinal)
    
    model = LinearRegression()
    
    model.fit(X_train, y_train)
    
    
    y_test_predict = model.predict(X_test)
    
    puntuation = r2_score(y_test, y_test_predict)
    mae = mean_absolute_error(y_test, y_test_predict)
    mse = mean_squared_error(y_test, y_test_predict)
    
    print(f"Coeficientes del modelo: {model.coef_}")
    print(f"Intresección del modelo: {model.intercept_}")
    print(f"Número de coeficientes del modelo: {len(model.coef_)}")
    
    print(f"Score r2: {puntuation}")
    print(f"Score mae: {mae}")
    print(f"Score mse: {mse}")
    
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib') )
    
    print(train)
